"""Reference-tag motion history expressed in the headset's cam_0 frame."""

import csv

import cv2
import numpy as np


class ReferenceTrail:
    def __init__(self, poses, timestamps_us, status, reference_id, seconds):
        self.poses = np.asarray(poses)
        self.timestamps = np.asarray(timestamps_us, dtype=np.int64)
        self.status = np.asarray(status, dtype="U12")
        self.reference_id = int(reference_id)
        self.seconds = float(seconds)
        if not np.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Trail duration must be a positive finite number of seconds")
        if self.poses.shape != (len(self.timestamps), 4, 4) or len(self.status) != len(self.timestamps):
            raise ValueError("Trail poses, timestamps and status must have matching frame counts")
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Trail timestamps must increase strictly")
        self.positions = self.poses[:, :3, 3]
        self.valid = np.isin(self.status, ["measured", "flow_only", "interpolated"]) & np.isfinite(self.positions).all(axis=1)

    def segments(self, index):
        start = int(np.searchsorted(self.timestamps, self.timestamps[index] - self.seconds * 1e6))
        segments = []
        for end in range(start + 1, index + 1):
            if not self.valid[end - 1] or not self.valid[end]:
                continue
            # Do not bridge a timestamp gap even when both endpoint poses exist.
            if self.timestamps[end] - self.timestamps[end - 1] > 200000:
                continue
            age = (int(self.timestamps[index]) - int(self.timestamps[end])) * 1e-6
            opacity = .15 + .75 * max(0., 1. - age / self.seconds)
            estimated = "interpolated" in self.status[end - 1:end + 1]
            segments.append((end - 1, end, opacity, estimated))
        return segments

    def draw(self, image, camera, index):
        height, width = image.shape[:2]
        start = int(np.searchsorted(self.timestamps, self.timestamps[index] - self.seconds * 1e6))
        xyz = self.positions[start:index + 1] @ camera.R.T + camera.t
        scale = np.array([width / camera.width, height / camera.height])
        pixels = camera.project(xyz) * scale
        visible = np.isfinite(pixels).all(axis=1) & (np.max(np.abs(pixels), axis=1) < 1e6)
        visible &= np.linalg.norm(xyz, axis=1) > 1e-8
        for first, second, opacity, estimated in self.segments(index):
            a, b = first - start, second - start
            if not visible[a] or not visible[b]:
                continue
            p, q = np.rint(pixels[[a, b]]).astype(int)
            clipped, p, q = cv2.clipLine((0, 0, width, height), tuple(p), tuple(q))
            if not clipped:
                continue
            color = (0, 165, 255) if estimated else (255, 220, 40)
            left, right = max(0, min(p[0], q[0]) - 4), min(width, max(p[0], q[0]) + 5)
            top, bottom = max(0, min(p[1], q[1]) - 4), min(height, max(p[1], q[1]) + 5)
            region = image[top:bottom, left:right]
            overlay = region.copy()
            cv2.line(overlay, (p[0] - left, p[1] - top), (q[0] - left, q[1] - top), color, 3, cv2.LINE_AA)
            cv2.addWeighted(overlay, opacity, region, 1. - opacity, 0, dst=region)
        if self.valid[index] and visible[-1]:
            point = tuple(np.rint(pixels[-1]).astype(int))
            if 0 <= point[0] < width and 0 <= point[1] < height:
                color = (0, 165, 255) if self.status[index] == "interpolated" else (255, 220, 40)
                cv2.circle(image, point, 4, color, -1, cv2.LINE_AA)
        label = f"Trail: reference ID {self.reference_id}, {self.seconds:g}s | cyan: visual, orange: estimated"
        cv2.rectangle(image, (0, height - 27), (width, height), (12, 12, 12), -1)
        cv2.putText(image, label, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1, cv2.LINE_AA)
        return image

    def export(self, path):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "timestamp_us", "reference_tag_id", "coordinate_frame", "status", "x_m", "y_m", "z_m"])
            for index, (timestamp, status) in enumerate(zip(self.timestamps, self.status)):
                xyz = self.positions[index] if self.valid[index] else [np.nan] * 3
                writer.writerow([index, int(timestamp), self.reference_id, "cam_0", str(status), *xyz])
