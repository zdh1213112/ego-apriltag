"""Recover a missing rigid pose by reading tag36h11 cells in synchronized raw frames."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from track_ego_apriltag import STREAMS, fit_rigid_detections, matrix, project_pose


# OpenCV's unrotated dictionary drawing starts at its top left. The native
# AprilTag binding returns these four corners in a different order.
NATIVE_ORDER = np.array([1, 0, 3, 2])
CELL_CORNERS = np.float32([[.5, .5], [8.5, .5], [8.5, 8.5], [.5, 8.5]])
TEMPLATE_CORNERS = (CELL_CORNERS + .5) * 8 - .5


class PatternDecoder:
    def __init__(self, family, tag_ids):
        if family != "tag36h11":
            raise ValueError("Pattern recovery supports tag36h11")
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("Pattern recovery needs OpenCV with the aruco module")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        draw = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker
        self.labels = []
        self.templates = []
        codes = []
        for tag_id in tag_ids:
            if not 0 <= tag_id < len(dictionary.bytesList):
                raise ValueError(f"ID {tag_id} is outside the tag36h11 dictionary")
            marker = draw(dictionary, tag_id, 8)
            for rotation in range(4):
                grid = np.pad(np.rot90(marker, rotation), 1, constant_values=255)
                codes.append(grid[2:-2, 2:-2] > 0)
                self.labels.append((tag_id, rotation))
                self.templates.append(cv2.resize(grid, (80, 80), interpolation=cv2.INTER_NEAREST))
        self.codes = np.stack(codes)
        self.grid = np.dstack(np.meshgrid(np.arange(10), np.arange(10))).astype(np.float32).reshape(-1, 1, 2)
        self.white_border = np.ones((10, 10), dtype=bool)
        self.white_border[1:-1, 1:-1] = False
        self.black_border = np.ones((8, 8), dtype=bool)
        self.black_border[1:-1, 1:-1] = False

    def read_cells(self, gray, quad, max_errors):
        warp = cv2.getPerspectiveTransform(CELL_CORNERS, quad)
        xy = cv2.perspectiveTransform(self.grid, warp).reshape(10, 10, 2)
        if not np.isfinite(xy).all() or (xy < 0).any() or (xy >= np.array(gray.shape[::-1]) - 1).any():
            return None
        samples = cv2.remap(gray, xy, None, cv2.INTER_LINEAR).astype(np.float32)
        black = np.median(samples[1:-1, 1:-1][self.black_border])
        white = np.median(samples[self.white_border])
        if white - black < 28:
            return None
        core = samples[2:-2, 2:-2]
        errors = np.count_nonzero((core > (black + white) / 2) != self.codes, axis=(1, 2))
        best = int(errors.argmin())
        if errors[best] > max_errors:
            return None
        bits = self.codes[best]
        if core[bits].mean() - core[~bits].mean() < 22:
            return None
        correlation = float(np.corrcoef(core.ravel(), bits.ravel())[0, 1])
        if not np.isfinite(correlation) or correlation < .59:
            return None
        return best, int(errors[best]), correlation

    def verify(self, gray, quad, label):
        # Register the printed pattern locally, then read the bits again on
        # the original image. The quality score uses only the 36 data cells.
        warp = cv2.getPerspectiveTransform(TEMPLATE_CORNERS, quad)
        tile = cv2.warpPerspective(gray, warp, (80, 80), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .00015)
        try:
            score, correction = cv2.findTransformECC(self.templates[label], tile, np.eye(3, dtype=np.float32),
                                                      cv2.MOTION_HOMOGRAPHY, criteria, None, 7)
        except cv2.error:
            return None
        refined = cv2.perspectiveTransform(TEMPLATE_CORNERS[:, None], warp @ correction).reshape(4, 2)
        edge = np.mean(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1))
        if (not np.isfinite(score) or score < .65 or not np.isfinite(refined).all()
                or np.max(np.linalg.norm(refined - quad, axis=1)) > .28 * edge):
            return None
        read = self.read_cells(gray, refined, 2)
        if read is None or read[0] != label or read[2] < .70:
            return None
        tag_id, rotation = self.labels[label]
        # OpenCV coordinates have the first pixel centered at zero. The native
        # detector reports its center at half a pixel.
        return dict(tag_id=tag_id, corners=refined[(NATIVE_ORDER - rotation) % 4] + .5,
                    source="decoded", hamming=read[1], correlation=read[2])

    def detect(self, gray, shapes):
        found = []
        for quad in border_quads(gray):
            read = self.read_cells(gray, quad, 5)
            if read is not None:
                hit = self.verify(gray, quad, read[0])
                if hit is not None:
                    found.append(hit)
        decoded = {hit["tag_id"] for hit in found}
        small = cv2.resize(gray, None, fx=.625, fy=.625, interpolation=cv2.INTER_AREA)
        for tag_id, outlines in shapes.items():
            if tag_id in decoded:
                continue
            label = self.labels.index((tag_id, 0))
            for quad in projected_quads(small, outlines, self.templates[label]):
                hit = self.verify(gray, quad, label)
                if hit is not None:
                    found.append(hit)
                    break
        # Several intensity thresholds can find the same border. Preserve
        # spatially distinct copies for the rigid solver to disambiguate.
        unique = []
        for hit in sorted(found, key=lambda h: h["correlation"] - .035 * h["hamming"], reverse=True):
            if not any(h["tag_id"] == hit["tag_id"] and
                       np.linalg.norm(h["corners"].mean(0) - hit["corners"].mean(0)) < 6 for h in unique):
                unique.append(hit)
        return unique


def border_quads(gray):
    masks = [cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)[1] for t in range(35, 201, 15)]
    masks.extend(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, size, 5)
                 for size in (21, 41, 71))
    visited = set()
    for mask in masks:
        for contour in cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]:
            area = cv2.contourArea(contour)
            if not 125 <= area <= 18000:
                continue
            hull = cv2.convexHull(contour)
            quad = cv2.approxPolyDP(hull, .043 * cv2.arcLength(hull, True), True).reshape(-1, 2)
            if len(quad) != 4 or area < .75 * cv2.contourArea(hull):
                continue
            if cv2.contourArea(quad, oriented=True) < 0:
                quad = quad[::-1]
            quad = np.roll(quad, -quad.sum(axis=1).argmin(), axis=0).astype(np.float32)
            key = tuple(np.rint(quad / 1.5).astype(int).ravel())
            if key not in visited:
                visited.add(key)
                yield quad


def projected_quads(small, outlines, template):
    """Search for a moving, blurred border using the projected shape of this face."""
    peaks = []
    for outline in outlines:
        if not np.isfinite(outline).all() or cv2.contourArea(outline, oriented=True) < 175:
            continue
        relative = outline - outline.mean(axis=0)
        for degrees in (-20, 0, 20):
            rotation = cv2.getRotationMatrix2D((0, 0), degrees, 1)[:, :2].astype(np.float32)
            for scale in (.88, 1.10):
                quad = relative @ rotation.T * (scale * .625)
                low = np.floor((quad * 1.14).min(0)).astype(int)
                width, height = np.ceil((quad * 1.14).max(0)).astype(int) - low + 1
                if not (8 <= width <= min(125, small.shape[1]) and 8 <= height <= min(125, small.shape[0])):
                    continue
                quad -= low
                warp = cv2.getPerspectiveTransform(TEMPLATE_CORNERS, quad)
                tile = cv2.warpPerspective(template, warp, (width, height), borderValue=255)
                match = cv2.matchTemplate(small, cv2.GaussianBlur(tile, (0, 0), 1.3), cv2.TM_CCOEFF_NORMED)
                _, score, _, peak = cv2.minMaxLoc(match)
                if score > .43:
                    peaks.append((score, ((quad + peak + .5) / .625 - .5).astype(np.float32)))
    visited = []
    for _, quad in sorted(peaks, key=lambda p: p[0], reverse=True):
        if all(np.sqrt(np.mean((quad - previous) ** 2)) >= 2 for previous in visited):
            visited.append(quad)
            yield quad


def pose_guides(rows):
    anchors = np.array([i for i, r in enumerate(rows) if r["status"] in ("measured", "flow_only")])
    guides = {}
    for i, row in enumerate(rows):
        if row["status"] != "lost" or not len(anchors):
            continue
        j = anchors.searchsorted(i)
        nearby = anchors[max(0, j - 1):j + 1]
        if min(abs(rows[a]["time_s"] - row["time_s"]) for a in nearby) > .51:
            continue
        poses = [matrix(rows[a]["R"], rows[a]["t"]) for a in nearby]
        prior = poses[0]
        if len(poses) == 2:
            times = [rows[a]["time_s"] for a in nearby]
            if times[1] <= times[0]:
                continue
            fraction = (row["time_s"] - times[0]) / (times[1] - times[0])
            rotation = Slerp(times, Rotation.from_matrix([p[:3, :3] for p in poses]))(row["time_s"])
            prior = matrix(rotation.as_matrix(), (1 - fraction) * poses[0][:3, 3] + fraction * poses[1][:3, 3])
        guides[i] = [prior, *poses]
    return guides


def collect_patterns(capture, camera, times, rows, guides, decoder, points, max_sync_us, hits):
    if not len(times):
        return
    center_3d = np.concatenate(list(points.values())).mean(axis=0, keepdims=True)
    last = None
    for frame, poses in guides.items():
        timestamp = rows[frame]["timestamp_us"]
        j = int(np.searchsorted(times, timestamp))
        idx = min(range(max(0, j - 1), min(j + 1, len(times))), key=lambda i: abs(int(times[i]) - timestamp))
        if abs(int(times[idx]) - timestamp) > max_sync_us:
            continue
        if last != idx:
            if last is None or idx < last or idx - last > 20:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, idx):
                    raise ValueError(f"Cannot seek to {camera.id} frame {idx} for pattern recovery")
            else:
                for k in range(last + 1, idx):
                    if not capture.grab():
                        raise ValueError(f"Cannot read {camera.id} frame {k} for pattern recovery")
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot read {camera.id} frame {idx} for pattern recovery")
            last = idx
        prior = poses[0]
        center = project_pose(camera, prior[:3, :3], prior[:3, 3], center_3d)[0]
        if not np.isfinite(center).all():
            continue
        dimensions = np.array(image.shape[:2][::-1])
        side = np.minimum(320, dimensions)
        offset = np.clip(np.rint(center).astype(int) - side // 2, 0, dimensions - side)
        x, y = offset
        gray = cv2.cvtColor(image[y:y + side[1], x:x + side[0]], cv2.COLOR_BGR2GRAY)
        shapes = {tag: [(project_pose(camera, p[:3, :3], p[:3, 3], xyz)[NATIVE_ORDER] - .5 - offset)
                        .astype(np.float32) for p in poses] for tag, xyz in points.items()}
        for hit in decoder.detect(gray, shapes):
            hits[frame].append(dict(hit, camera=camera.id, corners=hit["corners"] + offset))


def recover_pattern_gaps(rows, recording, metadata, cameras, timestamps, points, size, alignment,
                         decoder, max_error, max_sync_us):
    if not np.array_equal([r["timestamp_us"] for r in rows], timestamps["cam_0"][:len(rows)]):
        raise ValueError("Pattern recovery requires matching reference camera and pose timestamps")
    guides = pose_guides(rows)
    if not guides:
        return []
    hits = {i: [] for i in guides}
    for camera_id, stream in STREAMS.items():
        capture = cv2.VideoCapture(str(recording / metadata["files"][f"video_{stream}"]))
        try:
            collect_patterns(capture, cameras[camera_id], timestamps[camera_id], rows, guides, decoder,
                             points, max_sync_us, hits)
        finally:
            capture.release()
    report = []
    for frame, detections in hits.items():
        if len({d["camera"] for d in detections}) < 2:
            continue
        fit = fit_rigid_detections(detections, cameras, points, size, alignment, guides[frame][0], max_error)
        if fit is None or fit["n_views"] < 2:
            continue
        observations = fit.pop("observations")
        rows[frame].update(fit, pose_source="pattern_redetected", pattern_observations=observations)
        report.append(dict(frame=frame, rms_px=fit["source_rms"], observations=[
            dict(camera=d["camera"], tag_id=d["tag_id"], corners=d["corners"].tolist(),
                 hamming=d["hamming"], correlation=d["correlation"]) for d in observations]))
    return report
