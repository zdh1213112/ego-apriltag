#!/usr/bin/env python3
"""Render saved recovered EGO poses, without running detection again."""

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from track_ego_apriltag import STREAMS, draw_tag_overlay, load_recording, object_points, project_pose


@contextmanager
def h264_writer(path, width, height, fps):
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.8f}", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-threads", "2",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        yield process.stdin
    except BaseException:
        process.terminate()
        process.stdin.close()
        process.stdin = None
        process.communicate()
        raise
    else:
        process.stdin.close()
        process.stdin = None
        _, error = process.communicate()
        if process.returncode:
            raise RuntimeError(f"Video encoding failed: {error.decode(errors='replace')}")


def draw_recovered_frame(image, camera, row, pose, transforms, size):
    status, source = row["status"], row["pose_source"]
    estimated = status == "interpolated"
    color = (0, 190, 255) if estimated else ((60, 80, 255) if status == "lost" else (80, 240, 120))
    axes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]]) * size * .6
    scale = np.array([image.shape[1] / camera.width, image.shape[0] / camera.height])
    if status != "lost":
        for tag_id, offset in transforms.items():
            tag_pose = pose @ offset
            center = camera.R @ tag_pose[:3, 3] + camera.t
            normal = camera.R @ tag_pose[:3, 2]
            # +Z points into the printed plane; omit its back-facing side.
            if np.dot(normal, center) <= 0:
                continue
            corners = project_pose(camera, tag_pose[:3, :3], tag_pose[:3, 3], object_points(size)) * scale
            if not np.isfinite(corners).all() or np.max(np.abs(corners)) >= 5000:
                continue
            if (np.any(corners.max(axis=0) < 0)
                    or np.any(corners.min(axis=0) >= image.shape[1::-1])):
                continue
            cv2.polylines(image, [np.rint(corners).astype(np.int32)], True,
                          (0, 190, 255) if estimated else (180, 220, 180), 1, cv2.LINE_AA)
            draw_tag_overlay(image, camera, None, {"status": status, "R": tag_pose[:3, :3],
                                                  "t": tag_pose[:3, 3]}, tag_id, axes)
    source_names = {"original": "original", "original_unverified": "original / unverified",
                    "redetected": "re-detected", "flow_recovered": "flow recovered",
                    "interpolated": "interpolated / no image measurement"}
    state = "ESTIMATED" if estimated else ("LOST" if status == "lost" else "TRACKING")
    source_text = source_names.get(source, source)
    ids = row["source_tag_ids"].replace(";", ",") or "none"
    cv2.rectangle(image, (0, 0), (image.shape[1], 88), (12, 12, 12), -1)
    lines = [f'{camera.id} / {STREAMS[camera.id]} | frame {row["frame"]} | {float(row["time_s"]):.3f}s',
             f"GLOBAL: {state} | {source_text}",
             f'image IDs: {ids} | views: {row["n_views"]} | axes from rigid tag layout']
    for index, text in enumerate(lines):
        cv2.putText(image, text, (10, 23 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    color if index == 1 else (240, 240, 240), 1, cv2.LINE_AA)
    return image


def render(results, width=960):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg with libx264 is required to export H.264 previews")
    report = json.loads((results / "summary.json").read_text())
    original = Path(report["original_results"])
    summary = json.loads((original / "summary.json").read_text())
    recording = Path(summary["recording"])
    metadata, cameras, timestamps = load_recording(recording)
    alignment = json.loads((results / "tag_alignment.json").read_text())
    transforms = {int(key): np.array(value) for key, value in alignment["T_reference_tag"].items()}
    with (results / "global_poses.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with np.load(results / "global_poses.npz") as payload:
        poses = payload["T_cam0_global_raw"].copy()
        if len(rows) != len(poses):
            raise ValueError("Recovered CSV and NPZ frame counts differ")
        if not np.array_equal(payload["timestamp_us"], timestamps["cam_0"][:len(rows)]):
            raise ValueError("Recovered poses and recording timestamps differ")
    fps = 1e6 / float(np.median(np.diff(timestamps["cam_0"])))
    height = round(width * cameras["cam_0"].height / cameras["cam_0"].width / 2) * 2
    previews = results / "camera_previews"
    previews.mkdir(exist_ok=True)
    samples = {0, len(rows) - 1} | {index for index, row in enumerate(rows) if row["status"] == "interpolated"}
    for camera_id, stream in STREAMS.items():
        capture = cv2.VideoCapture(str(recording / metadata["files"][f"video_{stream}"]))
        try:
            with h264_writer(previews / f"{camera_id}.mp4", width, height, fps) as sink:
                for index, row in enumerate(rows):
                    ok, image = capture.read()
                    if not ok:
                        raise RuntimeError(f"Recording ended early: {camera_id}/{index}")
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    image = draw_recovered_frame(image, cameras[camera_id], row, poses[index],
                                                  transforms, summary["tagsize_m"])
                    sink.write(image.tobytes())
        finally:
            capture.release()
        print(f"Rendered {camera_id}: {len(rows)} frames", flush=True)
    captures = {key: cv2.VideoCapture(str(previews / f"{key}.mp4")) for key in STREAMS}
    try:
        with h264_writer(results / "tracking_preview.mp4", width * 2, height * 2, fps) as sink:
            for index in range(len(rows)):
                images = {}
                for key, capture in captures.items():
                    ok, image = capture.read()
                    if not ok:
                        raise RuntimeError(f"Preview ended early: {key}/{index}")
                    images[key] = image
                mosaic = np.vstack((np.hstack((images["cam_0"], images["cam_1"])),
                                    np.hstack((images["cam_3"], images["cam_2"]))))
                sink.write(mosaic.tobytes())
                if index in samples:
                    cv2.imwrite(str(results / f"recovered_frame_{index:04d}.jpg"), mosaic)
    finally:
        for capture in captures.values():
            capture.release()
    print(f"Saved {results / 'tracking_preview.mp4'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Recovered results directory")
    args = parser.parse_args()
    render(args.results)


if __name__ == "__main__":
    main()
