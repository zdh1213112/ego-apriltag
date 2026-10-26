#!/usr/bin/env python3
"""Detect AprilTags and optionally estimate their pose in the camera frame."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# Use the module built by CMake without requiring a system installation.
sys.path.insert(0, str(ROOT / "build"))

try:
    from apriltag import apriltag
except ImportError as exc:
    raise SystemExit(
        "Cannot import the AprilTag binding. Build it in the repository's "
        "build directory using the same Python interpreter, or set PYTHONPATH "
        f"to your build directory.\n{exc}"
    ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", type=Path,
        default=ROOT / "test/data/33369213973_9d9bb4cc96_c.jpg",
        help="Input image (defaults to the bundled tag36h11 sample)",
    )
    parser.add_argument("--family", default="tag36h11", help="Tag family")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--decimate", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="Save the annotated image")
    parser.add_argument(
        "--tagsize", type=float,
        help="Physical side length between detection corners, in meters",
    )
    parser.add_argument(
        "--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"),
        help="Calibrated camera intrinsics in pixels, at the input image resolution",
    )
    parser.add_argument(
        "--distortion", nargs="+", type=float, metavar="D",
        help="OpenCV pinhole distortion coefficients; undistort before detection "
             "(4, 5, 8, 12 or 14 values; starts with k1 k2 p1 p2)",
    )
    args = parser.parse_args()
    if args.threads < 1 or not np.isfinite(args.decimate) or args.decimate < 1:
        parser.error("threads and decimate must be at least 1")
    pose_requested = any(value is not None for value in (
        args.tagsize, args.intrinsics, args.distortion,
    ))
    camera_matrix = None
    if pose_requested:
        if args.tagsize is None or args.intrinsics is None:
            parser.error("Pose estimation requires both --tagsize and --intrinsics")
        if not np.isfinite(args.tagsize) or args.tagsize <= 0:
            parser.error("tagsize must be a positive finite value in meters")
        fx, fy, cx, cy = args.intrinsics
        if not np.isfinite(args.intrinsics).all() or fx <= 0 or fy <= 0:
            parser.error("Intrinsics must be finite, with positive fx and fy")
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    if args.distortion is not None:
        if len(args.distortion) not in (4, 5, 8, 12, 14):
            parser.error("distortion requires 4, 5, 8, 12 or 14 coefficients")
        if not np.isfinite(args.distortion).all():
            parser.error("Distortion coefficients must be finite")

    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    if args.distortion is not None:
        # Preserve K so the pose estimator and axis drawing use the same intrinsics.
        image = cv2.undistort(
            image, camera_matrix, np.array(args.distortion),
            newCameraMatrix=camera_matrix,
        )
        print("Detection coordinates and output image use the undistorted frame")

    detector = apriltag(
        args.family, threads=args.threads, decimate=args.decimate,
    )
    detections = detector.detect(image)
    print(f"Detected {len(detections)} tags ({args.family})")
    poses = []
    for det in detections:
        print(f"ID={det['id']} hamming={det['hamming']} margin={det['margin']:.2f}")
        print("  center:", det["center"])
        print("  corners (lb, rb, rt, lt):", det["lb-rb-rt-lt"])
        if pose_requested:
            # This binding accepts positional arguments for estimate_tag_pose.
            pose = detector.estimate_tag_pose(det, args.tagsize, *args.intrinsics)
            poses.append(pose)
            transform = np.eye(4)
            transform[:3, :3] = pose["R"]
            transform[:3, 3] = pose["t"].ravel()
            print("  t_camera_tag (meters):", pose["t"].ravel())
            print("  R_camera_tag:\n", pose["R"])
            print("  T_camera_tag (p_camera = R @ p_tag + t):\n", transform)
            print("  object-space error:", pose["error"])

    if args.output:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for index, det in enumerate(detections):
            corners = np.rint(det["lb-rb-rt-lt"]).astype(np.int32)
            center = tuple(np.rint(det["center"]).astype(int))
            cv2.polylines(annotated, [corners], True, (0, 255, 0), 2)
            cv2.putText(annotated, str(det["id"]), center,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if pose_requested:
                pose = poses[index]
                rvec, _ = cv2.Rodrigues(pose["R"])
                # The image is undistorted (or assumed so); no distortion here.
                cv2.drawFrameAxes(
                    annotated, camera_matrix, None, rvec, pose["t"],
                    args.tagsize / 2,
                )
        if not cv2.imwrite(str(args.output), annotated):
            parser.error(f"Cannot save image: {args.output}")
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
