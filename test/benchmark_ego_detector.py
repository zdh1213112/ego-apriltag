#!/usr/bin/env python3
"""Compare CPU/CUDA full-image detection on evenly spaced recording frames."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ego_detector import create_detector
from track_ego_apriltag import load_recording, STREAMS


def compare(cpu, gpu):
    errors = []
    for tag_id in {row["id"] for row in cpu} & {row["id"] for row in gpu}:
        a = [row["lb-rb-rt-lt"] for row in cpu if row["id"] == tag_id]
        b = [row["lb-rb-rt-lt"] for row in gpu if row["id"] == tag_id]
        distances = np.linalg.norm(np.mean(a, axis=1)[:, None] - np.mean(b, axis=1)[None], axis=-1)
        rows, cols = linear_sum_assignment(distances)
        for i, j in zip(rows, cols):
            if distances[i, j] < 5:
                errors.append(float(np.sqrt(np.mean(np.sum((a[i] - b[j]) ** 2, axis=1)))))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--samples", type=int, default=12, help="Samples per camera")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    cpu, cuda = create_detector("tag36h11", "cpu"), create_detector("tag36h11", "cuda")
    report = {"gpu": cuda.gpu_name, "cameras": {}}
    metadata, _, timestamps = load_recording(args.recording)
    for camera, stream in STREAMS.items():
        capture = cv2.VideoCapture(str(args.recording / metadata["files"][f"video_{stream}"]))
        times = {"cpu": [], "cuda": []}
        counts = {"cpu": 0, "cuda": 0}
        errors = []
        for index in np.unique(np.linspace(0, len(timestamps[camera]) - 1, args.samples, dtype=int)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read {camera}/{index}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Warm both detectors after buffer/shape initialization, before timing.
            if not times["cpu"]:
                cpu.detect(gray)
                cuda.detect(gray)
            detections = {}
            for backend, detector in (("cpu", cpu), ("cuda", cuda)):
                start = time.perf_counter()
                raw = detector.detect(gray)
                times[backend].append(time.perf_counter() - start)
                detections[backend] = [row for row in raw if row["margin"] >= 25 and row["id"] in (0, 1, 2, 3)]
                counts[backend] += len(detections[backend])
            errors.extend(compare(detections["cpu"], detections["cuda"]))
        capture.release()
        row = {"frames": len(times["cpu"]), "detections": counts,
               "median_ms": {key: float(np.median(values) * 1000) for key, values in times.items()},
               "matched_instances": len(errors),
               "corner_difference_median_px": float(np.median(errors)) if errors else None,
               "corner_difference_p95_px": float(np.percentile(errors, 95)) if errors else None}
        report["cameras"][camera] = row
        print(camera, json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
