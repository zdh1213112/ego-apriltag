#!/usr/bin/env python3
"""Audit saved inter-tag transforms without claiming ground-truth accuracy."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def audit_alignment(results):
    results = Path(results)
    summary = json.loads((results / "summary.json").read_text())
    tracking = Path(summary.get("original_results", str(results)))
    tag_summary = json.loads((tracking / "summary.json").read_text())
    alignment = json.loads((results / "tag_alignment.json").read_text())
    reference = alignment["reference_tag_id"]
    tracks, image_quality = {}, {}
    for tag_id in tag_summary["tag_ids"]:
        directory = tracking / f"tag_{tag_id}"
        with (directory / "poses.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        with np.load(directory / "poses.npz") as arrays:
            tracks[tag_id] = {key: arrays[key].copy() for key in ("T_cam0_tag_raw", "status", "timestamp_us")}
        tracks[tag_id]["views"] = np.array([int(row["n_views"]) for row in rows])
        quality = json.loads((directory / "summary.json").read_text())
        image_quality[str(tag_id)] = {key: quality[key] for key in (
            "valid_frames", "reprojection_rms_px_median", "reprojection_rms_px_p95", "view_count_histogram")}
    origin = tracks[reference]
    count = len(origin["status"])
    pairs = []
    for tag_id, track in tracks.items():
        if tag_id == reference or str(tag_id) not in alignment["T_reference_tag"]:
            continue
        if not np.array_equal(origin["timestamp_us"], track["timestamp_us"]):
            raise ValueError("Per-tag timestamps must match before checking alignment")
        valid = ((origin["status"] == "measured") & (track["status"] == "measured")
                 & (origin["views"] >= 2) & (track["views"] >= 2))
        relative = np.linalg.inv(origin["T_cam0_tag_raw"][valid]) @ track["T_cam0_tag_raw"][valid]
        target = np.array(alignment["T_reference_tag"][str(tag_id)])
        pair = {"reference_tag_id": reference, "tag_id": tag_id,
                "simultaneous_multiview_measured_frames": int(valid.sum())}
        if not len(relative):
            pair["available"] = False
            pairs.append(pair)
            continue
        distance = np.linalg.norm(relative[:, :3, 3] - target[:3, 3], axis=1) * 1000
        angle = np.rad2deg((Rotation.from_matrix(target[:3, :3]).inv()
                           * Rotation.from_matrix(relative[:, :3, :3])).magnitude())
        pair.update(available=True, translation_median_mm=float(np.median(distance)),
                    translation_p95_mm=float(np.percentile(distance, 95)),
                    rotation_median_deg=float(np.median(angle)), rotation_p95_deg=float(np.percentile(angle, 95)))
        first = np.flatnonzero(valid) < count // 2
        pair["temporal_half_frames"] = [int(first.sum()), int((~first).sum())]
        if min(pair["temporal_half_frames"]) >= 3:
            centers = [(np.median(relative[part, :3, 3], axis=0),
                        Rotation.from_matrix(relative[part, :3, :3]).mean()) for part in (first, ~first)]
            pair["temporal_half_difference_mm"] = float(np.linalg.norm(centers[0][0] - centers[1][0]) * 1000)
            pair["temporal_half_difference_deg"] = float(np.rad2deg((centers[0][1].inv() * centers[1][1]).magnitude()))
        pairs.append(pair)
    return {"reference_tag_id": reference, "frames": count, "tracking_results": str(tracking.resolve()),
            "interpretation": "Internal consistency and temporal stability; no ground-truth accuracy validation.",
            "limitation": "The same layout selected these observations; these are in-sample diagnostics, not independent holdout errors.",
            "sample_rule": "Both tags measured with at least two cameras; flow-only and interpolated poses excluded.",
            "alignment_paths": alignment["paths"], "unaligned_tag_ids": alignment.get("unaligned_tag_ids", []),
            "per_tag_image_quality": image_quality, "reference_pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Tracking or recovered multi-tag results")
    args = parser.parse_args()
    report = audit_alignment(args.results)
    path = args.results / "alignment_quality.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for pair in report["reference_pairs"]:
        if pair.get("available"):
            print(f"ID {pair['tag_id']} -> ID {pair['reference_tag_id']}: "
                  f"{pair['simultaneous_multiview_measured_frames']} frames; "
                  f"position P95 {pair['translation_p95_mm']:.2f} mm; rotation P95 {pair['rotation_p95_deg']:.2f} deg")
    print("These are internal consistency errors, not absolute accuracy.")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
