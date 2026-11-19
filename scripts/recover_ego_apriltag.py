#!/usr/bin/env python3
"""Offline recovery of short EGO tracking gaps using the calibrated tag layout.

Re-detection is distinguished from interpolation. Original results are preserved.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from track_ego_apriltag import (
    AdaptiveFilter, STREAMS, export_global_results, load_recording, local_detection,
    fit_rigid_detections, flow_corners, matrix, object_points, plot_tag_results, project_pose,
)


def rigid_points(alignment, size):
    points = object_points(size)
    return {int(key): points @ np.array(value)[:3, :3].T + np.array(value)[:3, 3]
            for key, value in alignment["T_reference_tag"].items()}


def fill_short_pose_gaps(results, max_frames=3, max_span_s=.15):
    """Bounded two-sided interpolation; never extrapolate or fill long gaps."""
    completed = [dict(row) for row in results]
    indices = np.flatnonzero([row["status"] == "lost" for row in results])
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    for run in runs:
        if not len(run) or len(run) > max_frames:
            continue
        left, right = int(run[0]) - 1, int(run[-1]) + 1
        if left < 0 or right >= len(results):
            continue
        before, after = results[left], results[right]
        span = after["time_s"] - before["time_s"]
        if not 0 < span <= max_span_s:
            continue
        if np.linalg.norm(after["t"] - before["t"]) / span > 5:
            continue
        if Rotation.from_matrix(before["R"].T @ after["R"]).magnitude() / span > 15:
            continue
        slerp = Slerp([before["time_s"], after["time_s"]], Rotation.from_matrix([before["R"], after["R"]]))
        for index in run:
            row = completed[index]
            fraction = (row["time_s"] - before["time_s"]) / span
            row.update(status="interpolated", pose_source="interpolated",
                       R=slerp(row["time_s"]).as_matrix(), t=before["t"] * (1 - fraction) + after["t"] * fraction,
                       source_tag_ids=[], n_valid_tags=0, n_views=0, source_rms=np.nan)
    return completed


def reject_isolated_excursions(results, max_frames=3):
    """Flag brief, unverified jumps between two independently image-recovered poses."""
    output = [dict(row) for row in results]
    indices = np.flatnonzero([row.get("pose_source") == "original_unverified" for row in results])
    verified = {"redetected", "flow_recovered"}
    for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        if not len(run) or len(run) > max_frames or run[0] == 0 or run[-1] == len(results) - 1:
            continue
        left, right = results[run[0] - 1], results[run[-1] + 1]
        if left.get("pose_source") not in verified or right.get("pose_source") not in verified:
            continue
        span = right["time_s"] - left["time_s"]
        if not 0 < span <= .15 or np.linalg.norm(left["t"] - right["t"]) > .15:
            continue
        if Rotation.from_matrix(left["R"].T @ right["R"]).magnitude() > np.deg2rad(40):
            continue
        errors = []
        for index in run:
            fraction = (results[index]["time_s"] - left["time_s"]) / span
            predicted = left["t"] * (1 - fraction) + right["t"] * fraction
            errors.append(np.linalg.norm(results[index]["t"] - predicted))
        # Preserve plausible rows next to an isolated jump in the same run.
        # Both enclosing anchors still have independent image support.
        for index, error in zip(run, errors):
            if error <= .25:
                continue
            output[index].update(status="lost", pose_source="rejected_isolated_excursion",
                                 source_tag_ids=[], n_valid_tags=0, n_views=0, source_rms=np.nan)
            output[index].pop("R", None)
            output[index].pop("t", None)
    return output


def load_results(directory):
    with (directory / "global_poses.csv").open() as stream:
        table = list(csv.DictReader(stream))
    with np.load(directory / "global_poses.npz") as arrays:
        results = []
        for index, row in enumerate(table):
            result = {"frame": index, "timestamp_us": int(row["timestamp_us"]), "time_s": float(row["time_s"]),
                      "status": row["status"], "original_status": row["status"],
                      "n_valid_tags": int(row["n_valid_tags"]), "n_views": int(row["n_views"]),
                      "source_tag_ids": [int(value) for value in row["source_tag_ids"].split(";") if value],
                      "source_rms": float(row["source_rms_px"]),
                      "pose_source": "lost" if row["status"] == "lost" else "original"}
            if row["status"] != "lost":
                T = arrays["T_cam0_global_raw"][index]
                result.update(R=T[:3, :3].copy(), t=T[:3, 3].copy())
            results.append(result)
    tag_results = {}
    for tag_id in json.loads((directory / "summary.json").read_text())["tag_ids"]:
        with np.load(directory / f"tag_{tag_id}/poses.npz") as arrays:
            tag_results[tag_id] = [{"status": str(value)} for value in arrays["status"]]
    return results, tag_results


def recovery_windows(results, max_anchor_frames=45):
    focus = {index for index, row in enumerate(results) if row["status"] == "lost"}
    # Inspect isolated single-ID position spikes as well as explicit dropouts.
    # They only select re-detection windows; this does not reject fast motion.
    for index in range(1, len(results) - 1):
        before, row, after = results[index - 1:index + 2]
        if any(value["status"] == "lost" for value in (before, row, after)):
            continue
        if len(row["source_tag_ids"]) < 2 and np.linalg.norm(row["t"] - (before["t"] + after["t"]) / 2) > .20:
            focus.add(index)
    lost = np.array(sorted(focus), dtype=int)
    anchors = np.flatnonzero([row["status"] == "measured" and len(row["source_tag_ids"]) >= 2 for row in results])
    windows = []
    for run in np.split(lost, np.flatnonzero(np.diff(lost) != 1) + 1):
        if not len(run):
            continue
        before, after = anchors[anchors < run[0]], anchors[anchors > run[-1]]
        if not len(before) or not len(after):
            continue
        left, right = int(before[-1]), int(after[0])
        if run[0] - left <= max_anchor_frames and right - run[-1] <= max_anchor_frames:
            if windows and left <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(right, windows[-1][1]))
            else:
                windows.append((left, right))
    return windows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Existing multi-tag output directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-gap-frames", type=int, default=3, help="Maximum interpolation gap; 0 disables interpolation")
    parser.add_argument("--cutoff", type=float, default=4., help="Minimum adaptive filter cutoff, Hz")
    parser.add_argument("--no-filter", action="store_true", help="Skip pose smoothing")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        help="Detector backend; follows the original tracking results when omitted")
    parser.add_argument("--max-reprojection", type=float, default=3., help="Maximum reprojection error, pixels")
    args = parser.parse_args(argv)
    if args.max_gap_frames < 0:
        parser.error("max-gap-frames must be nonnegative")
    if any(not np.isfinite(value) or value <= 0 for value in (args.cutoff, args.max_reprojection)):
        parser.error("cutoff and max-reprojection must be positive finite values")
    output = args.output or args.results.with_name(args.results.name + "_recovered")
    if output.resolve() == args.results.resolve():
        parser.error("output must differ from the original results directory")
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.results / "summary.json").read_text())
    alignment = json.loads((args.results / "tag_alignment.json").read_text())
    recording = Path(summary["recording"])
    metadata, cameras, _ = load_recording(recording)
    results, tag_results = load_results(args.results)
    windows = recovery_windows(results)
    points = rigid_points(alignment, summary["tagsize_m"])
    priors = {}
    for left, right in windows:
        a, b = results[left], results[right]
        slerp = Slerp([a["time_s"], b["time_s"]], Rotation.from_matrix([a["R"], b["R"]]))
        for index in range(left, right + 1):
            fraction = (results[index]["time_s"] - a["time_s"]) / (b["time_s"] - a["time_s"])
            priors[index] = matrix(slerp(results[index]["time_s"]).as_matrix(), a["t"] * (1 - fraction) + b["t"] * fraction)
    from ego_detector import create_detector
    args.device = args.device or summary.get("detector_device", "cpu")
    detector = create_detector(summary["family"], args.device)
    detections = {index: [] for index in priors}
    for camera_id, stream in STREAMS.items():
        capture = cv2.VideoCapture(str(recording / metadata["files"][f"video_{stream}"]))
        try:
            for left, right in windows:
                capture.set(cv2.CAP_PROP_POS_FRAMES, left)
                gray_frames = {}
                for index in range(left, right + 1):
                    ok, image = capture.read()
                    if not ok:
                        raise RuntimeError(f"Cannot decode {camera_id}/{index}")
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray_frames[index] = gray
                    raw = [row for row in detector.detect(gray) if row["id"] in points and row["margin"] >= 25]
                    for row in raw:
                        detections[index].append({"camera": camera_id, "tag_id": row["id"],
                                                  "corners": row["lb-rb-rt-lt"], "source": "decoded"})
                    prior = priors[index]
                    for tag_id, xyz in points.items():
                        projected = project_pose(cameras[camera_id], prior[:3, :3], prior[:3, 3], xyz)
                        if np.any(projected < 4) or np.any(projected[:, 0] >= cameras[camera_id].width - 4) or np.any(projected[:, 1] >= cameras[camera_id].height - 4):
                            continue
                        found = local_detection(gray, cameras[camera_id], projected, detector, tag_id)
                        if found is not None:
                            detections[index].append({**found, "camera": camera_id, "tag_id": tag_id,
                                                      "corners": np.array(found["corners"])})
                # Track from both clear endpoints; every image pair still passes
                # forward/backward KLT checks, independently of the 3D prediction.
                for anchor, direction, source in ((right, -1, "backward_flow"), (left, 1, "forward_flow")):
                    for tag_id, xyz in points.items():
                        projected = project_pose(cameras[camera_id], priors[anchor][:3, :3], priors[anchor][:3, 3], xyz)
                        choices = [row for row in detections[anchor]
                                   if row["camera"] == camera_id and row["tag_id"] == tag_id]
                        if not choices:
                            continue
                        seed = min(choices, key=lambda row: np.mean((row["corners"] - projected) ** 2))
                        if np.sqrt(np.mean((seed["corners"] - projected) ** 2)) > 5:
                            continue
                        corners = np.array(seed["corners"])
                        stop = max(left - 1, anchor - 31) if direction < 0 else min(right + 1, anchor + 31)
                        for index in range(anchor + direction, stop, direction):
                            tracked = flow_corners(gray_frames[index - direction], gray_frames[index], corners)
                            if tracked is None:
                                break
                            corners = tracked
                            detections[index].append({"camera": camera_id, "tag_id": tag_id,
                                                      "corners": tracked, "source": source})
        finally:
            capture.release()
        print(f"Re-detected {camera_id}: {len(priors)} frames", flush=True)
    diagnostics = []
    recovered = [dict(row) for row in results]
    for index in sorted(priors):
        fit = fit_rigid_detections(detections[index], cameras, points, summary["tagsize_m"], alignment,
                                   priors[index], max_error=args.max_reprojection)
        row = recovered[index]
        diagnostics.append({"frame": index, "original_status": row["status"], "detections": len(detections[index]),
                            "recovered_ids": [] if fit is None else fit["source_tag_ids"]})
        if fit is not None:
            row.update({key: value for key, value in fit.items() if key != "observations"})
            row.update(pose_source="redetected" if row["status"] == "measured" else "flow_recovered")
        elif row["status"] != "lost":
            # A linear anchor prediction cannot invalidate measurements during
            # fast head motion. Preserve them when image recovery is inconclusive.
            row["pose_source"] = "original_unverified"
        print(f'frame {index}: {results[index]["status"]} -> {row["status"]}, IDs {row["source_tag_ids"]}', flush=True)
    recovered = reject_isolated_excursions(recovered)
    completed = fill_short_pose_gaps(recovered, args.max_gap_frames)
    state = AdaptiveFilter(None if args.no_filter else args.cutoff)
    for row in completed:
        if row["status"] != "lost":
            row["filtered_R"], row["filtered_t"] = state.update(row["R"], row["t"], row["time_s"])
    export_global_results(output, completed, alignment, list(tag_results))
    plot_tag_results(output, tag_results, completed, alignment["reference_tag_id"])
    with (output / "global_poses.csv").open() as stream:
        reader = csv.DictReader(stream)
        fields, table = reader.fieldnames, list(reader)
    with (output / "global_poses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ["pose_source", "original_status"])
        writer.writeheader()
        for row, result in zip(table, completed):
            writer.writerow({**row, "pose_source": result["pose_source"], "original_status": result["original_status"]})
    with np.load(output / "global_poses.npz") as arrays:
        payload = {key: arrays[key].copy() for key in arrays.files}
    payload.update(pose_source=np.array([row["pose_source"] for row in completed]),
                   original_status=np.array([row["original_status"] for row in completed]))
    np.savez_compressed(output / "global_poses.npz", **payload)
    original_lost = np.flatnonzero([row["status"] == "lost" for row in results])
    original_gaps = [run for run in np.split(original_lost, np.flatnonzero(np.diff(original_lost) != 1) + 1) if len(run)]
    for item in diagnostics:
        row = completed[item["frame"]]
        item.update(final_status=row["status"], pose_source=row["pose_source"])
    report = {"original_results": str(args.results.resolve()), "frames": len(results), "windows": windows,
              "detector_device": args.device,
              "settings": {"max_gap_frames": args.max_gap_frames, "cutoff": args.cutoff,
                           "no_filter": args.no_filter, "max_reprojection": args.max_reprojection},
              "original_gaps": [{"first_frame": int(run[0]), "last_frame": int(run[-1]), "frames": len(run),
                                  "start_time_s": results[run[0]]["time_s"], "end_time_s": results[run[-1]]["time_s"]}
                                 for run in original_gaps],
              "original_lost_frames": sum(row["status"] == "lost" for row in results),
              "remaining_lost_frames": sum(row["status"] == "lost" for row in completed),
              "redetected_original_lost_frames": sum(a["status"] == "lost" and b["pose_source"] == "redetected" for a, b in zip(results, completed)),
              "flow_recovered_original_lost_frames": sum(a["status"] == "lost" and b["pose_source"] == "flow_recovered" for a, b in zip(results, completed)),
              "interpolated_frames": sum(row["status"] == "interpolated" for row in completed),
              "rejected_excursion_frames": [index for index, row in enumerate(recovered) if row["pose_source"] == "rejected_isolated_excursion"],
              "image_supported_frames": sum(row["status"] in ("measured", "flow_only") for row in completed),
              "completed_frames": sum(row["status"] != "lost" for row in completed),
              "diagnostics": diagnostics,
              "limitations": ["Uses fixed inter-tag transforms estimated from the recording.",
                              "Interpolation is an estimate, not a visual measurement; long and unbounded gaps remain lost.",
                              "Only windows near original tracking gaps and isolated single-ID spikes are re-detected."]}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "diagnostics"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
