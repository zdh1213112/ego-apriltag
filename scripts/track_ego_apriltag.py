#!/usr/bin/env python3
"""Offline, calibrated four-camera AprilTag tracking for EGOViewer recordings.

Output transform convention: p_cam0 = R_cam0_tag @ p_tag + t_cam0_tag.
The KB model is inverted without OpenCV's peripheral theta clipping.
"""

import argparse
import csv
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))

STREAMS = {
    "cam_0": "left", "cam_1": "right",
    "cam_2": "right_side", "cam_3": "left_side",
}


@dataclass
class Camera:
    id: str
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray  # cam_0 -> this camera
    t: np.ndarray  # cam_0 -> this camera, meters
    width: int = 1920
    height: int = 1200

    @property
    def center(self):
        return -self.R.T @ self.t

    def project(self, points):
        """KB projection, supporting rays near/beyond 90 degrees."""
        points = np.asarray(points, dtype=float)
        rho = np.linalg.norm(points[..., :2], axis=-1)
        theta = np.arctan2(rho, points[..., 2])
        theta_d = theta.copy()
        for index, coefficient in enumerate(self.D):
            theta_d += coefficient * theta ** (2 * index + 3)
        scale = np.divide(theta_d, rho, out=np.zeros_like(rho), where=rho > 1e-12)
        normalized = points[..., :2] * scale[..., None]
        return normalized * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]

    def unproject(self, pixels):
        xy = (np.asarray(pixels, dtype=float) - self.K[:2, 2]) / [self.K[0, 0], self.K[1, 1]]
        radius = np.linalg.norm(xy, axis=-1)
        theta = radius.copy()
        for _ in range(25):
            value, derivative = theta.copy(), np.ones_like(theta)
            for index, coefficient in enumerate(self.D):
                power = 2 * index + 3
                value += coefficient * theta ** power
                derivative += power * coefficient * theta ** (power - 1)
            update = (value - radius) / derivative
            theta = np.clip(theta - update, 0, np.pi - 1e-5)
            if np.max(np.abs(update)) < 1e-12:
                break
        scale = np.divide(np.sin(theta), radius, out=np.ones_like(radius), where=radius > 1e-12)
        return np.concatenate((xy * scale[..., None], np.cos(theta)[..., None]), axis=-1)


def load_recording(directory):
    metadata = json.loads(next(directory.glob("meta_*.json")).read_text())
    calibration_path = directory / metadata["files"]["calibration_camera"]
    calibration = yaml.safe_load(calibration_path.read_text())
    cameras = {}
    for row in calibration["cameras"]:
        if row["distortion_model"] != "KB":
            raise ValueError("This tracker requires the recording's KB fisheye calibration")
        intrinsic = row["intrinsics"]
        K = np.array([[intrinsic["fx"], 0, intrinsic["cx"]],
                      [0, intrinsic["fy"], intrinsic["cy"]], [0, 0, 1.]])
        camera = Camera(
            row["id"], K, np.array([row["distortion"][f"k{i}"] for i in range(1, 5)]),
            np.array(row["extrinsics"]["rotation"], dtype=float),
            np.array(row["extrinsics"]["translation"], dtype=float) * 1e-3,
            row["image_width"], row["image_height"],
        )
        if not np.allclose(camera.R.T @ camera.R, np.eye(3), atol=1e-5):
            raise ValueError(f"Invalid camera rotation: {camera.id}")
        cameras[camera.id] = camera
    if not np.allclose(cameras["cam_0"].R, np.eye(3)) or np.linalg.norm(cameras["cam_0"].t) > 1e-6:
        raise ValueError("cam_0 must be the reference camera")
    timestamps = {
        camera_id: np.loadtxt(directory / metadata["files"][f"video_{stream}_pts"],
                              dtype=np.int64, skiprows=1, ndmin=1)
        for camera_id, stream in STREAMS.items()
    }
    return metadata, cameras, timestamps


def object_points(size):
    # The binding's corner order is left-bottom, right-bottom, right-top, left-top.
    return np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=float) * size / 2


def look_at(direction):
    z = np.asarray(direction) / np.linalg.norm(direction)
    up = np.array([0., 1., 0.])
    if abs(np.dot(up, z)) > 0.95:
        up = np.array([1., 0., 0.])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


def triangulate(observations, cameras):
    rays = [cameras[o["camera"]].unproject(o["corners"]) @ cameras[o["camera"]].R
            for o in observations]
    origins = [cameras[o["camera"]].center for o in observations]
    points = []
    for corner in range(4):
        normals = [np.eye(3) - np.outer(ray[corner], ray[corner]) for ray in rays]
        A = np.sum(normals, axis=0)
        if np.linalg.cond(A) > 1e7:
            raise ValueError("Insufficient triangulation angle")
        point = np.linalg.solve(A, np.sum([N @ origin for N, origin in zip(normals, origins)], axis=0))
        if any(np.dot(point - origin, ray[corner]) <= 0 for ray, origin in zip(rays, origins)):
            raise ValueError("Triangulation is behind an observed ray")
        points.append(point)
    return np.array(points)


def rigid_fit(object_xyz, measured_xyz):
    U, _, Vt = np.linalg.svd((object_xyz - object_xyz.mean(0)).T @ (measured_xyz - measured_xyz.mean(0)))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, measured_xyz.mean(0) - R @ object_xyz.mean(0)


def monocular_candidates(observation, camera, points):
    rays = camera.unproject(observation["corners"])
    virtual_R = look_at(rays.mean(0))
    virtual = rays @ virtual_R.T
    pixels = np.ascontiguousarray(virtual[:, :2] / virtual[:, 2:])
    result = cv2.solvePnPGeneric(points, pixels, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    if result[0]:
        for rvec, tvec in zip(result[1], result[2]):
            R_cam_tag = virtual_R.T @ cv2.Rodrigues(rvec)[0]
            t_cam_tag = virtual_R.T @ tvec.ravel()
            candidates.append((camera.R.T @ R_cam_tag, camera.R.T @ (t_cam_tag - camera.t)))
    return candidates


def project_pose(camera, R, t, points):
    world = points @ R.T + t
    return camera.project(world @ camera.R.T + camera.t)


def pose_residual(parameters, observations, cameras, points, weighted=True):
    R = Rotation.from_rotvec(parameters[:3]).as_matrix()
    residuals = []
    for observation in observations:
        difference = project_pose(cameras[observation["camera"]], R, parameters[3:], points) - observation["corners"]
        weight = 0.45 if observation["source"] == "flow" else 1.
        residuals.extend((difference * (weight if weighted else 1.)).ravel())
    return np.asarray(residuals)


def fit_pose(observations, cameras, size, previous=None, max_error=2.5):
    if not observations:
        return None
    points = object_points(size)
    candidates = []
    edge_sizes = []
    for pair in itertools.combinations(observations, 2):
        try:
            X = triangulate(pair, cameras)
            edges = np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1)
            if np.all((edges > size * 0.55) & (edges < size * 1.7)):
                candidates.append(rigid_fit(points, X))
                edge_sizes.extend(edges.tolist())
        except (ValueError, np.linalg.LinAlgError):
            pass
    for observation in observations:
        candidates.extend(monocular_candidates(observation, cameras[observation["camera"]], points))
    if previous is not None:
        candidates.append(previous)
    scored = []
    for R, t in candidates:
        if not np.all(np.isfinite(t)) or not 0.03 < np.linalg.norm(t) < 20:
            continue
        parameters = np.r_[Rotation.from_matrix(R).as_rotvec(), t]
        residual = pose_residual(parameters, observations, cameras, points, weighted=False).reshape(-1, 4, 2)
        camera_errors = np.sqrt(np.mean(residual ** 2, axis=(1, 2)))
        # Median over views prevents a single mis-tracked camera dominating initialization.
        score = np.median(camera_errors) + 0.1 * np.mean(np.minimum(camera_errors, 20))
        if previous is not None and len(observations) == 1:
            # A small planar marker has two near-equivalent monocular orientations.
            score += 0.3 * Rotation.from_matrix(previous[0].T @ R).magnitude()
        scored.append((score, parameters))
    if not scored:
        return None
    parameters = min(scored, key=lambda row: row[0])[1]
    kept = list(observations)
    for _ in range(len(observations) + 1):
        result = least_squares(
            pose_residual, parameters, args=(kept, cameras, points), loss="soft_l1",
            f_scale=0.8, max_nfev=60, xtol=1e-7, ftol=1e-7, gtol=1e-7,
            x_scale=np.array([1., 1., 1., 0.2, 0.2, 0.2]),
        )
        parameters = result.x
        residual = pose_residual(parameters, kept, cameras, points, weighted=False).reshape(-1, 4, 2)
        errors = np.sqrt(np.mean(residual ** 2, axis=(1, 2)))
        worst = int(np.argmax(errors))
        if len(kept) > 1 and errors[worst] > max_error:
            kept.pop(worst)
            continue
        break
    rms = float(np.sqrt(np.mean(pose_residual(parameters, kept, cameras, points, weighted=False) ** 2)))
    if rms > max_error or not np.all(np.isfinite(parameters)):
        return None
    R = Rotation.from_rotvec(parameters[:3]).as_matrix()
    t = parameters[3:]
    center = np.mean(points @ R.T + t, axis=0)
    for observation in kept:
        camera = cameras[observation["camera"]]
        direction = camera.unproject(np.mean(observation["corners"], axis=0)[None])[0]
        if np.dot(camera.R @ center + camera.t, direction) <= 0:
            return None
    return {"R": R, "t": t, "rms": rms, "observations": kept,
            "triangulated_edges_m": edge_sizes}


def local_detection(gray, camera, last_corners, detector, tag_id):
    direction = camera.unproject(last_corners).mean(0)
    virtual_R = look_at(direction)
    side, focal = 256, 750.
    yy, xx = np.indices((side, side), dtype=float)
    rays = np.stack(((xx - side / 2) / focal, (yy - side / 2) / focal, np.ones_like(xx)), axis=-1)
    raw_pixels = camera.project(rays @ virtual_R)
    virtual = cv2.remap(gray, raw_pixels[..., 0].astype(np.float32),
                        raw_pixels[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    choices = [d for d in detector.detect(virtual) if d["id"] == tag_id and d["margin"] >= 25]
    if len(choices) != 1:
        return None
    detection = choices[0]
    corners = detection["lb-rb-rt-lt"]
    rays = np.column_stack(((corners[:, 0] - side / 2) / focal,
                            (corners[:, 1] - side / 2) / focal, np.ones(4)))
    corners = camera.project(rays @ virtual_R)
    if np.any(corners < 3) or np.any(corners[:, 0] >= camera.width - 3) or np.any(corners[:, 1] >= camera.height - 3):
        return None
    return {"corners": corners.tolist(), "margin": float(detection["margin"]),
            "hamming": detection["hamming"], "source": "rectified"}


def flow_corners(previous_gray, gray, corners):
    previous = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    current, status, error = cv2.calcOpticalFlowPyrLK(previous_gray, gray, previous, None, **options)
    if current is None or not status.all() or np.max(error) > 25:
        return None
    returned, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, current, None, **options)
    if returned is None or not back_status.all() or np.max(np.linalg.norm(returned - previous, axis=2)) > 0.8:
        return None
    polygon = current.reshape(4, 2)
    if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 100:
        return None
    return polygon


def collect_observations(directory, output, metadata, cameras, timestamps, args):
    from apriltag import apriltag

    detector = apriltag(args.family, threads=2, maxhamming=1, decimate=1., refine_edges=True)
    summaries = {}
    for camera_id, stream in STREAMS.items():
        cache = output / f"observations_{camera_id}.json"
        expected = {"recording": str(directory.resolve()),
                    "family": args.family, "id": args.tag_id, "frames": args.max_frames,
                    "flow_max_frames": args.flow_max_frames, "tracking_version": 2}
        if args.reuse_detections and cache.exists():
            payload = json.loads(cache.read_text())
            if payload.get("config") == expected:
                summaries[camera_id] = payload["frames"]
                print(f"Reused detections for {camera_id}", flush=True)
                continue
            raise ValueError(f"Cached detection settings differ: {cache}")
        path = directory / metadata["files"][f"video_{stream}"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        camera = cameras[camera_id]
        previous_gray, previous_corners, rectification_corners, last_decoded = None, None, None, -100
        observations = []
        limit = min(len(timestamps[camera_id]), args.max_frames or len(timestamps[camera_id]))
        for index in range(limit):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Early video end: {path}, frame {index}")
            if image.shape[:2] != (camera.height, camera.width):
                raise ValueError("Video resolution differs from calibration")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            matches = [d for d in detector.detect(gray) if d["id"] == args.tag_id and d["margin"] >= 25]
            observation = None
            if len(matches) == 1:
                detection = matches[0]
                observation = {"corners": detection["lb-rb-rt-lt"].tolist(),
                               "margin": float(detection["margin"]),
                               "hamming": detection["hamming"], "source": "decoded"}
            if observation is None and rectification_corners is not None and index - last_decoded <= 8:
                observation = local_detection(gray, camera, rectification_corners, detector, args.tag_id)
            if observation is not None:
                last_decoded = index
            elif previous_gray is not None and previous_corners is not None and index - last_decoded <= args.flow_max_frames:
                tracked = flow_corners(previous_gray, gray, previous_corners)
                if tracked is not None:
                    observation = {"corners": tracked.tolist(), "margin": None,
                                   "hamming": None, "source": "flow"}
            if observation is not None:
                observation.update(camera=camera_id, frame=index,
                                   timestamp_us=int(timestamps[camera_id][index]))
                previous_corners = np.array(observation["corners"])
                rectification_corners = previous_corners.copy()
            else:
                # KLT coordinates must belong to previous_gray, never an older frame.
                previous_corners = None
            previous_gray = gray
            observations.append(observation)
            if index % 120 == 0:
                detected = sum(o is not None for o in observations)
                print(f"{camera_id} {index + 1}/{limit}: {detected} observed", flush=True)
        capture.release()
        payload = {"config": expected, "frames": observations}
        cache.write_text(json.dumps(payload))
        summaries[camera_id] = observations
    return summaries


class AdaptiveFilter:
    """Causal position and SO(3) filtering with a motion-dependent cutoff."""
    def __init__(self, cutoff=4., beta=8.):
        self.cutoff, self.beta = cutoff, beta
        self.R = self.t = self.raw_t = self.raw_R = self.timestamp = None
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def update(self, R, t, timestamp):
        if self.timestamp is None or timestamp - self.timestamp > 0.2:
            self.R, self.t, self.raw_R, self.raw_t = R.copy(), t.copy(), R.copy(), t.copy()
            self.velocity[:] = 0
            self.angular_velocity[:] = 0
        else:
            dt = timestamp - self.timestamp
            speed = (t - self.raw_t) / dt
            angular = Rotation.from_matrix(self.raw_R.T @ R).as_rotvec() / dt
            derivative_alpha = 1 / (1 + 1 / (2 * np.pi * 2 * dt))
            self.velocity += derivative_alpha * (speed - self.velocity)
            self.angular_velocity += derivative_alpha * (angular - self.angular_velocity)
            alpha_t = 1 / (1 + 1 / (2 * np.pi * (self.cutoff + self.beta * np.linalg.norm(self.velocity)) * dt))
            alpha_r = 1 / (1 + 1 / (2 * np.pi * (self.cutoff + 0.8 * np.linalg.norm(self.angular_velocity)) * dt))
            self.t += alpha_t * (t - self.t)
            delta = Rotation.from_matrix(self.R.T @ R).as_rotvec()
            self.R = self.R @ Rotation.from_rotvec(delta * alpha_r).as_matrix()
            self.raw_t, self.raw_R = t.copy(), R.copy()
        self.timestamp = timestamp
        return self.R.copy(), self.t.copy()


def associate_frames(observations, timestamps, max_sync_us, limit):
    frames = []
    for index, timestamp in enumerate(timestamps["cam_0"][:limit]):
        matched = []
        for camera_id, camera_times in timestamps.items():
            insertion = int(np.searchsorted(camera_times, timestamp))
            choices = [j for j in (insertion - 1, insertion) if 0 <= j < len(camera_times)]
            nearest = min(choices, key=lambda j: abs(int(camera_times[j]) - int(timestamp)))
            if abs(int(camera_times[nearest]) - int(timestamp)) <= max_sync_us and nearest < len(observations[camera_id]):
                observation = observations[camera_id][nearest]
                if observation is not None:
                    matched.append(observation)
        frames.append(matched)
    return frames


def solve_sequence(frames, timestamps, cameras, args):
    results = []
    filter_state = AdaptiveFilter(args.cutoff)
    previous = None
    last_valid = -np.inf
    for index, observations in enumerate(frames):
        timestamp = (int(timestamps["cam_0"][index]) - int(timestamps["cam_0"][0])) * 1e-6
        seed = previous if timestamp - last_valid < 0.2 else None
        pose = fit_pose(observations, cameras, args.tagsize, seed, args.max_reprojection)
        result = {"frame": index, "time_s": timestamp,
                  "timestamp_us": int(timestamps["cam_0"][index]), "status": "lost"}
        if pose is not None:
            R, t = pose["R"], pose["t"]
            # Reject physically implausible jumps, rather than dragging the filter across them.
            jump_ok = True
            if previous is not None and timestamp - last_valid < 0.1:
                dt = timestamp - last_valid
                translation_jump = np.linalg.norm(t - previous[1])
                rotation_jump = Rotation.from_matrix(previous[0].T @ R).magnitude()
                jump_ok = translation_jump < 0.025 + 3. * dt and rotation_jump < np.deg2rad(15) + 6 * dt
            if jump_ok:
                kept = pose["observations"]
                decoded = sum(o["source"] != "flow" for o in kept)
                # Undecoded corners require agreement between at least two cameras.
                if decoded or len(kept) >= 2:
                    filtered_R, filtered_t = filter_state.update(R, t, timestamp)
                    result.update(status="measured" if decoded else "flow_only",
                                  R=R, t=t, filtered_R=filtered_R, filtered_t=filtered_t,
                                  rms=pose["rms"], n_views=len(kept), n_decoded=decoded,
                                  cameras=[o["camera"] for o in kept],
                                  edge_sizes=pose["triangulated_edges_m"])
                    previous = (R, t)
                    last_valid = timestamp
        # No fabricated poses during a long dropout; NaNs in the exported arrays.
        results.append(result)
        if index % 120 == 0:
            valid = sum(r["status"] != "lost" for r in results)
            print(f"pose {index + 1}/{len(frames)}: {valid} valid", flush=True)
    return results


def matrix(R, t):
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = R, t
    return transform


def export_results(output, results, cameras, timestamps, observations, args):
    count = len(results)
    raw = np.full((count, 4, 4), np.nan)
    filtered = raw.copy()
    header = ["frame", "timestamp_us", "time_s", "status", "n_views", "n_decoded", "reprojection_rms_px"]
    header += [f"{prefix}_{axis}" for prefix in ("raw_t_m", "filtered_t_m") for axis in "xyz"]
    header += [f"{prefix}_{axis}" for prefix in ("raw_q", "filtered_q") for axis in ("x", "y", "z", "w")]
    header += ["cameras"]
    previous_quaternions = [None, None]
    with (output / "poses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for result in results:
            valid = result["status"] != "lost"
            row = [result["frame"], result["timestamp_us"], result["time_s"], result["status"],
                   result.get("n_views", 0), result.get("n_decoded", 0), result.get("rms", np.nan)]
            if valid:
                i = result["frame"]
                raw[i] = matrix(result["R"], result["t"])
                filtered[i] = matrix(result["filtered_R"], result["filtered_t"])
                quaternions = []
                for slot, rotation in enumerate((result["R"], result["filtered_R"])):
                    quaternion = Rotation.from_matrix(rotation).as_quat()
                    if previous_quaternions[slot] is not None and np.dot(quaternion, previous_quaternions[slot]) < 0:
                        quaternion = -quaternion
                    previous_quaternions[slot] = quaternion
                    quaternions.extend(quaternion)
                row += [*result["t"], *result["filtered_t"], *quaternions]
            else:
                row += [np.nan] * 14
            writer.writerow(row + [";".join(result.get("cameras", []))])
    np.savez_compressed(output / "poses.npz", T_cam0_tag_raw=raw,
                        T_cam0_tag_filtered=filtered,
                        timestamp_us=np.array([r["timestamp_us"] for r in results]),
                        status=np.array([r["status"] for r in results]))
    valid = [row for row in results if row["status"] != "lost"]
    edges = [edge for row in valid for edge in row["edge_sizes"]]
    sync = {}
    for camera_id, times in timestamps.items():
        differences = []
        for timestamp in timestamps["cam_0"][:count]:
            index = int(np.searchsorted(times, timestamp))
            differences.append(min(abs(int(times[j]) - int(timestamp))
                                   for j in (index - 1, index) if 0 <= j < len(times)))
        sync[camera_id] = {"median_us": float(np.median(differences)), "max_us": int(max(differences))}
    summary = {
        "recording": str(args.recording.resolve()), "family": args.family, "id": args.tag_id,
        "tagsize_m": args.tagsize, "reference_frame": "cam_0 / Color_Left optical frame",
        "transform_convention": "p_cam0 = R_cam0_tag @ p_tag + t_cam0_tag",
        "extrinsics_translation_input_unit": "mm", "output_translation_unit": "m",
        "camera_model": "KB: theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9",
        "frames": count, "valid_frames": len(valid), "lost_frames": count - len(valid),
        "decoded_pose_frames": sum(r["status"] == "measured" for r in results),
        "flow_only_pose_frames": sum(r["status"] == "flow_only" for r in results),
        "view_count_histogram": {str(n): sum(r.get("n_views", 0) == n for r in results) for n in range(5)},
        "reprojection_rms_px_median": float(np.median([r["rms"] for r in valid])) if valid else None,
        "reprojection_rms_px_p95": float(np.percentile([r["rms"] for r in valid], 95)) if valid else None,
        "triangulated_edge_median_mm": float(np.median(edges) * 1000) if edges else None,
        "timestamp_sync": sync,
        "cameras": {key: {"stream": STREAMS[key], "center_in_cam0_m": camera.center.tolist(),
                           "observations": {source: sum(o is not None and o["source"] == source
                                                        for o in observations[key])
                                            for source in ("decoded", "rectified", "flow")}}
                    for key, camera in cameras.items()},
        "limitations": ["Reprojection error measures internal consistency, not absolute 3D accuracy.",
                        "A four-centimeter tag limits monocular tilt observability.",
                        "IMU is not fused; output is relative to cam_0, not a fixed world frame.",
                        "The causal filter introduces a small motion-dependent delay.",
                        "Long dropouts remain lost/NaN; optical-flow-only frames are explicitly marked."],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot_results(output, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.array([row["time_s"] for row in results])
    raw = np.array([row.get("t", [np.nan] * 3) for row in results])
    filtered = np.array([row.get("filtered_t", [np.nan] * 3) for row in results])
    figure, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
    for dimension, axis in enumerate(axes[:3]):
        axis.plot(times, raw[:, dimension], alpha=.45, lw=.8, label="joint estimate")
        axis.plot(times, filtered[:, dimension], lw=1.3, label="causal filtered")
        axis.set_ylabel(f"{'XYZ'[dimension]} (m)")
        axis.grid(alpha=.25)
    axes[0].legend()
    axes[3].plot(times, [row.get("rms", np.nan) for row in results], lw=.9)
    axes[3].set_ylabel("reproj RMS (px)")
    axes[4].step(times, [row.get("n_views", 0) for row in results], where="mid")
    axes[4].set_ylabel("camera views")
    axes[4].set_yticks(range(5))
    axes[4].set_xlabel("Recording time (s)")
    figure.suptitle("EGO AprilTag tracking in cam_0: raw and causal filtered")
    figure.savefig(output / "trajectory.png", dpi=150)
    plt.close(figure)
    valid = [row for row in results if row["status"] != "lost"]
    if valid:
        initial = valid[0]["R"]
        figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
        for key, label, alpha in (("R", "joint estimate", .4),
                                  ("filtered_R", "causal filtered", 1.)):
            vectors = np.array([
                np.rad2deg(Rotation.from_matrix(initial.T @ row[key]).as_rotvec())
                if row["status"] != "lost" else [np.nan] * 3 for row in results
            ])
            for dimension, axis in enumerate(axes):
                axis.plot(times, vectors[:, dimension], label=label, alpha=alpha, lw=1.)
                axis.set_ylabel(f"rotvec {'XYZ'[dimension]} (deg)")
                axis.grid(alpha=.25)
        axes[0].legend()
        axes[-1].set_xlabel("Recording time (s)")
        figure.suptitle("Tag orientation relative to first valid orientation (rotation vector, not Euler angles)")
        figure.savefig(output / "orientation.png", dpi=150)
        plt.close(figure)


def render_video(directory, output, metadata, cameras, frames, results, args):
    writer = cv2.VideoWriter(str(output / "tracking_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             30., (1280, 800))
    if not writer.isOpened():
        raise RuntimeError("Cannot open preview video writer")
    # Render streams separately to bound memory and H.264 decoder thread use.
    preview_directory = output / "camera_previews"
    preview_directory.mkdir(exist_ok=True)
    axes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]]) * args.tagsize * .6
    for camera_id, stream in STREAMS.items():
        cap = cv2.VideoCapture(str(directory / metadata["files"][f"video_{stream}"]))
        camera_writer = cv2.VideoWriter(str(preview_directory / f"{camera_id}.mp4"),
                                        cv2.VideoWriter_fourcc(*"mp4v"), 30., (640, 400))
        if not camera_writer.isOpened():
            raise RuntimeError(f"Cannot open {camera_id} preview writer")
        for index, result in enumerate(results):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Preview video ended early: {stream}")
            image = cv2.resize(image, (640, 400), interpolation=cv2.INTER_AREA)
            found = next((o for o in frames[index] if o["camera"] == camera_id), None)
            if found is not None:
                corners = np.rint(np.array(found["corners"]) / 3).astype(np.int32)
                color = (0, 255, 0) if found["source"] != "flow" else (0, 180, 255)
                cv2.polylines(image, [corners], True, color, 1, cv2.LINE_AA)
            if result["status"] != "lost":
                R, t = result["filtered_R"], result["filtered_t"]
                camera = cameras[camera_id]
                pixels = project_pose(camera, R, t, axes) / 3
                if np.all(np.isfinite(pixels)) and np.all(np.abs(pixels) < 5000):
                    for j, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0)), 1):
                        cv2.line(image, tuple(np.rint(pixels[0]).astype(int)),
                                 tuple(np.rint(pixels[j]).astype(int)), color, 2, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (640, 50), (12, 12, 12), -1)
            cv2.putText(image, f'{camera_id} / {stream} | {result["time_s"]:.2f}s',
                        (10, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            if result["status"] != "lost":
                t = result["filtered_t"]
                text = f'{result["status"]} views={result["n_views"]} xyz(m)={t[0]:.3f},{t[1]:.3f},{t[2]:.3f}'
            else:
                text = 'LOST - no reliable pose'
            cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (60, 220, 255), 1, cv2.LINE_AA)
            camera_writer.write(image)
        cap.release()
        camera_writer.release()
        print(f"Rendered {camera_id} preview", flush=True)
    captures = {key: cv2.VideoCapture(str(preview_directory / f"{key}.mp4")) for key in STREAMS}
    for index in range(len(results)):
        cells = {}
        for key, capture in captures.items():
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read preview frame {key}/{index}")
            cells[key] = image
        mosaic = np.vstack((np.hstack((cells["cam_0"], cells["cam_1"])),
                            np.hstack((cells["cam_3"], cells["cam_2"]))))
        writer.write(mosaic)
        if index in (0, len(results) // 2):
            cv2.imwrite(str(output / f"tracking_frame_{index:04d}.jpg"), mosaic)
    writer.release()
    for capture in captures.values():
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--tagsize", type=float, required=True, help="Detection-square side in meters")
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ego_multicam")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the entire recording")
    parser.add_argument("--max-sync-us", type=int, default=1000)
    parser.add_argument("--max-reprojection", type=float, default=2.5)
    parser.add_argument("--flow-max-frames", type=int, default=15,
                        help="Maximum frames tracked without decoding; flow-only poses require two views")
    parser.add_argument("--cutoff", type=float, default=4., help="Minimum causal filter cutoff, Hz")
    parser.add_argument("--reuse-detections", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.tagsize) or args.tagsize <= 0 or args.max_frames < 0:
        parser.error("tagsize must be positive and max-frames nonnegative")
    if args.flow_max_frames < 0 or args.tag_id < 0 or args.max_sync_us < 0:
        parser.error("flow-max-frames, tag-id and max-sync-us must be nonnegative")
    if not np.isfinite(args.cutoff) or args.cutoff <= 0 or not np.isfinite(args.max_reprojection) or args.max_reprojection <= 0:
        parser.error("cutoff and max-reprojection must be positive finite values")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata, cameras, timestamps = load_recording(args.recording)
    started = time.monotonic()
    observations = collect_observations(args.recording, args.output, metadata, cameras, timestamps, args)
    limit = min(len(timestamps["cam_0"]), args.max_frames or len(timestamps["cam_0"]))
    frames = associate_frames(observations, timestamps, args.max_sync_us, limit)
    results = solve_sequence(frames, timestamps, cameras, args)
    summary = export_results(args.output, results, cameras, timestamps, observations, args)
    plot_results(args.output, results)
    if not args.no_video:
        render_video(args.recording, args.output, metadata, cameras, frames, results, args)
    print(json.dumps({"frames": summary["frames"], "valid": summary["valid_frames"],
                      "rms_median_px": summary["reprojection_rms_px_median"],
                      "elapsed_s": round(time.monotonic() - started, 1),
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
