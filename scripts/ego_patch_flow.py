"""Short-range patch tracking from decoded tags bordering an offline pose gap."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from track_ego_apriltag import STREAMS, fit_rigid_detections, matrix, project_pose


def affine_corners(previous, current, corners):
    """Align the textured tag patch both ways when individual corner LK fails."""
    initial = np.eye(2, 3, dtype=np.float32)
    options = (cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 35, 1e-4), None, 5)
    try:
        score, forward = cv2.findTransformECC(previous, current, initial.copy(), *options)
        reverse_score, reverse = cv2.findTransformECC(current, previous, initial.copy(), *options)
    except cv2.error:
        return None
    if min(score, reverse_score) < .76 or not np.isfinite(forward).all() or not np.isfinite(reverse).all():
        return None
    scales = np.linalg.svd(forward[:, :2], compute_uv=False)
    if np.linalg.det(forward[:, :2]) <= 0 or scales.min() < .76 or scales.max() > 1.32:
        return None
    tracked = corners @ forward[:, :2].T + forward[:, 2]
    returned = tracked @ reverse[:, :2].T + reverse[:, 2]
    if np.max(np.linalg.norm(returned - corners, axis=1)) > .9:
        return None
    return tracked


def track_patch(previous, current, corners):
    """Locate the tag by correlation, then fit its corners with subpixel round-trip checks."""
    height, width = previous.shape
    low = np.floor(corners.min(axis=0)).astype(int) - 4
    high = np.ceil(corners.max(axis=0)).astype(int) + 5
    size = high - low
    if (low < 0).any() or (high > [width, height]).any() or size.min() < 11 or size.max() > 160:
        return None
    x, y = low
    right, bottom = high
    template = previous[y:bottom, x:right]
    search_lo = np.maximum(0, low - 100)
    search_hi = np.minimum([width, height], high + 100)
    search = current[search_lo[1]:search_hi[1], search_lo[0]:search_hi[0]]
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, peak = cv2.minMaxLoc(scores)
    if score < .58:
        return None
    delta = search_lo + peak - low
    # Crop the same extent around both matched locations, including at image borders.
    border_lo = np.maximum.reduce([low - 56, np.zeros(2, int), -delta])
    border_hi = np.minimum.reduce([high + 56, np.array([width, height]), np.array([width, height]) - delta])
    x, y = border_lo
    right, bottom = border_hi
    dx, dy = delta
    a = previous[y:bottom, x:right]
    b = current[y + dy:bottom + dy, x + dx:right + dx]
    local = (corners - border_lo).astype(np.float32)
    opts = dict(winSize=(17, 17), maxLevel=1, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
    tracked, status, error = cv2.calcOpticalFlowPyrLK(a, b, local, local.copy(), **opts)
    if tracked is not None and status.all() and error.max() <= 30:
        returned, status, _ = cv2.calcOpticalFlowPyrLK(b, a, tracked, local.copy(), **opts)
        if returned is not None and status.all() and np.max(np.linalg.norm(returned - local, axis=1)) <= .8:
            area = cv2.contourArea(tracked, oriented=True)
            old_area = cv2.contourArea(local, oriented=True)
            if (np.max(np.linalg.norm(tracked - local, axis=1)) <= 9 and cv2.isContourConvex(tracked)
                    and abs(area) >= 100 and old_area and .55 < area / old_area < 1.8):
                return tracked + border_lo + delta
    # Restrict ECC to the tag and a narrow border so background motion cannot dominate.
    x, y = np.maximum(border_lo, low - 6)
    right, bottom = np.minimum(border_hi, high + 6)
    tracked = affine_corners(previous[y:bottom, x:right], current[y + dy:bottom + dy, x + dx:right + dx],
                             corners - [x, y])
    return None if tracked is None else tracked + [x + dx, y + dy]


def gap_anchors(rows):
    anchors = np.array([i for i, row in enumerate(rows)
                        if row["status"] == "measured" and row["n_views"] >= 2], dtype=int)
    runs, priors = {}, {}
    if not len(anchors):
        return runs, priors
    for i, row in enumerate(rows):
        if row["status"] != "lost":
            continue
        j = anchors.searchsorted(i)
        adjacent = anchors[max(0, j - 1):min(j + 1, len(anchors))]
        for anchor in adjacent:
            if abs(i - anchor) <= 15 and abs(row["time_s"] - rows[anchor]["time_s"]) <= .51:
                direction = 1 if i > anchor else -1
                runs.setdefault((int(anchor), direction), set()).add(i)
        a, b = rows[adjacent[0]], rows[adjacent[-1]]
        if len(adjacent) == 1:
            priors[i] = matrix(a["R"], a["t"])
        else:
            times = [a["time_s"], b["time_s"]]
            fraction = (row["time_s"] - times[0]) / (times[1] - times[0])
            rotation = Slerp(times, Rotation.from_matrix([a["R"], b["R"]]))(row["time_s"])
            priors[i] = matrix(rotation.as_matrix(), (1 - fraction) * a["t"] + fraction * b["t"])
    return runs, priors


def decoded_seeds(gray, camera, anchor, points, detect, max_error):
    projected = {tag: project_pose(camera, anchor["R"], anchor["t"], xyz) for tag, xyz in points.items()}
    poly = np.concatenate(list(projected.values()))
    low = np.maximum(0, np.floor(poly.min(axis=0) - 16).astype(int))
    high = np.minimum(gray.shape[::-1], np.ceil(poly.max(axis=0) + 17).astype(int))
    if (high - low).min() < 12:
        return {}
    crop = np.ascontiguousarray(gray[low[1]:high[1], low[0]:high[0]])
    # Pattern recovery has already read these exact pixels at the synchronized
    # anchor. They also seed flow when the native detector misses a blurred tag.
    seeds = {obs["tag_id"]: obs["corners"] for obs in anchor.get("pattern_observations", [])
             if obs["camera"] == camera.id}
    for obs in detect(crop):
        tag = obs["id"]
        corners = np.asarray(obs["lb-rb-rt-lt"]) + low
        if tag in points and obs["margin"] >= 25 and np.sqrt(np.mean((corners - projected[tag]) ** 2)) <= max_error:
            seeds[tag] = corners
    return seeds


def collect_tracks(capture, camera, camera_times, rows, runs, points, detect, max_error, max_sync_us, hits):
    if not len(camera_times):
        return
    for (anchor, direction), targets in sorted(runs.items()):
        end = max(targets, key=lambda index: abs(index - anchor))
        start, stop = sorted((anchor, end))
        ref_times = np.array([row["timestamp_us"] for row in rows[start:stop + 1]])
        right = np.clip(np.searchsorted(camera_times, ref_times), 0, len(camera_times) - 1)
        left = np.maximum(right - 1, 0)
        indices = np.where(abs(camera_times[left] - ref_times) <= abs(camera_times[right] - ref_times), left, right)
        valid = abs(camera_times[indices] - ref_times) <= max_sync_us
        # Follow only a contiguous synchronized sequence starting at the decoded anchor.
        ordered = range(0, len(indices)) if direction == 1 else range(len(indices) - 1, -1, -1)
        pairs = []
        for k in ordered:
            if not valid[k] or (pairs and indices[k] != pairs[-1][1] + direction):
                break
            pairs.append((start + k, int(indices[k])))
        if len(pairs) < 2:
            continue
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, min(pairs[0][1], pairs[-1][1])):
            raise ValueError(f"Cannot seek {camera.id} for patch recovery")
        images = {}
        for frame, index in sorted(pairs):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot read {camera.id} frame {index} for patch recovery")
            images[frame] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        seeds = decoded_seeds(images[anchor], camera, rows[anchor], points, detect, max_error)
        for frame, _ in pairs[1:]:
            if not seeds:
                break
            tracked = {}
            for tag, corners in seeds.items():
                result = track_patch(images[frame - direction], images[frame], corners)
                if result is not None:
                    tracked[tag] = result
                    if frame in targets:
                        hits.setdefault(frame, []).append({
                            "camera": camera.id, "tag_id": tag, "corners": result,
                            "source": "forward_flow" if direction == 1 else "backward_flow",
                            "anchor": anchor,
                        })
            seeds = tracked


def supported_patch_fit(index, rows, fitted, candidates, max_error):
    near, bidirectional = set(), set()
    for kept in fitted:
        camera = kept["camera"]
        directions = set()
        for other in candidates:
            if other["camera"] != camera or other["tag_id"] != kept["tag_id"]:
                continue
            if np.sqrt(np.mean((other["corners"] - kept["corners"]) ** 2)) > max_error:
                continue
            directions.add(other["source"])
            anchor = other["anchor"]
            if abs(anchor - index) <= 4 and abs(rows[index]["time_s"] - rows[anchor]["time_s"]) <= .14:
                near.add(camera)
        if len(directions) == 2:
            bidirectional.add(camera)
    # Longer one-sided tracks can drift together even with low reprojection error.
    return len(near) >= 3 or len(bidirectional) >= 2


def recover_patch_gaps(rows, recording, metadata, cameras, timestamps, points, size, alignment,
                       detector, max_error, max_sync_us):
    if not np.array_equal([row["timestamp_us"] for row in rows], timestamps["cam_0"][:len(rows)]):
        raise ValueError("Patch recovery requires matching reference camera and pose timestamps")
    runs, priors = gap_anchors(rows)
    if not runs:
        return []
    hits = {}
    for camera_id, stream in STREAMS.items():
        capture = cv2.VideoCapture(str(recording / metadata["files"][f"video_{stream}"]))
        try:
            collect_tracks(capture, cameras[camera_id], timestamps[camera_id], rows, runs, points,
                           getattr(detector, "detect_local", detector.detect), max_error, max_sync_us, hits)
        finally:
            capture.release()
    additions = {}
    for index, candidates in hits.items():
        if len({r["camera"] for r in candidates}) < 2:
            continue
        fit = fit_rigid_detections(candidates, cameras, points, size, alignment, priors[index], max_error)
        if fit is None or not supported_patch_fit(index, rows, fit["observations"], candidates, max_error):
            continue
        additions[index] = {key: value for key, value in fit.items() if key != "observations"}
    for index, fit in additions.items():
        rows[index].update(fit, pose_source="patch_flow")
    return sorted(additions)
