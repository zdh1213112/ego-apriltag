"""Geometry, missing-frame and estimate handling for reference-tag trails."""

import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_ego_alignment import audit_alignment
from ego_trails import ReferenceTrail
from track_ego_apriltag import Camera, matrix


class TrailTests(unittest.TestCase):
    def poses(self, count):
        poses = np.tile(np.eye(4), (count, 1, 1))
        poses[:, 0, 3] = np.arange(count) * .05
        poses[:, 2, 3] = 1.
        return poses

    def test_time_window_fades_and_does_not_bridge_loss(self):
        trail = ReferenceTrail(self.poses(6), np.arange(6) * 100000,
                               ["measured", "flow_only", "lost", "measured", "interpolated", "measured"], 7, .5)
        segments = trail.segments(5)
        self.assertEqual([(a, b) for a, b, _, _ in segments], [(0, 1), (3, 4), (4, 5)])
        self.assertEqual([estimated for _, _, _, estimated in segments], [False, True, True])
        self.assertLess(segments[0][2], segments[-1][2])
        trail.seconds = .11
        self.assertEqual([(a, b) for a, b, _, _ in trail.segments(5)], [(4, 5)])

    def test_draw_uses_camera_extrinsics_and_marks_estimated_points(self):
        camera = Camera("cam_1", np.array([[100., 0, 80.], [0, 100., 60.], [0, 0, 1.]]),
                        np.zeros(4), np.eye(3), np.array([-.1, 0., 0.]), width=160, height=120)
        trail = ReferenceTrail(self.poses(3), np.arange(3) * 100000, ["measured"] * 3, 7, 1.)
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        trail.draw(image, camera, 2)
        np.testing.assert_array_equal(image[60, 80], [255, 220, 40])
        np.testing.assert_array_equal(image[60, 90], [0, 0, 0])
        trail.status[-1] = "interpolated"
        trail.draw(image, camera, 2)
        np.testing.assert_array_equal(image[60, 80], [0, 165, 255])

    def test_export_keeps_camera_relative_positions_and_chosen_reference(self):
        poses = self.poses(3)
        poses[1] = np.nan
        trail = ReferenceTrail(poses, np.arange(3) * 100000, ["measured", "lost", "interpolated"], 7, 1.)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "trail.csv"
            trail.export(path)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(rows[2]["reference_tag_id"], "7")
        self.assertEqual(rows[2]["coordinate_frame"], "cam_0")
        self.assertAlmostEqual(float(rows[2]["x_m"]), .1)
        self.assertEqual(rows[2]["status"], "interpolated")
        self.assertTrue(np.isnan(float(rows[1]["x_m"])))


class AlignmentAuditTests(unittest.TestCase):
    def test_relative_stability_uses_selected_reference_and_excludes_flow(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            offset = matrix(np.eye(3), np.array([.03, 0., .01]))
            (root / "summary.json").write_text(json.dumps({"tag_ids": [1, 0]}))
            (root / "tag_alignment.json").write_text(json.dumps({
                "reference_tag_id": 1, "T_reference_tag": {"1": np.eye(4).tolist(), "0": offset.tolist()},
                "paths": {"1": [1], "0": [1, 0]}, "unaligned_tag_ids": []}))
            reference_poses = np.array([matrix(Rotation.from_rotvec([.02 * i, .01, 0.]).as_matrix(),
                                               np.array([i * .02, 0., .4])) for i in range(8)])
            relative = np.tile(offset, (8, 1, 1))
            relative[4:, 0, 3] += .002
            for tag_id, poses in ((1, reference_poses), (0, reference_poses @ relative)):
                directory = root / f"tag_{tag_id}"
                directory.mkdir()
                status = np.array(["measured"] * 8)
                if tag_id == 0:
                    status[1] = "flow_only"
                np.savez(directory / "poses.npz", T_cam0_tag_raw=poses, status=status,
                         timestamp_us=np.arange(8) * 33333)
                with (directory / "poses.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["n_views"])
                    writer.writerows([[2]] * 8)
                (directory / "summary.json").write_text(json.dumps({
                    "valid_frames": 8, "reprojection_rms_px_median": .5,
                    "reprojection_rms_px_p95": .8, "view_count_histogram": {"2": 8}}))
            report = audit_alignment(root)
        pair = report["reference_pairs"][0]
        self.assertEqual(report["reference_tag_id"], 1)
        self.assertEqual(pair["simultaneous_multiview_measured_frames"], 7)
        self.assertEqual(pair["temporal_half_frames"], [3, 4])
        self.assertAlmostEqual(pair["translation_p95_mm"], 2., places=8)
        self.assertAlmostEqual(pair["temporal_half_difference_mm"], 2., places=8)
        self.assertLess(pair["rotation_p95_deg"], 1e-8)
        self.assertIn("no ground-truth", report["interpretation"])
