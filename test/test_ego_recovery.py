"""Checks for recovery from duplicate IDs and bounded pose interpolation."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from track_ego_apriltag import Camera, matrix, object_points
from recover_ego_apriltag import (
    fill_short_pose_gaps, fit_rigid_detections, recovery_windows, reject_isolated_excursions, rigid_points,
)


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        K = np.array([[520., 0., 960.], [0., 520., 600.], [0., 0., 1.]])
        cameras = {f"cam_{i}": Camera(f"cam_{i}", K, np.zeros(4), np.eye(3), np.array([-.10 * i, 0., 0.]))
                   for i in range(2)}
        offsets = {0: np.eye(4), 1: matrix(Rotation.from_euler("y", -60, degrees=True).as_matrix(),
                                         np.array([.03, 0., .02]))}
        alignment = {"T_reference_tag": {str(key): value.tolist() for key, value in offsets.items()}}
        points = rigid_points(alignment, .03)
        pose = matrix(Rotation.from_rotvec([.1, -.15, .05]).as_matrix(), np.array([.05, .02, .45]))
        observations = []
        for tag_id, xyz in points.items():
            for camera in cameras.values():
                local = (xyz @ pose[:3, :3].T + pose[:3, 3]) @ camera.R.T + camera.t
                corners = cv2.fisheye.projectPoints(local.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                                   K, np.zeros(4))[0].reshape(4, 2)
                observations.append({"camera": camera.id, "tag_id": tag_id,
                                     "corners": corners, "source": "decoded"})
        return cameras, alignment, points, pose, observations

    def test_tag_layout_selects_the_target_in_the_presence_of_duplicate_ids(self):
        cameras, alignment, points, expected, observations = self.fixture()
        false_pose = matrix(np.eye(3), np.array([.45, .3, .2]))
        for camera in cameras.values():
            local = (object_points(.03) @ false_pose[:3, :3].T + false_pose[:3, 3]) @ camera.R.T + camera.t
            corners = cv2.fisheye.projectPoints(local.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                               camera.K, camera.D)[0].reshape(4, 2)
            observations.insert(0, {"camera": camera.id, "tag_id": 0, "corners": corners, "source": "decoded"})
        fitted = fit_rigid_detections(observations, cameras, points, .03, alignment)
        self.assertIsNotNone(fitted)
        self.assertEqual(fitted["source_tag_ids"], [0, 1])
        np.testing.assert_allclose(matrix(fitted["R"], fitted["t"]), expected, atol=1e-6)

    def test_flow_only_recovery_requires_two_cameras(self):
        cameras, alignment, points, expected, observations = self.fixture()
        for observation in observations:
            observation["source"] = "backward_flow"
        fitted = fit_rigid_detections(observations, cameras, points, .03, alignment, expected)
        self.assertEqual(fitted["status"], "flow_only")
        one_camera = [row for row in observations if row["camera"] == "cam_0"]
        self.assertIsNone(fit_rigid_detections(one_camera, cameras, points, .03, alignment, expected))

    def sequence(self, count=12):
        return [{"frame": index, "time_s": index / 30, "timestamp_us": 1000000 + index * 33333,
                 "status": "measured", "R": Rotation.from_rotvec([0., 0., index * .02]).as_matrix(),
                 "t": np.array([index * .002, 0., .5]), "source_tag_ids": [0]}
                for index in range(count)]

    def test_short_gap_uses_two_sided_pose_interpolation_and_keeps_endpoints(self):
        rows = self.sequence()
        expected = matrix(rows[3]["R"], rows[3]["t"])
        rows[2] = {"frame": 2, "time_s": 2 / 30, "status": "lost"}
        rows[3] = {"frame": 3, "time_s": 3 / 30, "status": "lost"}
        before = rows[1]["t"].copy()
        result = fill_short_pose_gaps(rows)
        self.assertEqual(result[3]["status"], "interpolated")
        np.testing.assert_allclose(matrix(result[3]["R"], result[3]["t"]), expected, atol=1e-12)
        np.testing.assert_array_equal(result[1]["t"], before)
        self.assertEqual(rows[3]["status"], "lost")

    def test_long_leading_trailing_and_implausible_gaps_remain_lost(self):
        rows = self.sequence()
        for index in [0, 3, 4, 5, 6, 11]:
            rows[index] = {"frame": index, "time_s": index / 30, "status": "lost"}
        result = fill_short_pose_gaps(rows)
        self.assertTrue(all(result[index]["status"] == "lost" for index in [0, 3, 4, 5, 6, 11]))
        rows = self.sequence()
        rows[2] = {"frame": 2, "time_s": 2 / 30, "status": "lost"}
        rows[3]["t"] = np.array([2., 0., .5])
        self.assertEqual(fill_short_pose_gaps(rows)[2]["status"], "lost")

    def test_only_short_unverified_excursions_between_image_anchors_are_rejected(self):
        rows = self.sequence()
        rows[3]["pose_source"] = "flow_recovered"
        rows[7]["pose_source"] = "redetected"
        for index in (4, 5, 6):
            rows[index].update(pose_source="original_unverified", t=np.array([.7, .5, 0.]))
        rejected = reject_isolated_excursions(rows)
        self.assertTrue(all(rejected[i]["status"] == "lost" for i in (4, 5, 6)))
        completed = fill_short_pose_gaps(rejected)
        self.assertTrue(all(completed[i]["status"] == "interpolated" for i in (4, 5, 6)))
        rows[3]["pose_source"] = "original_unverified"
        self.assertEqual(reject_isolated_excursions(rows)[4]["status"], "measured")

    def test_single_id_spike_selects_a_window_without_rejecting_observations(self):
        rows = self.sequence()
        for row in rows:
            row["source_tag_ids"] = [0, 1]
        rows[5].update(source_tag_ids=[0], t=np.array([.8, .3, 0.]))
        self.assertEqual(recovery_windows(rows), [(4, 6)])
        self.assertEqual(rows[5]["status"], "measured")

    def test_valid_unverified_neighbor_does_not_hide_an_isolated_jump(self):
        rows = self.sequence()
        rows[3]["pose_source"] = "redetected"
        rows[6]["pose_source"] = "flow_recovered"
        rows[4]["pose_source"] = "original_unverified"
        rows[5].update(pose_source="original_unverified", t=np.array([.7, .5, 0.]))
        rejected = reject_isolated_excursions(rows)
        self.assertEqual(rejected[4]["status"], "measured")
        np.testing.assert_array_equal(rejected[4]["t"], rows[4]["t"])
        self.assertEqual(rejected[5]["status"], "lost")
        completed = fill_short_pose_gaps(rejected)
        self.assertEqual(completed[5]["status"], "interpolated")
        np.testing.assert_allclose(completed[5]["t"], np.array([.010, 0., .5]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
