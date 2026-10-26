"""Numerical checks for calibrated EGO tracking; run with python3 -m unittest."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from track_ego_apriltag import Camera, associate_frames, fit_pose, object_points, solve_sequence, triangulate


class GeometryTests(unittest.TestCase):
    def setUp(self):
        K = np.array([[505., 0, 940.], [0, 506., 590.], [0, 0, 1.]])
        D = np.array([.08, -.02, .006, -.0008])
        rotations = [np.eye(3), Rotation.from_rotvec([0, -.005, 0]).as_matrix(),
                     Rotation.from_euler("y", -55, degrees=True).as_matrix(),
                     Rotation.from_euler("y", 55, degrees=True).as_matrix()]
        centers = [[0, 0, 0], [.10, 0, 0], [.14, .01, -.02], [-.04, .01, -.02]]
        self.cameras = {f"cam_{i}": Camera(f"cam_{i}", K, D, R, -R @ center)
                        for i, (R, center) in enumerate(zip(rotations, map(np.array, centers)))}
        self.R = Rotation.from_rotvec([.1, -.2, .05]).as_matrix()
        self.t = np.array([.20, -.10, .40])
        self.size = .04

    def observations(self):
        # Independent OpenCV forward projection produces the synthetic observations.
        points = object_points(self.size) @ self.R.T + self.t
        output = []
        for camera in self.cameras.values():
            local = points @ camera.R.T + camera.t
            self.assertTrue(np.all(local[:, 2] > 0))
            pixels = cv2.fisheye.projectPoints(local.reshape(-1, 1, 3),
                                               np.zeros(3), np.zeros(3), camera.K, camera.D)[0]
            output.append({"camera": camera.id, "corners": pixels.reshape(4, 2), "source": "decoded"})
        return output

    def test_kb_matches_opencv_and_inverts_peripheral_rays(self):
        for observation in self.observations():
            camera = self.cameras[observation["camera"]]
            points = (object_points(self.size) @ self.R.T + self.t) @ camera.R.T + camera.t
            np.testing.assert_allclose(camera.project(points), observation["corners"], atol=1e-9)
            directions = points / np.linalg.norm(points, axis=1)[:, None]
            np.testing.assert_allclose(camera.unproject(observation["corners"]), directions, atol=1e-10)
        # Beyond the pinhole hemisphere, the KB bearing still has an unambiguous direction.
        theta = np.deg2rad([0., 80., 89., 95.])
        rays = np.column_stack((np.sin(theta), np.zeros(4), np.cos(theta)))
        camera = self.cameras["cam_0"]
        np.testing.assert_allclose(camera.unproject(camera.project(rays)), rays, atol=1e-10)

    def test_triangulation_recovers_metric_tag(self):
        points = triangulate(self.observations(), self.cameras)
        expected = object_points(self.size) @ self.R.T + self.t
        np.testing.assert_allclose(points, expected, atol=1e-9)

    def test_joint_pose_rejects_bad_camera(self):
        observations = self.observations()
        observations[-1]["corners"] = observations[-1]["corners"] + [60., -45.]
        pose = fit_pose(observations, self.cameras, self.size)
        self.assertIsNotNone(pose)
        self.assertEqual(len(pose["observations"]), 3)
        np.testing.assert_allclose(pose["t"], self.t, atol=1e-6)
        self.assertLess(Rotation.from_matrix(pose["R"].T @ self.R).magnitude(), 1e-5)

    def test_timestamp_association_uses_time_instead_of_frame_number(self):
        times = {"cam_0": np.array([10000, 43333]), "cam_1": np.array([500, 10008, 43339])}
        observations = {key: [{"camera": key, "frame": i} for i in range(len(t))]
                        for key, t in times.items()}
        matched = associate_frames(observations, times, 100, 2)
        self.assertEqual(matched[0][1]["frame"], 1)
        self.assertEqual(matched[1][1]["frame"], 2)
        self.assertIsNone(fit_pose([], self.cameras, self.size))

    def test_flow_only_requires_two_views_and_long_dropout_is_lost(self):
        observations = self.observations()
        for observation in observations:
            observation["source"] = "flow"
        frames = [[observations[0]], observations[:2], []]
        times = {"cam_0": np.array([10000, 43333, 76666])}
        args = SimpleNamespace(tagsize=self.size, cutoff=4., max_reprojection=2.5)
        results = solve_sequence(frames, times, self.cameras, args)
        self.assertEqual([r["status"] for r in results], ["lost", "flow_only", "lost"])
        self.assertNotIn("t", results[-1])


if __name__ == "__main__":
    unittest.main()
