"""Numerical checks for calibrated EGO tracking; run with python3 -m unittest."""

from contextlib import redirect_stderr
import csv
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from track_ego_apriltag import (
    Camera, STREAMS, TagTrackingState, associate_frames, collect_tag_observations,
    export_results, export_tag_results, fit_pose, load_recording, object_points,
    parse_args, solve_sequence, track_tag, triangulate,
)


class RecordingTests(unittest.TestCase):
    def test_single_and_multi_tag_arguments(self):
        base = ["recording", "--tagsize", "0.04"]
        self.assertEqual(parse_args(base).tag_id, 0)
        self.assertEqual(parse_args(base + ["--tag-id", "2"]).tag_id, 2)
        self.assertEqual(parse_args(base + ["--tag-ids", "0", "1", "2"]).tag_ids, [0, 1, 2])
        for options in (["--tag-ids", "0", "0"], ["--tag-ids", "-1"],
                        ["--tag-id", "0", "--tag-ids", "1"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + options)

    def test_load_recording_accepts_single_and_multi_column_timestamps(self):
        expected = [1793091404379147, 1793091404412482]
        formats = {
            "single_column": "timestamp_us\n1793091404379147\n1793091404412482\n",
            "multi_column": ("timestamp_us,timestamp_gap,sequence\n"
                             "1793091404379147,,781\n1793091404412482,33335,782\n"),
            "single_frame": "timestamp_us,timestamp_gap,sequence\n1793091404379147,,781\n",
        }
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            calibration = {"cameras": [
                {"id": camera_id, "distortion_model": "KB",
                 "intrinsics": {"fx": 505., "fy": 506., "cx": 940., "cy": 590.},
                 "distortion": {f"k{i}": 0. for i in range(1, 5)},
                 "extrinsics": {"rotation": np.eye(3).tolist(), "translation": [0., 0., 0.]},
                 "image_width": 1920, "image_height": 1200}
                for camera_id in STREAMS
            ]}
            (directory / "calibration.yaml").write_text(json.dumps(calibration))
            files = {"calibration_camera": "calibration.yaml"}
            files.update({f"video_{stream}_pts": f"{stream}.csv" for stream in STREAMS.values()})
            (directory / "meta_test.json").write_text(json.dumps({"files": files}))
            for name, contents in formats.items():
                with self.subTest(format=name):
                    for stream in STREAMS.values():
                        (directory / f"{stream}.csv").write_text(contents)
                    _, cameras, timestamps = load_recording(directory)
                    self.assertEqual(set(cameras), set(STREAMS))
                    self.assertEqual(set(timestamps), set(STREAMS))
                    for values in timestamps.values():
                        self.assertEqual(values.dtype, np.dtype("int64"))
                        np.testing.assert_array_equal(values, expected[:1] if name == "single_frame" else expected)


class MultiTagDetectionTests(unittest.TestCase):
    def detection(self, tag_id):
        corners = np.array([[4., 12.], [12., 12.], [12., 4.], [4., 4.]]) + [tag_id * 12, 0]
        return {"id": tag_id, "margin": 100., "hamming": 0, "lb-rb-rt-lt": corners}

    def test_shared_detection_and_cache_keep_ids_and_dropouts_separate(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        captures = [Mock() for _ in STREAMS]
        for capture in captures:
            capture.read.side_effect = [(True, image), (True, image)]
        detector = Mock()
        detector.detect.side_effect = [[self.detection(0), self.detection(1)], [self.detection(1)]] * 4
        metadata = {"files": {f"video_{stream}": f"{stream}.mp4" for stream in STREAMS.values()}}
        cameras = {key: SimpleNamespace(width=32, height=24) for key in STREAMS}
        timestamps = {key: np.array([1000, 34333]) for key in STREAMS}
        args = SimpleNamespace(family="tag36h11", max_frames=0, flow_max_frames=0, reuse_detections=False)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {tag_id: root / f"tag_{tag_id}" for tag_id in (0, 1, 2)}
            for output in outputs.values():
                output.mkdir()
            with patch("apriltag.apriltag", return_value=detector), \
                    patch("track_ego_apriltag.cv2.VideoCapture", side_effect=captures) as open_video, \
                    patch("track_ego_apriltag.local_detection", return_value=None):
                observations = collect_tag_observations(root, outputs, metadata, cameras, timestamps, args)
                self.assertEqual(detector.detect.call_count, 8)
                self.assertEqual(open_video.call_count, 4)
                for camera_id in STREAMS:
                    self.assertEqual(observations[0][camera_id][0]["tag_id"], 0)
                    self.assertIsNone(observations[0][camera_id][1])
                    self.assertEqual([o["tag_id"] for o in observations[1][camera_id]], [1, 1])
                    self.assertEqual(observations[2][camera_id], [None, None])
                args.reuse_detections = True
                detector.detect.reset_mock()
                open_video.reset_mock()
                reused = collect_tag_observations(root, outputs, metadata, cameras, timestamps, args)
                self.assertEqual(reused, observations)
                detector.detect.assert_not_called()
                open_video.assert_not_called()

    def test_flow_cannot_take_the_identity_of_another_decoded_tag(self):
        other = self.detection(1)
        state = TagTrackingState(self.detection(0)["lb-rb-rt-lt"], self.detection(0)["lb-rb-rt-lt"], 0)
        gray = np.zeros((24, 32), dtype=np.uint8)
        with patch("track_ego_apriltag.local_detection", return_value=None), \
                patch("track_ego_apriltag.flow_corners", return_value=other["lb-rb-rt-lt"]):
            observation = track_tag(gray, gray, None, [other], None, 0, state, 1, 15)
        self.assertIsNone(observation)
        self.assertIsNone(state.previous_corners)
        independent = track_tag(gray, gray, None, [other], None, 1, TagTrackingState(), 1, 15)
        self.assertEqual(independent["tag_id"], 1)
        self.assertEqual(independent["source"], "decoded")


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

    def observations(self, translation=None):
        # Independent OpenCV forward projection produces the synthetic observations.
        points = object_points(self.size) @ self.R.T + (self.t if translation is None else translation)
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

    def test_pose_fit_rejects_mixed_tag_ids(self):
        observations = self.observations()
        observations[0]["tag_id"] = 0
        observations[1]["tag_id"] = 1
        with self.assertRaisesRegex(ValueError, "only one tag ID"):
            fit_pose(observations, self.cameras, self.size)

    def test_multi_tag_exports_keep_identity_time_and_lost_poses(self):
        times = {key: np.array([10000, 43333]) for key in self.cameras}
        positions = {0: self.t, 1: self.t + [.02, .01, .03]}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {tag_id: root / f"tag_{tag_id}" for tag_id in (1, 0)}
            summaries = {}
            for tag_id, output in outputs.items():
                output.mkdir()
                observations = self.observations(positions[tag_id])
                for observation in observations:
                    observation["tag_id"] = tag_id
                frames = [[], []]
                frames[tag_id] = observations
                args = SimpleNamespace(tagsize=self.size, cutoff=4., max_reprojection=2.5,
                                       recording=root, family="tag36h11", tag_id=tag_id)
                results = solve_sequence(frames, times, self.cameras, args)
                by_camera = {key: [None, None] for key in self.cameras}
                for observation in observations:
                    by_camera[observation["camera"]][tag_id] = observation
                summaries[tag_id] = export_results(output, results, self.cameras, times, by_camera, args)
            summary = export_tag_results(root, outputs, summaries)
            self.assertEqual(summary["frames_any_tag_valid"], 2)
            self.assertEqual(summary["frames_all_tags_valid"], 0)
            with np.load(root / "poses.npz") as payload:
                np.testing.assert_array_equal(payload["tag_ids"], [1, 0])
                np.testing.assert_array_equal(payload["timestamp_us"], times["cam_0"])
                np.testing.assert_array_equal(payload["status"], [["lost", "measured"], ["measured", "lost"]])
                self.assertEqual(payload["T_cam0_tag_raw"].shape, (2, 2, 4, 4))
                np.testing.assert_allclose(payload["T_cam0_tag_raw"][1, 0, :3, 3], positions[1], atol=1e-6)
                np.testing.assert_allclose(payload["T_cam0_tag_raw"][0, 1, :3, 3], positions[0], atol=1e-6)
                self.assertTrue(np.isnan(payload["T_cam0_tag_raw"][0, 0]).all())
            with (root / "poses.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(int(row["frame"]), int(row["tag_id"])) for row in rows],
                             [(0, 1), (0, 0), (1, 1), (1, 0)])

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
