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
    Camera, STREAMS, TagTrackingState, aggregate_tag_status, associate_frames, collect_tag_observations,
    estimate_tag_alignment, export_global_results, export_results, export_tag_results,
    fit_pose, fuse_tag_results, load_recording, main, matrix, object_points, parse_args,
    solve_sequence, track_tag, triangulate,
)


class RecordingTests(unittest.TestCase):
    def test_invalid_recording_directory_has_an_actionable_error(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "does not exist or is not a directory"):
                load_recording(root / "missing")
            with self.assertRaisesRegex(ValueError, "original EGOViewer recording directory"):
                load_recording(root)
            (root / "meta_first.json").write_text("{}")
            (root / "meta_second.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Multiple meta_\\*.json"):
                load_recording(root)

    def test_invalid_input_fails_before_creating_output_directories(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results"
            with self.assertRaisesRegex(SystemExit, "No meta_\\*.json"):
                main([str(root), "--tagsize", "0.03", "--tag-ids", "0", "1", "2",
                      "--output", str(output)])
            self.assertFalse(output.exists())

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


class GlobalTrackingTests(unittest.TestCase):
    def rigid_sequences(self):
        offsets = {
            0: np.eye(4),
            1: matrix(Rotation.from_euler("z", 90, degrees=True).as_matrix(), np.array([.04, 0., .02])),
            2: matrix(Rotation.from_euler("y", -60, degrees=True).as_matrix(), np.array([.08, .01, .04])),
        }
        visible = {0: set(range(4)), 1: set(range(9)), 2: set(range(4, 8)) | {9, 10, 11}}
        tracks = {tag_id: [] for tag_id in offsets}
        expected = []
        for index in range(13):
            pose = matrix(Rotation.from_rotvec(np.array([.02, -.01, .03]) * index).as_matrix(),
                          np.array([.1 + .005 * index, .2, .4 + .002 * index]))
            expected.append(pose)
            for tag_id, offset in offsets.items():
                row = {"frame": index, "timestamp_us": 1000000 + index * 33333,
                       "time_s": index * .033333, "status": "lost"}
                if index in visible[tag_id]:
                    observed = pose @ offset
                    views = 4 if tag_id == 0 else 2
                    row.update(status="flow_only" if index == 11 else "measured",
                               R=observed[:3, :3], t=observed[:3, 3], n_views=views,
                               rms=.4 if tag_id == 0 else .7,
                               cameras=[f"cam_{i}" for i in range(views)])
                tracks[tag_id].append(row)
        return tracks, np.array(expected), offsets

    def test_alignment_can_connect_tags_without_direct_overlap(self):
        tracks, _, offsets = self.rigid_sequences()
        alignment = estimate_tag_alignment(tracks)
        self.assertEqual(alignment["paths"]["2"], [0, 1, 2])
        for tag_id, expected in offsets.items():
            np.testing.assert_allclose(alignment["T_reference_tag"][str(tag_id)], expected, atol=1e-12)

    def test_switching_visible_id_preserves_one_reference_pose(self):
        tracks, expected, _ = self.rigid_sequences()
        results, _ = fuse_tag_results(tracks)
        for index in range(12):
            np.testing.assert_allclose(matrix(results[index]["R"], results[index]["t"]), expected[index], atol=1e-12)
        self.assertEqual(results[8]["source_tag_ids"], [1])
        self.assertEqual(results[10]["source_tag_ids"], [2])
        self.assertEqual(results[11]["status"], "flow_only")
        self.assertEqual(results[12]["status"], "lost")
        self.assertNotIn("t", results[12])

    def test_alignment_and_fusion_reject_a_large_pose_outlier(self):
        tracks, expected, offsets = self.rigid_sequences()
        tracks[1][0]["t"] = tracks[1][0]["t"] + [.2, 0., 0.]
        results, alignment = fuse_tag_results(tracks)
        np.testing.assert_allclose(alignment["T_reference_tag"]["1"], offsets[1], atol=1e-12)
        self.assertEqual(results[0]["source_tag_ids"], [0])
        np.testing.assert_allclose(results[0]["t"], expected[0, :3, 3], atol=1e-12)

    def test_disconnected_visible_tags_cannot_be_stitched_without_alignment(self):
        tracks, _, _ = self.rigid_sequences()
        for row in tracks[1][:4] + tracks[2]:
            row["status"] = "lost"
        with self.assertRaisesRegex(ValueError, "Cannot align tag IDs"):
            fuse_tag_results(tracks)

    def test_global_pose_export_is_one_pose_per_frame_with_nan_for_loss(self):
        tracks, expected, _ = self.rigid_sequences()
        results, alignment = fuse_tag_results(tracks)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_global_results(root, results, alignment, list(tracks))
            with np.load(root / "global_poses.npz") as data:
                self.assertEqual(data["T_cam0_global_raw"].shape, (13, 4, 4))
                np.testing.assert_allclose(data["T_cam0_global_raw"][:12], expected[:12], atol=1e-12)
                self.assertTrue(np.isnan(data["T_cam0_global_raw"][12]).all())
                np.testing.assert_array_equal(data["source_mask"][10], [False, False, True])
                self.assertEqual(int(data["reference_tag_id"]), 0)
            with (root / "global_poses.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 13)
            self.assertEqual(rows[10]["source_tag_ids"], "2")
            self.assertAlmostEqual(float(rows[10]["raw_t_m_x"]), expected[10, 0, 3])
            self.assertEqual(rows[12]["status"], "lost")

    def test_global_is_lost_only_when_all_tags_are_lost(self):
        status = np.array([
            ["lost", "lost", "lost"],
            ["measured", "lost", "lost"],
            ["lost", "measured", "lost"],
            ["lost", "lost", "flow_only"],
            ["flow_only", "measured", "flow_only"],
        ])
        overall, counts = aggregate_tag_status(status)
        np.testing.assert_array_equal(overall, ["lost", "measured", "measured", "flow_only", "measured"])
        np.testing.assert_array_equal(counts, [0, 1, 1, 1, 3])

    def test_invalid_status_cannot_count_as_stable_tracking(self):
        for status in (np.array([["detected"]]), np.empty((2, 0)), np.array(["measured"])):
            with self.subTest(status=status), self.assertRaises(ValueError):
                aggregate_tag_status(status)


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
            self.assertEqual(summary["valid_frames"], 2)
            self.assertEqual(summary["lost_frames"], 0)
            self.assertEqual(summary["coverage_percent"], 100.)
            with np.load(root / "poses.npz") as payload:
                np.testing.assert_array_equal(payload["tag_ids"], [1, 0])
                np.testing.assert_array_equal(payload["timestamp_us"], times["cam_0"])
                np.testing.assert_array_equal(payload["status"], [["lost", "measured"], ["measured", "lost"]])
                np.testing.assert_array_equal(payload["global_status"], ["measured", "measured"])
                np.testing.assert_array_equal(payload["n_valid_tags"], [1, 1])
                self.assertEqual(payload["T_cam0_tag_raw"].shape, (2, 2, 4, 4))
                np.testing.assert_allclose(payload["T_cam0_tag_raw"][1, 0, :3, 3], positions[1], atol=1e-6)
                np.testing.assert_allclose(payload["T_cam0_tag_raw"][0, 1, :3, 3], positions[0], atol=1e-6)
                self.assertTrue(np.isnan(payload["T_cam0_tag_raw"][0, 0]).all())
            with (root / "poses.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(int(row["frame"]), int(row["tag_id"])) for row in rows],
                             [(0, 1), (0, 0), (1, 1), (1, 0)])
            self.assertTrue(all(row["global_status"] == "measured" for row in rows))
            self.assertEqual(rows[0]["status"], "lost")
            with (root / "global_tracking.csv").open() as stream:
                overall_rows = list(csv.DictReader(stream))
            self.assertEqual([row["active_tag_ids"] for row in overall_rows], ["0", "1"])
            self.assertTrue(all(row["status"] == "measured" for row in overall_rows))

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
