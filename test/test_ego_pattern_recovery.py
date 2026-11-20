"""Checks for ID, corner convention and independent views in raw-pattern recovery."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ego_detector import cpu_detector
from ego_pattern_recovery import PatternDecoder, collect_patterns, pose_guides, recover_pattern_gaps


@unittest.skipUnless(hasattr(cv2, "aruco"), "OpenCV aruco is needed for raw-pattern decoding")
class PatternDecoderTests(unittest.TestCase):
    def test_decode_all_rotations_in_native_corner_order(self):
        decoder = PatternDecoder("tag36h11", [1, 2])
        native = cpu_detector("tag36h11")
        canvas = np.pad(decoder.templates[decoder.labels.index((2, 0))], 25, constant_values=255)
        for angle in range(4):
            gray = cv2.GaussianBlur(np.ascontiguousarray(np.rot90(canvas, angle)), (5, 5), 1.2)
            result = decoder.detect(gray, {})
            self.assertEqual([(r["tag_id"], r["hamming"]) for r in result], [(2, 0)])
            original, = native.detect(gray)
            self.assertEqual(original["id"], 2)
            np.testing.assert_allclose(result[0]["corners"], original["lb-rb-rt-lt"], atol=.5)

    def test_known_shape_cannot_supply_the_missing_code(self):
        decoder = PatternDecoder("tag36h11", [1, 2])
        other = PatternDecoder("tag36h11", [98])
        image = np.pad(other.templates[0], 25, constant_values=255)
        border = np.float32([[32.5, 32.5], [96.5, 32.5], [96.5, 96.5], [32.5, 96.5]])
        shapes = {tag: [border] for tag in (1, 2)}
        self.assertEqual(decoder.detect(image, shapes), [])
        self.assertEqual(decoder.detect(np.full_like(image, 80), shapes), [])


class PatternPoseTests(unittest.TestCase):
    def test_search_uses_only_visual_anchors_close_to_a_gap(self):
        rows = [dict(time_s=i / 30, status="lost") for i in range(50)]
        rows[0].update(status="measured", R=np.eye(3), t=np.array([0., 0., .5]))
        rows[2].update(status="flow_only", R=np.eye(3), t=np.array([.02, 0., .5]))
        rows[20].update(status="interpolated")
        guides = pose_guides(rows)
        np.testing.assert_allclose(guides[1][0][:3, 3], [.01, 0, .5])
        self.assertNotIn(21, guides)
        self.assertNotIn(2, guides)

    def test_out_of_sync_camera_is_never_read(self):
        cap = Mock()
        hits = {0: []}
        collect_patterns(cap, Mock(), np.array([5000]), [{"timestamp_us": 1000}],
                         {0: [np.eye(4)]}, Mock(), {1: np.zeros((4, 3))}, 1000, hits)
        cap.read.assert_not_called()
        self.assertEqual(hits[0], [])

    def test_two_tags_from_one_camera_cannot_supply_a_pose(self):
        rows = [dict(timestamp_us=i, time_s=i / 30, status="lost") for i in range(2)]
        rows[0].update(status="measured", R=np.eye(3), t=np.array([0., 0., .5]))
        def one_camera(cap, camera, times, rows, guides, decoder, points, max_sync_us, hits):
            hits[1].extend(dict(tag_id=t, camera="cam_0") for t in (1, 2))
        metadata = {"files": {"video_left": "left.mp4"}}
        with patch("ego_pattern_recovery.STREAMS", {"cam_0": "left"}), \
                patch("ego_pattern_recovery.cv2.VideoCapture"), \
                patch("ego_pattern_recovery.collect_patterns", side_effect=one_camera), \
                patch("ego_pattern_recovery.fit_rigid_detections") as fit:
            report = recover_pattern_gaps(rows, Path("."), metadata, {"cam_0": Mock()},
                                          {"cam_0": np.arange(2)}, {}, .03, {}, Mock(), 2.5, 1000)
        fit.assert_not_called()
        self.assertEqual(report, [])
        self.assertEqual(rows[1]["status"], "lost")
