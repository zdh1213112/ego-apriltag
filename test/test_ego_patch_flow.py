"""Regression checks for localized patch tracking during offline recovery."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ego_patch_flow import affine_corners, collect_tracks, gap_anchors, supported_patch_fit, track_patch


class PatchFlowTests(unittest.TestCase):
    def test_moving_patch_against_stationary_background(self):
        rng = np.random.default_rng(11)
        tile = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        previous = rng.integers(15, 60, (180, 240), dtype=np.uint8)
        current = previous.copy()
        previous[60:92, 70:102] = tile
        current[83:115, 115:147] = cv2.GaussianBlur(tile, (5, 5), 1)
        corners = np.array([[70, 91], [101, 91], [101, 60], [70, 60]], dtype=float)
        found = track_patch(previous, current, corners)
        self.assertIsNotNone(found)
        np.testing.assert_allclose(found, corners + [45, 23], atol=1)
        self.assertIsNone(track_patch(previous, np.zeros_like(current), corners))

    def test_affine_patch_round_trip(self):
        patch = cv2.GaussianBlur(np.random.default_rng(2).integers(0, 256, (72, 72), dtype=np.uint8), (5, 5), 1)
        warp = cv2.getRotationMatrix2D((36, 36), 4, 1.03)
        moved = cv2.warpAffine(patch, warp, (72, 72), borderMode=cv2.BORDER_REFLECT)
        corners = np.array([[16, 16], [56, 16], [56, 56], [16, 56]], dtype=float)
        found = affine_corners(patch, moved, corners)
        self.assertIsNotNone(found)
        np.testing.assert_allclose(found, corners @ warp[:, :2].T + warp[:, 2], atol=.35)

    def test_flow_does_not_reseed_itself_or_bridge_long_gaps(self):
        rows = [dict(status="lost", time_s=i/30) for i in range(42)]
        for i in (0, 41):
            rows[i].update(status="measured", n_views=3, R=np.eye(3), t=np.array([0., 0., .5]))
        rows[20].update(status="flow_only", n_views=4)
        runs, _ = gap_anchors(rows)
        self.assertEqual(runs, {(0, 1): set(range(1, 16)), (41, -1): set(range(26, 41))})

    def test_missing_camera_frame_interrupts_track(self):
        rows = [dict(timestamp_us=t) for t in (100, 200, 300)]
        capture = Mock()
        hits = {}
        collect_tracks(capture, Mock(), np.array([100, 300]), rows, {(0, 1): {2}, (2, -1): {0}},
                       {}, Mock(), 2.5, 10, hits)
        capture.read.assert_not_called()
        self.assertFalse(hits)

    def test_long_one_sided_tracks_require_independent_reverse_confirmation(self):
        rows = [dict(time_s=i / 30) for i in range(21)]
        forward = [dict(camera=f"cam_{i}", tag_id=0, corners=np.zeros((4, 2)),
                        source="forward_flow", anchor=0) for i in range(4)]
        self.assertTrue(supported_patch_fit(3, rows, forward, forward, 2.5))
        self.assertFalse(supported_patch_fit(10, rows, forward, forward, 2.5))
        reverse = [dict(r, anchor=20, source="backward_flow") for r in forward[:2]]
        self.assertTrue(supported_patch_fit(10, rows, forward, forward + reverse, 2.5))
        reverse[0]["corners"] = np.ones((4, 2)) * 4
        self.assertFalse(supported_patch_fit(10, rows, forward, forward + reverse, 2.5))
