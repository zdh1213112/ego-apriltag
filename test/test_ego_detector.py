"""Backend selection, cache isolation and optional real CUDA geometry checks."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ego_detector import CudaDetector, create_detector, detection_cache_settings, load_cuda_library


class DetectorTests(unittest.TestCase):
    def test_cuda_is_explicit_and_does_not_fall_back_silently(self):
        with patch("ego_detector.load_cuda_library", side_effect=RuntimeError("CUDA unavailable")), \
                patch("ego_detector.cpu_detector") as cpu:
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                create_detector("tag36h11", "cuda")
            cpu.assert_not_called()

    def test_gpu_cache_cannot_be_mistaken_for_cpu_cache(self):
        self.assertEqual(detection_cache_settings("cpu"), {})
        self.assertEqual(detection_cache_settings("cuda")["detector"]["input_scale"], 2)
        self.assertNotEqual(detection_cache_settings("cuda"), detection_cache_settings("cpu"))

    def test_missing_library_explains_setup(self):
        with patch("ego_detector.CUDA_LIBRARY", Path("/tmp/nonexistent-ego-backend-test.so")):
            with self.assertRaisesRegex(RuntimeError, "setup_ego_cuda.sh"):
                load_cuda_library()

    def test_unsupported_family_rejected_before_accessing_gpu(self):
        with self.assertRaisesRegex(ValueError, "tag36h11"):
            create_detector("tag25h9", "cuda")

    def test_overlapping_tiles_preserve_distinct_instances_of_same_id(self):
        # Two sightings of one tag in overlapping tiles, plus a different tag
        # with the same ID. Only the repeated geometric instance is deduplicated.
        detector = CudaDetector.__new__(CudaDetector)
        corners = np.array([[850., 320.], [890., 320.], [890., 280.], [850., 280.]])

        def row(points, margin):
            return {"id": 2, "margin": margin, "hamming": 0, "lb-rb-rt-lt": points.copy()}

        tiles = [[row(corners * 2, 40)],
                 [row((corners - [832, 0]) * 2, 50),
                  row((corners - [832, 0] + [200, 0]) * 2, 45)]]
        with patch.object(detector, "_detect_native", side_effect=tiles):
            found = detector.detect(np.full((600, 1920), 255, np.uint8))
        self.assertEqual(len(found), 2)
        self.assertEqual([item["id"] for item in found], [2, 2])
        self.assertEqual(found[0]["margin"], 50)
        np.testing.assert_allclose(found[0]["lb-rb-rt-lt"], corners)


@unittest.skipUnless(os.environ.get("EGO_TEST_CUDA") == "1", "Requires a visible NVIDIA GPU and local CUDA build")
class CudaIntegrationTests(unittest.TestCase):
    def test_rotations_duplicate_ids_and_canonical_corners(self):
        root = Path(__file__).resolve().parents[1]
        image = np.full((600, 960), 255, dtype=np.uint8)
        for tag_id, turns, x, y in ((0, 0, 50, 30), (2, 1, 420, 20), (2, 3, 620, 310), (1, 2, 100, 350)):
            tag = cv2.imread(str(root / f"tag_family_examples/apriltag/tag36h11/id_{tag_id:04d}.png"), 0)
            self.assertIsNotNone(tag)
            tag = np.rot90(cv2.resize(tag, (160, 160), interpolation=cv2.INTER_NEAREST), turns)
            image[y:y + 160, x:x + 160] = tag
        cpu = create_detector("tag36h11", "cpu").detect(image)
        cuda = create_detector("tag36h11", "cuda").detect(image)
        for found in (cpu, cuda):
            self.assertEqual(sorted(row["id"] for row in found), [0, 1, 2, 2])
        for row in cpu:
            candidates = [other for other in cuda if other["id"] == row["id"]]
            match = min(candidates, key=lambda other: np.linalg.norm(other["center"] - row["center"]))
            np.testing.assert_allclose(row["lb-rb-rt-lt"], match["lb-rb-rt-lt"], atol=1.)


if __name__ == "__main__":
    unittest.main()
