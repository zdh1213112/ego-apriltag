"""Regression test for segmented EGOViewer MCAP with per-picture timestamp SEI."""

from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from track_ego_apriltag import STREAMS, load_recording


@unittest.skipUnless(find_spec("mcap_protobuf"), "MCAP input requires mcap-protobuf-support")
class McapTests(unittest.TestCase):
    def test_segmented_video_keeps_camera_mapping_acquisition_times_and_seek_order(self):
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2
        from mcap_protobuf.writer import Writer

        # Use the same fields as the recorder without requiring generated Foxglove bindings.
        descriptor = descriptor_pb2.FileDescriptorProto(
            name="video.proto", package="foxglove", syntax="proto3",
            dependency=["google/protobuf/timestamp.proto"])
        fields = descriptor.message_type.add(name="CompressedVideo").field
        field = descriptor_pb2.FieldDescriptorProto
        fields.add(name="timestamp", number=1, type=field.TYPE_MESSAGE, type_name=".google.protobuf.Timestamp")
        fields.add(name="data", number=3, type=field.TYPE_BYTES)
        fields.add(name="format", number=4, type=field.TYPE_STRING)
        pool = descriptor_pool.DescriptorPool()
        pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
        video = message_factory.GetMessageClass(pool.Add(descriptor).message_types_by_name["CompressedVideo"])

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            streams = {f"color_{stream}": {"fps": 30} for stream in STREAMS.values()}
            files = {"recording": "session.mcap", "calibration_camera": "calibration.yaml"}
            (root / "meta_session.json").write_text(json.dumps({"files": files, "streams": streams}))
            cameras = [
                {"id": camera, "distortion_model": "KB", "image_width": 32, "image_height": 32,
                 "intrinsics": {"fx": 16, "fy": 16, "cx": 16, "cy": 16},
                 "distortion": {"k1": 0, "k2": 0, "k3": 0, "k4": 0},
                 "extrinsics": {"rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}}
                for camera in STREAMS]
            (root / files["calibration_camera"]).write_text(json.dumps({"cameras": cameras}))
            for index, color in enumerate(("black", "white"), 1):
                # Each part starts at a keyframe and ends in a non-picture SEI.
                packet = subprocess.check_output([
                    "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color={color}:s=32x32:r=30",
                    "-frames:v", "1", "-c:v", "libx264", "-bf", "0", "-threads", "1", "-f", "h264", "-"])
                part = root / f"part{index:04d}"
                part.mkdir()
                with Writer(str(part / f"session_{part.name}.mcap")) as writer:
                    for offset, stream in enumerate(STREAMS.values()):
                        timestamp = 1_000_000 + index * 34_567 + offset
                        sei = b"EGO-4B0000000_RS,timestamp_us=" + str(timestamp).encode()
                        sei_nal = b"\x00\x00\x00\x01\x06\x05" + bytes([len(sei)]) + sei + b"\x80"
                        message = video(format="h264", data=packet + sei_nal)
                        message.timestamp.FromMicroseconds(timestamp)
                        writer.write_message(f"/camera/{stream}", message, log_time=1, publish_time=2)

            prepared, _, times = load_recording(root)
            self.addCleanup(prepared.cleanup)
            for offset, (camera, stream) in enumerate(STREAMS.items()):
                np.testing.assert_array_equal(times[camera], [1_034_567 + offset, 1_069_134 + offset])
                capture = cv2.VideoCapture(prepared["files"][f"video_{stream}"])
                try:
                    self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 2)
                    for index, expected in ((1, 255), (0, 0)):
                        self.assertTrue(capture.set(cv2.CAP_PROP_POS_FRAMES, index))
                        ok, image = capture.read()
                        self.assertTrue(ok)
                        self.assertAlmostEqual(float(image.mean()), expected, delta=3)
                finally:
                    capture.release()
