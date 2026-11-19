"""Read EGOViewer's four H.264 camera topics and their acquisition timestamps."""

from contextlib import ExitStack
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import weakref

import cv2
import numpy as np


class McapMetadata(dict):
    """Keep seekable video files available for the lifetime of the returned metadata."""

    def __init__(self, metadata):
        super().__init__(metadata, files=metadata["files"].copy())
        workspace = TemporaryDirectory(prefix="ego-mcap-")
        self.media_dir = Path(workspace.name)
        self.cleanup = weakref.finalize(self, workspace.cleanup)


def recording_parts(directory, filename):
    declared = directory / filename
    if declared.is_file():
        return [declared]
    # EGOViewer lists a logical filename even when all media is in partNNNN/.
    parts = sorted(directory.glob(f"part[0-9][0-9][0-9][0-9]/{declared.stem}_part*.mcap"))
    if not parts:
        raise ValueError(f"Cannot find {declared} or its partNNNN segments")
    for index, path in enumerate(parts, 1):
        part = f"part{index:04d}"
        if path != directory / part / f"{declared.stem}_{part}.mcap":
            raise ValueError(f"Missing or misnumbered MCAP segment {part} in {directory}")
    return parts


def read_mcap_recording(directory, metadata, streams):
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as error:
        raise ValueError("MCAP recording requires mcap and mcap-protobuf-support "
                         "in the Python environment running the tracker") from error

    parts = recording_parts(directory, metadata["files"]["recording"])
    topics = {f"/camera/{stream}": camera for camera, stream in streams.items()}
    times = {camera: [] for camera in streams}
    prepared = McapMetadata(metadata)
    try:
        with ExitStack() as stack:
            packets = {camera: stack.enter_context((prepared.media_dir / f"{camera}.h264").open("wb"))
                       for camera in streams}
            for part in parts:
                with part.open("rb") as source:
                    reader = make_reader(source, decoder_factories=[DecoderFactory()])
                    for schema, channel, _, frame in reader.iter_decoded_messages(
                            topics=topics, log_time_order=False):
                        if (schema.name != "foxglove.CompressedVideo" or frame.format.lower() != "h264"
                                or not frame.HasField("timestamp") or not frame.data):
                            raise ValueError(f"Expected timestamped H.264 video on {channel.topic} in {part}")
                        camera = topics[channel.topic]
                        timestamp = frame.timestamp.ToMicroseconds()
                        if times[camera] and timestamp <= times[camera][-1]:
                            raise ValueError(f"Non-increasing frame timestamp on {channel.topic} in {part}")
                        times[camera].append(timestamp)
                        packets[camera].write(frame.data)

        for camera, stream in streams.items():
            count = len(times[camera])
            expected = metadata.get("statistics", {}).get(f"video_{stream}_frame_count")
            if not count or (expected is not None and count != expected):
                raise ValueError(f"/camera/{stream}: read {count} frames, metadata reports {expected}")
            raw = prepared.media_dir / f"{camera}.h264"
            video = raw.with_suffix(".mp4")
            fps = metadata["streams"][f"color_{stream}"]["fps"]
            # The MP4 index enables frame-accurate OpenCV seeks during gap recovery.
            # Frame association always uses the original device timestamps above.
            # EGOViewer appends a timestamp SEI after each picture. FFmpeg can
            # otherwise index the final SEI by itself as one extra video frame.
            mux = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-r", str(fps), "-f", "h264",
                                  "-i", str(raw), "-c:v", "copy", "-bsf:v", "filter_units=remove_types=6",
                                  str(video)], stderr=subprocess.PIPE, text=True)
            if mux.returncode:
                raise ValueError(f"Cannot remux /camera/{stream}: {mux.stderr.strip()}")
            raw.unlink()
            capture = cv2.VideoCapture(str(video))
            try:
                if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
                    raise ValueError(f"Remuxed video does not match {count} timestamps for /camera/{stream}")
            finally:
                capture.release()
            prepared["files"][f"video_{stream}"] = str(video)
    except BaseException:
        prepared.cleanup()
        raise
    return prepared, {camera: np.asarray(values, dtype=np.int64) for camera, values in times.items()}
