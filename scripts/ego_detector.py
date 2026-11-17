"""Explicit CPU / CUDA detection, with identical corner and confidence fields.

CUDA accelerates full-image thresholding, components and quad fitting. Bit
decoding and edge refinement use the upstream CPU decoder. Small rectified
local searches keep the original CPU detector, avoiding GPU buffer thrashing.
"""

import ctypes
from pathlib import Path
import sys
import weakref

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CUDA_LIBRARY = ROOT / "build/ego_cuda/libego_apriltag_cuda.so"
CUDA_CACHE = {"backend": "cuda", "version": 2, "quad_decimate": 2, "input_scale": 2,
              "tile_size": [1088, 728], "tile_overlap_min": 128,
              "gpu_revision": "65366174d5e735bf6d71e34328b2a064e16bffb4",
              "decoder_revision": "67c0f842a1c24fb6a7527eba77cd5c10135cf8e6"}


def cpu_detector(family):
    sys.path.insert(0, str(ROOT / "build"))
    from apriltag import apriltag
    return apriltag(family, threads=2, maxhamming=1, decimate=1., refine_edges=True)


def detection_cache_settings(device):
    # Leave existing CPU version-4 caches valid, but never mix them with CUDA.
    return {"detector": dict(CUDA_CACHE)} if device == "cuda" else {}


def load_cuda_library():
    if not CUDA_LIBRARY.exists():
        raise RuntimeError("CUDA 检测后端尚未编译。先执行 bash scripts/setup_ego_cuda.sh")
    try:
        lib = ctypes.CDLL(str(CUDA_LIBRARY))
    except OSError as error:
        raise RuntimeError(f"无法加载 CUDA 检测后端：{error}") from error
    lib.ego_cuda_error.restype = ctypes.c_char_p
    lib.ego_cuda_probe.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.ego_cuda_probe.restype = ctypes.c_int
    lib.ego_cuda_create.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.ego_cuda_create.restype = ctypes.c_void_p
    array = np.ctypeslib.ndpointer
    lib.ego_cuda_detect.argtypes = [
        ctypes.c_void_p, array(np.uint8, flags="C_CONTIGUOUS"),
        array(np.float64, flags="C_CONTIGUOUS"), array(np.int32, flags="C_CONTIGUOUS"),
        array(np.int32, flags="C_CONTIGUOUS"), array(np.float32, flags="C_CONTIGUOUS"), ctypes.c_int]
    lib.ego_cuda_detect.restype = ctypes.c_int
    lib.ego_cuda_destroy.argtypes = [ctypes.c_void_p]
    lib.ego_cuda_destroy.restype = None
    return lib


class CudaDetector:
    def __init__(self, family):
        if family != "tag36h11":
            raise ValueError("当前 CUDA 后端仅支持 tag36h11；其他家族请显式使用 --device cpu")
        self.lib = load_cuda_library()
        name = ctypes.create_string_buffer(256)
        if self.lib.ego_cuda_probe(name, len(name)):
            raise RuntimeError(f"CUDA 不可用：{self.error()}。请检查显卡驱动及运行环境的 GPU 访问权限。")
        self.gpu_name = name.value.decode()
        self.local = cpu_detector(family)
        self.shape = None
        self.handle = None
        self._release = None
        self.corners = np.empty((4096, 4, 2), dtype=np.float64)
        self.ids = np.empty(4096, dtype=np.int32)
        self.hamming = np.empty(4096, dtype=np.int32)
        self.margins = np.empty(4096, dtype=np.float32)

    def error(self):
        return self.lib.ego_cuda_error().decode(errors="replace")

    def detect_local(self, gray):
        return self.local.detect(gray)

    def _detect_native(self, gray):
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("CUDA detector requires a uint8 grayscale image")
        height, width = gray.shape
        bottom, right = (-height) % 8, (-width) % 8
        if bottom or right:
            gray = cv2.copyMakeBorder(gray, 0, bottom, 0, right, cv2.BORDER_REPLICATE)
        gray = np.ascontiguousarray(gray)
        if self.shape != gray.shape:
            if self._release is not None:
                self._release()
            self.handle = self.lib.ego_cuda_create(gray.shape[1], gray.shape[0])
            self.shape = None
            if not self.handle:
                raise RuntimeError(f"CUDA 初始化失败：{self.error()}")
            self._release = weakref.finalize(self, self.lib.ego_cuda_destroy, self.handle)
            self.shape = gray.shape
        count = self.lib.ego_cuda_detect(self.handle, gray, self.corners, self.ids,
                                         self.hamming, self.margins, len(self.ids))
        if count < 0:
            raise RuntimeError(f"CUDA 检测失败：{self.error()}")
        found = []
        for index in range(count):
            corners = self.corners[index].copy()
            if not np.all(np.isfinite(corners)) or not np.isfinite(self.margins[index]):
                continue
            center = corners.mean(axis=0)
            if not (0 <= center[0] < width and 0 <= center[1] < height):
                continue
            # Both decoders use canonical lb-rb-rt-lt, including rotated tags.
            # Preserve every instance of an ID for the rigid-layout selector.
            found.append({"id": int(self.ids[index]), "hamming": int(self.hamming[index]),
                          "margin": float(self.margins[index]), "lb-rb-rt-lt": corners,
                          "center": center})
        return found

    def detect(self, gray):
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("CUDA detector requires a uint8 grayscale image")
        # The upstream GPU finder decimates by two. Double each overlapping
        # tile first, so small tags retain the original image resolution during
        # quad search. Fixed tile sizes keep GPU buffers reusable and <4M pixels.
        height, width = gray.shape
        tile_width, tile_height = min(width, 1088), min(height, 728)
        xs = np.unique(np.linspace(0, width - tile_width, max(1, int(np.ceil(width / 960))), dtype=int))
        ys = np.unique(np.linspace(0, height - tile_height, max(1, int(np.ceil(height / 600))), dtype=int))
        found = []
        for y in ys:
            for x in xs:
                tile = cv2.resize(gray[y:y + tile_height, x:x + tile_width], None,
                                  fx=2., fy=2., interpolation=cv2.INTER_NEAREST)
                for row in self._detect_native(tile):
                    row["lb-rb-rt-lt"] = row["lb-rb-rt-lt"] / 2. + [x, y]
                    row["center"] = row["lb-rb-rt-lt"].mean(axis=0)
                    duplicates = [i for i, previous in enumerate(found) if previous["id"] == row["id"]
                                  and np.sqrt(np.mean(np.sum((previous["lb-rb-rt-lt"] - row["lb-rb-rt-lt"]) ** 2,
                                                             axis=1))) < 2.]
                    if not duplicates:
                        found.append(row)
                    elif row["margin"] > found[duplicates[0]]["margin"]:
                        found[duplicates[0]] = row
        return found


def create_detector(family, device="cpu"):
    if device == "cpu":
        return cpu_detector(family)
    if device == "cuda":
        return CudaDetector(family)
    raise ValueError(f"Unknown detector device: {device}")
