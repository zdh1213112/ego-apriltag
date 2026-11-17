// Thin local C ABI for the CUDA quad finder and its AprilTag CPU decoder.
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include "apriltag_gpu.h"
extern "C" {
#include "tag36h11.h"
}

namespace {
thread_local std::string last_error;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Detector {
  apriltag_family_t* family = nullptr;
  apriltag_detector_t* cpu = nullptr;
  std::unique_ptr<frc971::apriltag::GpuDetector> gpu;
  std::vector<uint8_t> yuyv;
  int width, height;
  Detector(int w, int h) : yuyv(w * h * 2, 128), width(w), height(h) {
    family = tag36h11_create();
    cpu = apriltag_detector_create();
    apriltag_detector_add_family_bits(cpu, family, 1);
    cpu->quad_decimate = 2;
    cpu->refine_edges = true;
    cpu->nthreads = 2;
    cpu->wp = workerpool_create(2);
    // Detection stays in raw pixels. Our Python KB model handles all poses.
    gpu = std::make_unique<frc971::apriltag::GpuDetector>(
        w, h, cpu, frc971::apriltag::CameraMatrix{1., 0., 1., 0.},
        frc971::apriltag::DistCoeffs{0., 0., 0., 0., 0.});
  }
  ~Detector() {
    gpu.reset();
    apriltag_detector_destroy(cpu);
    tag36h11_destroy(family);
  }
};
}

#define API extern "C" __attribute__((visibility("default")))
API const char* ego_cuda_error() { return last_error.c_str(); }
API int ego_cuda_probe(char* name, int capacity) {
  try {
    int count = 0;
    check(cudaGetDeviceCount(&count));
    if (count == 0) throw std::runtime_error("No CUDA device available");
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0));
    if (capacity > 0) snprintf(name, capacity, "%s", props.name);
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
API void* ego_cuda_create(int width, int height) {
  try {
    if (width < 16 || height < 16 || width % 8 || height % 8 ||
        static_cast<int64_t>(width) * height >= (1 << 22))
      throw std::runtime_error("CUDA images need multiples of 8 and fewer than 4194304 pixels");
    static std::once_flag logging;
    std::call_once(logging, [] { google::InitGoogleLogging("ego_apriltag_cuda"); });
    check(cudaFree(nullptr));
    return new Detector(width, height);
  } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
API int ego_cuda_detect(void* handle, const uint8_t* gray, double* corners,
                        int* ids, int* hamming, float* margins, int capacity) {
  try {
    auto& detector = *static_cast<Detector*>(handle);
    for (size_t i = 0; i < detector.yuyv.size() / 2; ++i)
      detector.yuyv[2 * i] = gray[i];
    detector.gpu->Detect(detector.yuyv.data());
    check(cudaGetLastError());
    const zarray_t* found = detector.gpu->Detections();
    int count = zarray_size(found);
    if (count > capacity) throw std::runtime_error("CUDA detection output capacity exceeded");
    for (int i = 0; i < count; ++i) {
      apriltag_detection_t* tag;
      zarray_get(found, i, &tag);
      ids[i] = tag->id;
      hamming[i] = tag->hamming;
      margins[i] = tag->decision_margin;
      for (int c = 0; c < 4; ++c)
        for (int axis = 0; axis < 2; ++axis)
          corners[i * 8 + c * 2 + axis] = tag->p[c][axis];
    }
    return count;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
API void ego_cuda_destroy(void* handle) { delete static_cast<Detector*>(handle); }
