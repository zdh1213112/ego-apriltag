#!/usr/bin/env bash
# Optional local build; no sudo, ROS, Docker or Python environment changes.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_COMPILER="${CUDACXX:-$(command -v nvcc || true)}"
if [[ -z "$CUDA_COMPILER" && -x /usr/local/cuda-12.8/bin/nvcc ]]; then
  CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
fi
if [[ -z "$CUDA_COMPILER" ]]; then
  echo '未找到 nvcc；请安装支持本机显卡的 CUDA Toolkit。' >&2
  exit 1
fi
GPU_ARCH="${EGO_CUDA_ARCHITECTURES:-}"
if [[ -z "$GPU_ARCH" ]]; then
  GPU_ARCH="$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader | tr -d '. ')"
fi
cmake -S "$PROJECT_ROOT/scripts/ego_cuda" -B "$PROJECT_ROOT/build/ego_cuda" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" "$@"
cmake --build "$PROJECT_ROOT/build/ego_cuda" --target ego_apriltag_cuda --parallel 2
echo 'CUDA 检测后端已生成。跟踪命令增加 --device cuda 即可使用。'
