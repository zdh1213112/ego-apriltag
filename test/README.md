# 测试与开发诊断

测试不参与日常跟踪。原有 C 测试验证官方检测器、码字解码与位姿接口；`test_ego_*.py` 验证四相机几何、背景同码筛选、恢复、命令参数传递和轨迹拖影。保留这些检查用于确认修改没有破坏原有功能。

以下命令在项目根目录执行，使用与编译相同的 Python。

## 1. 编译并验证官方接口

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_PYTHON_WRAPPER=ON \
  -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

## 2. 验证 EGO 功能

```bash
/usr/bin/python3 -m unittest discover -s test -p 'test_ego_*.py'
```

这些检查使用临时目录和合成观测，不依赖本机录制或旧输出。CUDA 实机检查默认跳过，不影响只使用 CPU 的电脑。

## 3. 验证 CUDA 后端（可选）

先按[安装说明](../docs/EGO_COMMANDS.md#启用-nvidia-gpu仅首次编译)编译 CUDA 后端，再执行：

```bash
EGO_TEST_CUDA=1 /usr/bin/python3 -m unittest discover -s test -p 'test_ego_detector.py'
```

这会在真实 GPU 上检查旋转标签、重复 ID 和角点顺序。需要可访问的 NVIDIA GPU，不会用 CPU 代替 CUDA 检查。

## 4. 比较检测耗时（可选开发工具）

```bash
/usr/bin/python3 test/benchmark_ego_detector.py \
  recordings/EGO-4B_AZLCA610091_20261117_110204 \
  --samples 12 --output outputs/ego_cuda_benchmark.json
```

需要自己的录制和可用的 CUDA 后端。该工具比较 CPU/CUDA 检测耗时、ID 0–3 的检出数量和角点差异，不是完整跟踪流程的速度或绝对精度评估。
