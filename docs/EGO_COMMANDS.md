# EGO-4B：安装与可选设置

日常操作只有三步：**进入项目 → 运行处理命令 → 打开结果视频**。完整操作见[使用说明](../README_EGO_4B.md)。

## 首次使用：安装与检查

**仅新电脑执行一次。** 安装完成后，回到[三步操作](../README_EGO_4B.md)。本机已经安装好的用户跳过本节。

在 Ubuntu 终端中按顺序执行下面整段：

```bash
cd /home/zdh/apriltag

sudo apt update
sudo apt install build-essential cmake python3-dev python3-numpy \
  python3-opencv python3-scipy python3-yaml python3-matplotlib python3-pil \
  fonts-noto-cjk ffmpeg

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_PYTHON_WRAPPER=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build --parallel 4
```

## 启用 NVIDIA GPU：仅首次编译

本机 RTX 5060 已编译好，直接在跟踪命令中写 `--device cuda`。新电脑需要 NVIDIA 驱动、支持该显卡的 CUDA Toolkit（含 `nvcc`）、上述 CPU 依赖和 `libgflags-dev`。本机验证使用 CUDA Toolkit 12.8、驱动 580.142。

在项目根目录依次执行一次：

```bash
sudo apt install libgflags-dev
bash scripts/setup_ego_cuda.sh
```

脚本按本机显卡架构编译，把固定版本的依赖与后端保存在 `build/ego_cuda/`，不安装 ROS、Docker 或更换 Python 环境。首次编译需要联网；这些构建产物不随 Git 分发。之后使用[完整命令](../README_EGO_4B.md#2-运行完整处理命令)中的 `--device cuda`；切换回 CPU 只需改为 `--device cpu`。主跟踪和离线恢复都会收到该参数。

CUDA 加速全图阈值、连通域和四边形查找；码字解码、边缘细化、局部重检、光流、KB 位姿拟合和视频编码仍使用 CPU。仅支持 `tag36h11`。GPU 失效时明确报错，不自动回退。为保留 20 mm 小码的检出能力，使用原始分辨率的重叠分块检测；CUDA 与 CPU 结果可能有少量差异，仍需检查外参和预览。

GPU 来源为 [Team766/apriltags_cuda](https://github.com/Team766/apriltags_cuda)，码字解码使用其指定的 [cgpadwick/apriltag](https://github.com/cgpadwick/apriltag)；构建版本固定在 `scripts/ego_cuda/CMakeLists.txt`。本机四路各抽 12 帧，检测中位耗时约从每帧 40–48 ms 降到 21–24 ms；这是检测阶段速度，不代表完整流程提速相同倍数。

## 必填的标签设置

处理命令必须包含 `--family` 和 `--tag-ids`，明确说明标签家族以及需要跟踪的 ID。例如 `--family tag36h11 --tag-ids 0 1 2 3`。程序不会为这两项选择默认值。

所有标签应在同一物体上，尺寸相同、相对固定；`tag36h11` 的边长量黑色外框外缘，不含外围白边。参考 ID 必须属于实际跟踪的标签。

## 可选参数

标签和优化参数都在[使用说明的完整命令](../README_EGO_4B.md#2-运行完整处理命令)中显式列出，逐项修改对应值。下面是常用调整；已经写过的参数替换原值即可。

| 需要什么 | 参数 |
|---|---|
| 以 ID 1 为参考 | `--reference 1`，替换原来的 `--reference 0` |
| 使用 NVIDIA GPU 检测 | `--device cuda`；首次编译见上节 |
| 使用 CPU 检测 | `--device cpu` |
| 只跟踪 ID 0、1、2 | 把原来的 ID 列表替换为 `--tag-ids 0 1 2` |
| 先试跑 120 帧 | `--max-frames 120` |
| 不插值补帧 | 将原来的插值帧数改成 `--max-gap-frames 0`，保留 `--recovery on` |
| 关闭离线恢复，只跟踪并生成视频 | `--recovery off` |
| 自己指定保存目录 | `--output outputs/my_result` |
| 复用已有角点检测缓存 | `--reuse-detections`；输入录制和检测设置须与缓存一致 |
| 显示参考 ID 最近 1.5 秒的轨迹 | `--trail-seconds 1.5`；`0` 关闭 |

## 结果说明

完成后，终端直接显示视频、外参和位姿的完整路径。使用指南中的 20 mm 尺寸和参考 ID 0 对应的结果目录为：

```text
outputs/EGO-4B_AZLCA610091_20261117_110204_20mm_ref0_cuda/
```

| 文件 | 用途 |
|---|---|
| `tracking_preview.mp4` | 最终预览视频 |
| `tag_alignment.json` | 各标签到参考标签的外参、标定质量 |
| `global_poses.csv` / `global_poses.npz` | 每帧的位姿与状态 |
| `trajectory.png` / `orientation.png` | 位置和朝向曲线 |
| `tracking/` | 原始跟踪结果、各码位姿及检测缓存 |
| `processing.log` | 出错时查看详细原因 |
| `run_settings.json` | 本次实际使用的标签及优化参数 |
| `reference_trail.csv` | 开启拖影时导出的参考 ID 中心轨迹，坐标为头环 `cam_0`，单位米 |
| `alignment_quality.json` | 运行外参检查后生成的内部一致性与分时段稳定性报告 |

示例命令明确设置了光流、滤波、误差门限、离线恢复和插值帧数。`--recovery on` 时执行离线重检及双向光流，`--max-gap-frames 3` 允许最多 3 帧短缺口插值。插值结果是估计值，会标为 `interpolated`，视频中显示 `ESTIMATED`；长缺口保留为 `lost`。设为 `--recovery off` 时不做离线恢复或插值。

参考标签决定物体上哪个坐标系的位姿被输出，位置仍表达在头环 `cam_0` 坐标系中。未可靠标定的标签会明确列出，不参与融合。

CUDA 默认目录有 `_cuda` 后缀，CPU 目录没有该后缀。再次运行相同录制、尺寸、参考 ID 和后端会更新同一个结果目录。需要保留旧结果时，用 `--output` 指定另一个目录。

程序会显示实际后端与分阶段进度。重复处理相同输入时可复用同一后端的检测缓存；它仍会重新做位姿拟合和刚性候选筛选。CPU 与 CUDA 缓存不能互相复用；切换后端时不要加 `--reuse-detections`。

已完成的恢复结果可直接用 `render_ego_recovery.py 结果目录 --trail-seconds 1.5 --output 另一个视频目录` 增加拖影，保留原视频。拖影采用参考 ID 在头环坐标系中的相对运动，青色表示视觉位姿，橙色表示插值线段。外参检查和重新渲染的完整命令见[使用指南](../README_EGO_4B.md)。

## 遇到问题

| 提示 | 怎么做 |
|---|---|
| 找不到 `meta_*.json` | 把录制目录改成包含该文件的那一层，不能填结果目录或 `part0001/` |
| 缺少 Python 模块或 `apriltag` | 完成上面的安装和编译，使用 `/usr/bin/python3` 运行 |
| 缺少 FFmpeg | 完成上面的安装步骤 |
| CUDA 后端尚未编译 | 完成上面的 CUDA 编译步骤，或显式使用 `--device cpu` |
| CUDA 不可用 | 检查 `nvidia-smi`、驱动及运行环境的 GPU 访问权限 |
| 未能标定某个标签 | 检查该码是否属于目标，让多个码有一段清晰共同可见、随物体运动的画面 |
| 程序报错 | 查看终端显示的 `processing.log` 路径 |

算法、坐标含义和底层脚本参数见[技术文档](EGO_MULTICAM.md)；需要修改源码时，按[测试说明](../test/README.md)验证官方接口、EGO 流程和可选 CUDA 后端。
