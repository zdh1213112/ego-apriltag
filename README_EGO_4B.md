# EGO-4B AprilTag 使用指南

**操作顺序：准备录制 → 运行处理 → 查看结果。** 跟踪、优化、恢复和视频生成由程序连续执行。

本机已经安装好 CPU 和 CUDA 检测后端，直接从第 1 步开始。新电脑先完成一次[安装与编译](docs/EGO_COMMANDS.md#首次使用安装与检查)；使用 NVIDIA GPU 时再完成[CUDA 后端编译](docs/EGO_COMMANDS.md#启用-nvidia-gpu仅首次编译)。

## 1. 准备录制并进入项目

先用 EGOViewer 录好四路相机视频，保留完整录制目录。下面使用本次的 `EGO-4B_AZLCA610091_20261119_181247`。

录制目录里应能找到 `meta_*.json`。命令要指向这一层，不能指向里面的 `part0001/`，也不能填之前生成的结果目录。

支持四路 MP4 配时间戳 CSV，也支持 EGOViewer 录制的 H.264 MCAP。两种格式使用同一条命令。对于 MCAP，程序从每路相机消息读取原始时间戳，用 FFmpeg 临时封装视频供跟踪、恢复和预览使用。

```bash
cd /home/zdh/apriltag
```

准备好同一物体上的 **ID 0、1、2、3**，标签家族为 **tag36h11**，各码尺寸相同、相对固定。标签边长要量黑色外框外缘，不含外围白边。

## 2. 运行完整处理命令

复制下面整段执行。这条命令处理完整录制，并以 **ID 0** 为参考标签：

```bash
/usr/bin/python3 scripts/run_ego_apriltag.py \
  recordings/EGO-4B_AZLCA610091_20261119_181247 \
  --family tag36h11 \
  --tag-ids 0 1 2 3 \
  --tagsize 0.030 \
  --reference 0 \
  --device cuda \
  --flow-max-frames 15 \
  --no-filter \
  --max-sync-us 1000 \
  --max-reprojection 2.5 \
  --alignment-max-translation 0.010 \
  --alignment-max-rotation 10 \
  --recovery on \
  --max-gap-frames 3 \
  --trail-seconds 1.5
```

这条命令使用 `--no-filter` 导出未滤波位姿。需要自适应平滑时，将它替换为 `--cutoff 4`。

命令中明确指定标签和计算后端，换录制时逐项核对：

| 命令中的内容 | 含义 | 怎么改 |
|---|---|---|
| `recordings/EGO-4B_AZLCA610091_20261119_181247` | 原始录制路径 | 换成自己的录制目录；也可以使用完整路径 |
| `--family tag36h11` | 标签家族 | 填实际使用的标签家族 |
| `--tag-ids 0 1 2 3` | 需要跟踪的标签 ID | 写出目标物体上的所有标签 ID |
| `--tagsize 0.030` | 标签边长，单位为米 | 20 mm 填 `0.020`，30 mm 填 `0.030` |
| `--reference 0` | 以 ID 0 的中心和坐标轴为物体参考 | 以 ID 1 为参考时改成 `--reference 1` |
| `--device cuda` | 用 NVIDIA GPU 加速全图标签检测 | 使用 CPU 时明确改成 `--device cpu` |

**标签家族和 ID 列表必须显式填写，程序不会自动选择。** 上面的命令跟踪 `tag36h11` 的 ID 0、1、2、3。参考 ID 必须属于这份列表；改参考 ID 不会减少跟踪的标签数量。

一键命令以显式填写的 `--reference` 为准，`--tag-ids` 的顺序不会覆盖它。以 ID 1 为参考时，把 `--reference 0` 改成 `--reference 1` 即可。

优化配置也直接写在同一条命令中，修改对应数值即可：

| 命令中的配置 | 含义 | 怎么改 |
|---|---|---|
| `--flow-max-frames 15` | 未重新解码时最多续跟踪 15 帧 | `0` 关闭主跟踪光流；增大可能增加漂移 |
| `--cutoff 4` | 自适应平滑的最低截止频率，单位 Hz | 调小主要增强低速平滑，也会增加延迟；同时用于主跟踪和恢复结果 |
| `--no-filter` | 跳过主跟踪和恢复阶段的位姿滤波 | `filtered_*` 字段沿用 `raw_*` 位姿，默认输出目录增加 `_no_filter` 后缀 |
| `--max-sync-us 1000` | 跨相机观测匹配允许的最大时间差，单位微秒 | 必须与实际相机同步情况相符 |
| `--max-reprojection 2.5` | 重投影误差门限，单位像素 | 调小检查更严格，也可能增加丢帧；同时用于主跟踪和恢复 |
| `--alignment-max-translation 0.010` | 码间位置一致性门限，单位米；这里为 10 mm | 需要更严格时可收紧，可能使部分码无法标定 |
| `--alignment-max-rotation 10` | 码间朝向一致性门限，单位度 | 需要更严格时可收紧 |
| `--recovery on` | 执行离线重检和双向光流恢复 | 改成 `off` 时只跟踪并生成视频 |
| `--max-gap-frames 3` | 恢复阶段最多插值补 3 帧 | 改为 `0` 表示不插值，仍执行视觉恢复；仅在恢复开启时有效 |
| `--trail-seconds 1.5` | 视频中显示参考 ID 最近 1.5 秒的轨迹拖影 | `0` 关闭拖影；改参考 ID 时拖影随之更换 |

背景同码筛选和多相机拟合属于主跟踪流程，使用上面的误差门限检查观测。本次实际使用的参数会保存到结果目录的 `run_settings.json`。

上面的命令明确启用了恢复，程序按顺序自动完成：

| 终端进度 | 自动处理的内容 |
|---|---|
| `1/3 跟踪标签并计算码间外参` | 筛选背景同码、多相机拟合、异常观测剔除和码间标定 |
| `2/3 恢复短暂丢帧` | 重新检测、双向光流恢复，并尝试插值补齐符合条件的短缺口 |
| `3/3 生成预览视频` | 用最终位姿生成四相机预览视频 |

等待终端显示“完成”后再查看结果。**不需要再分别运行跟踪、恢复和渲染脚本。**

`--device cuda` 加速主跟踪与离线恢复中的全图阈值、连通域和四边形查找。码字解码、边缘细化、局部鱼眼重检、OpenCV 光流、KB 多相机位姿拟合和视频生成仍在 CPU 上执行。保留原始分辨率的重叠分块检测用于减少小标签漏检，仍保留同 ID 的多个候选。当前 CUDA 后端支持 `tag36h11`；其他家族使用 `--device cpu`。

终端会明确显示后端，并持续输出相机帧进度、位姿求解和候选筛选进度；详细记录保存到 `processing.log`。CUDA 不可用时会报错，不会静默切回 CPU。未填写 `--device` 的旧命令仍使用 CPU。

## 3. 查看视频、外参和位姿

结束时终端会直接打印“视频”“外参”“位姿”的完整路径。上面的命令对应结果目录为：

```text
outputs/EGO-4B_AZLCA610091_20261119_181247_30mm_ref0_cuda_no_filter/
```

| 先后顺序 | 文件 | 用途 |
|---|---|---|
| 先看视频 | `tracking_preview.mp4` | 确认框线和坐标轴跟随手持物体，背景同码没有被当成目标 |
| 再看外参 | `tag_alignment.json` | 查看各标签到所选参考标签的固定变换和标定质量 |
| 需要导出数据时 | `global_poses.csv` / `global_poses.npz` | 每帧的位置、朝向和跟踪状态 |
| 需要看变化趋势时 | `trajectory.png` / `orientation.png` | 位置和朝向曲线 |
| 需要核对本次配置时 | `run_settings.json` | 实际采用的标签、优化和恢复参数 |

外参文件中的 `reference_tag_id` 表示所选参考 ID，`T_reference_tag` 保存各码到它的固定变换。统一位姿仍表达在头环的 `cam_0` 相机坐标系中，位置单位为米。

本例显式设置最多插值补 3 帧，且两端有效位姿间隔还须不超过 0.15 秒。插值是估计值，视频显示 `ESTIMATED`，数据标为 `interpolated`；长缺口保留为 `lost`。无法可靠标定的 ID 会被列出，不参与融合。

CUDA 结果目录自动加 `_cuda`，保留之前的 CPU 结果；CPU 目录没有这个后缀。再次运行相同录制、尺寸、参考 ID、后端和滤波配置会更新同一目录。使用 `--output` 时以指定路径为准。

## 已经处理完成：只给视频加拖影

已有位姿时，直接执行下面这条命令，读取原始视频重新渲染，不重复检测或求解位姿：

```bash
/usr/bin/python3 scripts/render_ego_recovery.py \
  outputs/EGO-4B_AZLCA610091_20261119_181247_30mm_ref0_cuda_no_filter \
  --trail-seconds 1.5 \
  --output outputs/EGO-4B_AZLCA610091_20261119_181247_30mm_ref0_cuda_no_filter/with_trail
```

完成后查看 `with_trail/tracking_preview.mp4`，原来的预览视频保留。`with_trail/reference_trail.csv` 保存各帧轨迹点和状态。

拖影显示**所选参考 ID 的中心在头环 `cam_0` 坐标系中的运动历史**，再按相机外参投影到四路画面。它包含物体与头环的相对运动。使用当前未平滑位姿与视频坐标轴对齐；越旧的线段越淡，青色表示视觉位姿，橙色表示包含插值的线段，丢失处断开。

## 检查外参是否稳定

```bash
/usr/bin/python3 scripts/check_ego_alignment.py \
  outputs/EGO-4B_AZLCA610091_20261119_181247_30mm_ref0_cuda_no_filter
```

检查报告保存为 `alignment_quality.json`，统计各标签相对参考 ID 的位置、朝向偏差，以及前后半段估计的差异。检查仅使用两个标签均有至少两路相机支持的解码位姿，排除纯光流和插值帧。

这些是同一录制内的一致性指标，且观测已经经过该布局筛选，存在选择偏差，不能当作绝对精度或独立验证误差。跟踪覆盖率表示有多少帧可以输出位姿；判断毫米级准确度还需要独立实测尺寸或外部定位真值。

## 常用操作，按需要选

下面是对第 2 步命令的调整，不是后续必做步骤：

| 需要什么 | 怎么做 |
|---|---|
| 先试跑一小段，确认设置正确 | 在命令末尾加 `--max-frames 120 --output outputs/ego_check`，结果单独保存在 `outputs/ego_check/`；确认后用第 2 步原命令处理完整录制 |
| 只保留视觉恢复，不插值补帧 | 将 `--max-gap-frames 3` 改为 `--max-gap-frames 0`，保留 `--recovery on` |
| 已经检测过，只调整位姿或优化参数 | 在相同输出目录下加 `--reuse-detections`，省去重新检测；录制、家族、ID、处理帧数、光流帧数、计算后端和缓存版本须匹配 |

其他标签 ID、指定结果目录和常见报错，见[安装与可选设置](docs/EGO_COMMANDS.md)。需要了解算法或坐标变换时，再看[技术说明](docs/EGO_MULTICAM.md)。
