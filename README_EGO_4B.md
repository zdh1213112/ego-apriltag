# EGO-4B AprilTag 上手指南

这份文档面向 **Orbbec EGO-4B 四相机头环**：用 EGOViewer 录好视频后，在本项目中跟踪 AprilTag，得到相对头环的三维位置、朝向和预览视频。

**日常流程：设置录制路径和标签尺寸 → 小段试跑 → 完整跟踪 → 按需恢复丢帧并生成视频。**

本指南独立于原有 [README.md](README.md)。详细算法说明见 [docs/EGO_MULTICAM.md](docs/EGO_MULTICAM.md)。

- [日常使用：按顺序复制执行](#日常使用按顺序复制执行)
- [结果怎么看](#结果怎么看)
- [其他常用指令](#其他常用指令)
- [首次使用：安装与检查](#首次使用安装与检查)
- [录制数据怎么准备](#录制数据怎么准备)
- [常见问题](#常见问题)

## 日常使用：按顺序复制执行

以下命令使用 **Bash**，在项目根目录执行，统一使用 `/usr/bin/python3`。新机器先完成[安装与检查](#首次使用安装与检查)。

### 1. 设置本次要处理的数据

示例使用本机已有的 `20261028_112103` 录制，其已有结果记录的设置为 `tag36h11`、ID `0 1 2`、检测边长 **20 mm**。录制数据不随 Git 分发；其他机器请换成自己的录制。

```bash
cd /home/zdh/apriltag

RUN="EGO-4B_AZLCA610091_20261028_112103"
TAGSIZE="0.020"
RECORDING="$PWD/recordings/$RUN"
OUT="$PWD/outputs/${RUN}_${TAGSIZE}m"
```

换数据时，修改 `RUN` 和 `TAGSIZE` 后，把上面这一整段重新执行。若项目放在其他位置，也要修改 `cd` 路径。若录制放在项目外，直接把 `RECORDING` 改为该录制的完整路径。

| 设置 | 含义 |
|---|---|
| `RUN` | 本次录制名称，也用于给结果目录命名 |
| `TAGSIZE` | 标签检测边长，单位为米；20 mm 填 `0.020`，30 mm 填 `0.030` |
| `RECORDING` | 原始录制目录，里面应有 `meta_*.json` |
| `OUT` | 保存本次结果的目录，由脚本自动创建 |

后续命令默认跟踪 **同一物体上、同一尺寸的 ID 0、1、2**。如果只贴了一个码，使用[单码跟踪命令](#只跟踪一个标签)。如果使用其他家族或 ID，修改命令里的 `--family`、`--tag-ids`。

**尺寸要实测：** `tag36h11` 量黑色外框外缘的一条边，不含外围白边，也不是整张纸的宽度。其他家族的测量边界见[尺寸说明](tag_family_examples/README.md#tagsize-怎么量)。多码命令要求所有码尺寸相同。

### 2. 先试跑前 120 帧

```bash
/usr/bin/python3 scripts/track_ego_apriltag.py "$RECORDING" \
  --family tag36h11 --tag-ids 0 1 2 --tagsize "$TAGSIZE" \
  --max-frames 120 \
  --output "${OUT}_check"
```

30 fps 的录制中，120 帧约为前 4 秒画面；处理耗时取决于电脑性能。完成后，先用视频播放器打开 `${OUT}_check/tracking_preview.mp4`，确认标签 ID、框线和坐标轴合理。

试跑结果单独写入带 `_check` 的目录。若提示 `Cannot align tag IDs`，参照[常见问题](#常见问题)，适当增加试跑帧数，让多个标签有共同可见的机会。

### 3. 处理完整录制

```bash
/usr/bin/python3 scripts/track_ego_apriltag.py "$RECORDING" \
  --family tag36h11 --tag-ids 0 1 2 --tagsize "$TAGSIZE" \
  --output "$OUT"
```

不带 `--max-frames` 就处理完整输入视频。完成后，终端会打印输出目录、各 ID 的有效帧数，以及整体跟踪覆盖率。结果先看 `$OUT/tracking_preview.mp4` 和 `$OUT/trajectory.png`。

**一般使用到这一步即可。** 如果视频里有短暂丢失，再执行下面两步。

### 4. 恢复短暂丢帧（可选）

这一步需要上一步的**多码结果目录**，同时需要原始录制仍保存在原路径。

```bash
/usr/bin/python3 scripts/recover_ego_apriltag.py "$OUT" \
  --output "${OUT}_recovered"
```

脚本在丢帧附近重新检测、双向光流跟踪，并尝试补齐符合条件的短缺口。默认最多插值补 3 帧，且两端有效位姿间隔不超过 0.15 秒；长缺口和首尾缺口保留为 `lost`。

恢复后的统一位姿、图表和统计写入 `${OUT}_recovered`，原始跟踪结果保留。**插值帧是估计值**，在表格中标为 `status=interpolated`，图中以橙色点标出。

如果不希望做插值，使用下面的命令替代上面的恢复命令：

```bash
/usr/bin/python3 scripts/recover_ego_apriltag.py "$OUT" \
  --max-gap-frames 0 --output "${OUT}_recovered"
```

### 5. 为恢复结果生成视频（可选）

恢复脚本只生成位姿、图表和统计。要看恢复后的画面，再运行：

```bash
/usr/bin/python3 scripts/render_ego_recovery.py "${OUT}_recovered"
```

完成后查看 `${OUT}_recovered/tracking_preview.mp4`；单路视频在同目录的 `camera_previews/` 中。视频里的 `ESTIMATED` 和橙色边框表示插值帧。此步骤需要带 `libx264` 编码器的 FFmpeg。

重新渲染还会读取原始跟踪结果和录制，所以请一起保留 `RECORDING`、`OUT` 和恢复目录。

## 结果怎么看

下表针对 ID `0 1 2` 的多码跟踪。优先看根目录的**统一位姿**；需要某个标签自己的结果时，再进入 `tag_0/`、`tag_1/` 或 `tag_2/`。

| 文件 | 用途 |
|---|---|
| `tracking_preview.mp4` | 四路拼接预览，检查跟踪效果 |
| `camera_previews/cam_0.mp4` 至 `cam_3.mp4` | 各相机单独预览 |
| `trajectory.png` | 统一位姿的位置曲线，单位为米 |
| `orientation.png` | 相对第一帧有效朝向的旋转向量曲线，单位为度，不是欧拉角 |
| `tracking_status.png` | 整体跟踪状态 |
| `global_poses.csv` / `global_poses.npz` | 每帧一份统一位姿，供后续程序使用 |
| `global_tracking.csv` | 原始多码跟踪中每帧的整体状态、有效码数量 |
| `poses.csv` / `poses.npz` | 原始多码跟踪中各个 ID 的独立位姿合集 |
| `tag_0/`、`tag_1/`、`tag_2/` | 各 ID 的独立结果、曲线和检测缓存 |
| `tag_alignment.json` | 多个标签到统一参考标签的固定变换 |
| `summary.json` | 跟踪质量或恢复统计 |

恢复目录主要保存更新后的 `global_poses.*`、图表和恢复统计，生成视频后再增加预览文件；各 ID 的独立结果仍从原始结果目录读取。

**坐标含义：** `cam_0` 是头环前左相机，X 向右、Y 向下、Z 向前。位置单位为米，四元数顺序为 `x, y, z, w`。`global_poses` 的 `global` 表示多个码融合后的统一参考；参考标签是 `--tag-ids` 中的第一个 ID，本例为 ID 0。

变换方向为 `p_cam0 = R @ p_tag + t`。输出相对头环相机，头环移动时坐标也随之变化；当前没有融合 IMU 来建立固定世界坐标轨迹。

统一位姿要求多个码相对固定，并有足够共同观测来估计码间关系。ID 0 暂时不可见时，可由其他已对齐的码推算该参考位姿；腕带明显变形会影响结果。

| `status` | 含义 |
|---|---|
| `measured` | 当前帧有成功解码并通过位姿检查的标签 |
| `flow_only` | 当前帧依赖通过几何检查的光流观测 |
| `interpolated` | 恢复流程插值得到的估计，没有当前帧视觉测量 |
| `lost` | 无可靠位姿，变换矩阵为 NaN |

`raw` 是联合估计，`filtered` 是平滑后的位姿，平滑会带来一定延迟。恢复结果还提供 `pose_source`，区分重新解码、光流恢复、插值及沿用原始结果。

需要在 Python 中读取统一位姿时，在设置过 `OUT` 的终端执行：

```bash
/usr/bin/python3 - "$OUT/global_poses.npz" <<'PY'
import sys
import numpy as np

data = np.load(sys.argv[1])
T = data["T_cam0_global_filtered"]  # 形状：(帧数, 4, 4)
visual_valid = np.isin(data["status"], ["measured", "flow_only"])
positions_m = T[visual_valid, :3, 3]
print("总帧数：", len(T))
print("视觉有效帧数：", int(visual_valid.sum()))
print("有效位置数组形状：", positions_m.shape)
PY
```

读取恢复结果时，把路径换成 `"${OUT}_recovered/global_poses.npz"`。上述筛选会排除插值帧和丢失帧。

## 其他常用指令

以下命令沿用第一步设置的 `RECORDING`、`TAGSIZE` 和 `OUT`。新开终端后需要重新执行那一段设置。

### 只跟踪一个标签

例如只跟踪 ID 0：

```bash
/usr/bin/python3 scripts/track_ego_apriltag.py "$RECORDING" \
  --family tag36h11 --tag-id 0 --tagsize "$TAGSIZE" \
  --output "${OUT}_id0"
```

`--tag-id` 与 `--tag-ids` 二选一。单码结果直接保存在指定目录的 `poses.csv`、`poses.npz` 中，矩阵键名为 `T_cam0_tag_raw` 和 `T_cam0_tag_filtered`；没有多码的 `global_poses.*`，不使用上面的多码恢复流程。

### 已经跟踪过，只重算位姿和图表

在已有**完整跟踪结果**的基础上复用检测缓存：

```bash
/usr/bin/python3 scripts/track_ego_apriltag.py "$RECORDING" \
  --family tag36h11 --tag-ids 0 1 2 --tagsize "$TAGSIZE" \
  --output "$OUT" --reuse-detections --no-video
```

录制路径、家族、ID、处理帧数、光流设置和代码缓存版本需要与缓存一致。`_check` 目录只包含试跑帧的缓存，不能直接当作全量缓存使用。提示 `Cached detection settings differ` 时，去掉 `--reuse-detections` 重新检测。

该命令会更新结果表格和图表。`--no-video` 会跳过视频生成，已有视频也不会随之更新；需要同步更新视频时，去掉 `--no-video`。

### 本机已有录制用过哪些尺寸

下表来自已有结果的 `summary.json`，便于核对历史命令；新录制的标签仍应实测。

| `RUN` | 家族 / ID | `TAGSIZE` |
|---|---|---|
| `EGO-4B_AZLCA610091_20261028_112103` | `tag36h11` / `0 1 2` | `0.020`（20 mm） |
| `EGO-4B_AZLCA610091_20261027_190651` | `tag36h11` / `0 1 2` | `0.030`（30 mm） |
| `EGO-4B_AZLCA610091_20261027_182506` | `tag36h11` / `0 1 2` | `0.040`（40 mm） |

### 查看全部参数

```bash
/usr/bin/python3 scripts/track_ego_apriltag.py --help
/usr/bin/python3 scripts/recover_ego_apriltag.py --help
/usr/bin/python3 scripts/render_ego_recovery.py --help
```

| 跟踪参数 | 用法 |
|---|---|
| `--max-frames 120` | 只处理前 120 帧；不传或填 `0` 则处理完整输入 |
| `--no-video` | 跳过预览视频，只输出数据和图表 |
| `--reuse-detections` | 复用指定输出目录里的检测缓存 |
| `--flow-max-frames 15` | 默认值，最多连续 15 帧未解码时继续角点跟踪 |
| `--flow-max-frames 0` | 关闭光流续跟踪 |
| `--cutoff 4` | 默认平滑最低截止频率，单位 Hz |

## 首次使用：安装与检查

本机已有可用配置：Ubuntu、`/usr/bin/python3` 3.10、OpenCV 4.5.4。下面采用系统 Python 安装方式。

### 安装依赖并编译

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

脚本自动从项目的 `build/` 加载 AprilTag Python 绑定，按上述方式编译后即可运行。

如果选择自己的虚拟环境，可参考 [requirements-ego.txt](requirements-ego.txt) 安装 Python 依赖，并用该环境的 Python 重新配置、编译绑定。编译和运行必须使用相同的 Python / NumPy 环境。

### 确认检测器可运行

```bash
/usr/bin/python3 example/python_demo.py \
  tag_family_examples/apriltag/tag36h11/id_0000.png \
  --family tag36h11 --output build/detection_check.jpg
```

正常会显示 `Detected 1 tags (tag36h11)`、`ID=0`，并生成 `build/detection_check.jpg`。检查通过后，回到[日常使用](#日常使用按顺序复制执行)。

## 录制数据怎么准备

先在 EGOViewer 中完成四路相机录制，保留其原始目录结构。把录制放在 `recordings/` 下，或者通过 `RECORDING` 指定外部目录。录制后的离线处理不要求设备持续连接。

一个带 `part0001` 子目录的录制通常如下：

```text
recordings/
└── EGO-4B_设备序列号_日期_时间/        ← RECORDING 指向这一层
    ├── meta_*.json
    ├── *_calibration_camera.yaml
    └── part0001/
        ├── *_camera_left_part0001.mp4
        ├── *_camera_left_part0001_pts.csv
        ├── *_camera_right_part0001.mp4
        ├── *_camera_right_part0001_pts.csv
        ├── *_camera_right_side_part0001.mp4
        ├── *_camera_right_side_part0001_pts.csv
        ├── *_camera_left_side_part0001.mp4
        └── *_camera_left_side_part0001_pts.csv
```

脚本通过 `meta_*.json` 中的 `files` 字段找到相机标定、四路视频和时间戳 CSV，兼容文件直接位于录制根目录的布局。保留元数据中的相对路径，避免只复制 MP4 或单独传入 `part0001/`。

当前脚本按 `files` 指向的四路视频处理，不自动遍历、拼接 `segments` 中的多个分段。长录制若有多个分段，需要先另行整理元数据、视频和时间戳。

| 相机 | EGOViewer 视频名 | 位置 |
|---|---|---|
| `cam_0` | `left` | 前左，输出坐标的参考相机 |
| `cam_1` | `right` | 前右 |
| `cam_2` | `right_side` | 右侧 |
| `cam_3` | `left_side` | 左侧 |

标签图案可从 [tag36h11 示例目录](tag_family_examples/apriltag/tag36h11) 取得。PNG 没有固定的物理打印尺寸，打印后仍须测量。录制时让多个码有一段共同可见的清晰画面，并遮住背景中与目标重复的同 ID 标签。

## 常见问题

| 现象 | 处理方法 |
|---|---|
| `No meta_*.json found` | 跟踪命令需要原始录制根目录。检查是否误传了结果目录、`part0001/` 或 `recordings/` 的上层目录 |
| 提示缺少 `global_poses.npz` 或 `tag_alignment.json` | 恢复命令需要成功生成统一位姿的多码结果根目录，不能传单码目录、`tag_0/` 或原始录制 |
| `Cannot align tag IDs` | 多码缺少足够的共同解码观测；增加试跑帧数或补录清晰共同可见画面。当前至少需要 3 帧共同有效解码观测连接相关 ID，允许通过其他 ID 间接连接 |
| `Cannot import the AprilTag binding` / `No module named apriltag` | 先编译，再使用与编译一致的 `/usr/bin/python3`；确认 `build/` 中有对应 Python 版本的 `apriltag.*.so` |
| 提示缺少 `cv2`、`scipy`、`yaml` 等模块 | 在运行脚本所用的 Python 环境中安装依赖，参照安装章节 |
| `Cached detection settings differ` | 去掉 `--reuse-detections`，重新生成当前设置的缓存 |
| `Cannot open` / `Early video end` / 分辨率不符 | 检查录制是否完整，元数据、PTS、相机标定是否与视频配套；不要先裁剪或缩放视频 |
| 尺度明显不对 | 核实 `--tagsize` 的测量边界和米单位；目录名中的尺寸后缀不会设置实际标签尺寸 |
| 大量 `lost` | 先核对家族、ID、尺寸和画面清晰度；检查遮挡、模糊及重复标签，再考虑短缺口恢复 |
| 恢复后还没有视频 | 继续执行 `render_ego_recovery.py`；确认 FFmpeg 可用且带 `libx264` |
| 新终端中变量为空或路径不对 | 重新执行第一步的完整设置块；变量只保存在当前终端 |

结果目录重复使用时，同名数据和图表会被更新。需要保留某次结果时，为 `OUT` 换一个新的后缀；`recordings/`、`outputs/` 和 `build/` 都是本地目录，不随 Git 提交。
