# EGO 四相机 AprilTag 位姿跟踪

本扩展在 AprilTag 3 官方源码基础上增加 Python 示例和 Orbbec EGO 录制跟踪。
跟踪结果为标签坐标系到头环前左相机 `cam_0` 光学坐标系的刚体变换。

## 代码与数据目录

| 路径 | 内容 | Git 管理 |
|---|---|---|
| `scripts/track_ego_apriltag.py` | 四相机联合位姿估计与跟踪 | 是 |
| `example/python_demo.py` | 图片检测与单相机位姿估计 | 是 |
| `example/generate_tag_examples.py` | 根据官方码表生成标签示例 | 是 |
| `test/test_ego_multicam.py` | 相机几何、融合与状态门限测试 | 是 |
| `tag_family_examples/` | 各家族 ID 0、1、2 及尺寸标注图 | 是 |
| `outputs/` | 本机生成的跟踪视频、CSV、NPZ、曲线和缓存 | 否 |
| `recordings/` | 可选的本地录制存放目录 | 否 |
| `build/`、`.venv/`、`__pycache__/` | 编译产物、虚拟环境和缓存 | 否 |

录制目录可以放在项目之外，通过命令行传入。源码不依赖某个用户的绝对数据路径。

## 安装

当前已在 Ubuntu、系统 Python 3.10、OpenCV 4.5.4 下验证。
使用系统 Python 时，可安装对应系统包：

```bash
sudo apt install build-essential cmake python3-dev python3-numpy \
  python3-opencv python3-scipy python3-yaml python3-matplotlib python3-pil \
  fonts-noto-cjk

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_PYTHON_WRAPPER=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build --parallel 4
```

`requirements-ego.txt` 也列出了 Python 依赖，方便在自己的虚拟环境中安装。
如果切换 Python 或 NumPy 环境，应使用目标环境重新编译 Python 绑定；运行脚本与编译环境保持一致。
示例生成脚本的中文图示使用 Ubuntu 的 Noto CJK 字体。

## 检测单张图片

```bash
python3 example/python_demo.py --output build/detections.jpg

python3 example/python_demo.py /path/to/image.jpg \
  --family tag36h11 --tagsize 0.04 \
  --intrinsics 800 800 320 240 \
  --output build/pose.jpg
```

上述内参仅为示例，须替换为对应相机、对应分辨率的标定值。
`tagsize` 为检测四角之间的一条物理边长，单位为米。
单图示例的 `--distortion` 参数用于 OpenCV 普通针孔畸变模型；
EGO KB 鱼眼录制应使用下面的四相机脚本。

## 跟踪 EGOViewer 录制

输入目录应包含 `meta_*.json`、四路视频与时间戳 CSV，以及 `calibration_camera.yaml`。
脚本的第一个路径是原始录制目录，`--output` 后的路径是结果目录。
例如录制 `recordings/EGO-4B_AZLCA610091_20261027_190651` 可以输出到
`outputs/EGO-4B_AZLCA610091_20261027_190651_0.03`；`_0.03` 是结果目录的自定义后缀，
30 mm 的实际边长由 `--tagsize 0.030` 设置。输出目录不能替代原始录制目录作为输入。

```bash
python3 scripts/track_ego_apriltag.py /path/to/EGO_recording \
  --family tag36h11 --tag-id 0 --tagsize 0.04 \
  --output outputs/my_recording
```

多个同尺寸标签可以在一次处理中同时跟踪，例如三个 40 mm 的 `tag36h11`：

```bash
python3 scripts/track_ego_apriltag.py \
  recordings/EGO-4B_AZLCA610091_20261027_182506 \
  --family tag36h11 --tag-ids 0 1 2 --tagsize 0.040 \
  --output outputs/ego_20261027_182506
```

`--tag-ids` 与单码参数 `--tag-id` 二选一。每帧共享全图解码，每个 ID 独立维护角点、
光流、位姿和滤波状态；四相机仅融合相同 ID 的观测。身份由码字决定，转动后不会按画面左右位置重新编号。
标签不可见时，该 ID 独立标记为 `lost`，其余标签继续跟踪。`--tagsize 0.040` 表示每个码的检测边长为 40 mm。

多码输出目录中，`tag_0/`、`tag_1/`、`tag_2/` 分别保存与单码相同的位姿、统计、轨迹图和检测缓存。
每个子目录里的 `trajectory.png` 和 `orientation.png` 是该 ID 的单独图。
根目录的同名 `trajectory.png`、`orientation.png` 是统一位姿总览，每个分量只画一条平滑曲线。
统一参考为 `--tag-ids` 中的第一个 ID，本例为 ID 0 的中心和坐标轴；位置表达在 cam_0 坐标系，
朝向相对统一位姿的第一帧有效朝向。脚本从同时获得有效解码位姿的帧中稳健估计码间固定变换，
优先使用多相机观测并通过共同可见的 ID 连接各码，再把可用位姿换算到统一参考系后融合、统一滤波。
ID 0 不可见时，可以由 ID 1 或 2 估计同一个 ID 0 参考坐标系的位姿。
这按三个码相对固定来估计；腕带变形会影响统一位姿精度，码间关系应有足够的共同观测支持。
根目录 `tracking_status.png` 也是一条整体跟踪状态曲线；
`tracking_status_by_id.png` 另外保留各 ID 与 `GLOBAL` 的状态对照。
根目录的 `tracking_preview.mp4` 同时标注各 ID 和坐标轴，`summary.json` 汇总各码质量与覆盖率；
视频单独显示 `GLOBAL: TRACKING/LOST`：至少一个码通过现有重投影误差、运动突跳等位姿检查时，整体有效；
仅当所有码都丢失时，整体才是 `lost`。存在解码位姿时整体为 `measured`，否则有有效光流位姿时为 `flow_only`。
根目录 `poses.csv` 每帧每码一行，增加 `tag_id` 和 `global_status` 列；`status` 保留该码自己的状态。
根目录 `global_tracking.csv` 每帧一行，记录整体状态、有效码数量和 `active_tag_ids`；
`summary.json` 的 `valid_frames`、`lost_frames`、`coverage_percent` 也按上述整体规则统计。
`global_poses.csv` 和 `global_poses.npz` 保存每帧一份的统一位姿，包含实际参与融合的来源 ID；
`global_poses.npz` 中 `T_cam0_global_raw`、`T_cam0_global_filtered` 的形状均为 `(帧数, 4, 4)`。
`tag_alignment.json` 保存参考 ID、估计的码间变换、共同观测数量和对齐残差。
根目录 `poses.npz` 的变换数组形状为 `(帧数, 标签数, 4, 4)`，`status` 为 `(帧数, 标签数)`，
`global_status` 和 `n_valid_tags` 为 `(帧数,)`；第二维的顺序由 `tag_ids` 数组指定：

```python
import numpy as np

result = np.load("outputs/ego_20261027_182506/poses.npz")
slot = np.flatnonzero(result["tag_ids"] == 2)[0]
T_cam0_tag2 = result["T_cam0_tag_raw"][:, slot]
overall_valid = result["global_status"] != "lost"

global_result = np.load("outputs/ego_20261027_182506/global_poses.npz")
T_cam0_global = global_result["T_cam0_global_filtered"]
```

上述图片在处理录制时自动生成。已有检测缓存时，只想重新生成图表和统计，可运行：

```bash
python3 scripts/track_ego_apriltag.py \
  recordings/EGO-4B_AZLCA610091_20261027_182506 \
  --family tag36h11 --tag-ids 0 1 2 --tagsize 0.040 \
  --output outputs/ego_20261027_182506 --reuse-detections --no-video
```

## 离线恢复短暂丢帧

已有多码跟踪结果时，可以从前后清晰帧重新解码、双向跟踪，恢复快速运动时单向跟踪漏掉的帧：

```bash
python3 scripts/recover_ego_apriltag.py \
  outputs/EGO-4B_AZLCA610091_20261027_190651_0.03
```

这个恢复脚本的输入是**结果目录**，它从 `summary.json` 读取原始录制路径；原始跟踪命令的输入仍为录制目录。
默认输出到同级的 `EGO-4B_AZLCA610091_20261027_190651_0.03_recovered/`，保留原始结果。
恢复时重检原始丢帧和孤立单码位置尖峰附近的短窗口，保留同一 ID 的多个候选，使用已估计的腕带布局和多相机重投影选择目标，
从两端清晰图像做光流跟踪，每个相邻图像对保留正反向光流检查；仅光流支持时至少需要两路相机一致。
快速转头时，不仅依赖直线位姿预测来判断已有观测是否正确。

剩余缺口默认最多补 3 帧，且两端有效位姿间隔不超过 0.15 秒，采用位置线性插值和旋转 SLERP。
长缺口、序列首尾以及运动跳变过大的缺口不补；`--max-gap-frames 0` 可禁用插值。
有图像支撑的两端之间，少数孤立、未能再次从图像确认的大幅跳变也会作为短缺口处理。
`global_poses.csv` 和 `.npz` 保留 `original_status`、`pose_source`，区分：

- `redetected`：当前帧重新解码并通过几何检查。
- `flow_recovered`：前后图像跟踪并通过多相机几何检查，`status=flow_only`。
- `interpolated`：两端约束下的估计，`status=interpolated`；没有当前帧视觉测量。
- `original` / `original_unverified`：沿用原始结果，后者表示恢复窗口内未重新确认的观测。

恢复目录的轨迹和朝向图以橙色点标出插值帧，状态图单列 `ESTIMATED`，
`summary.json` 分别统计图像支撑、插值和仍然丢失的帧。需要纯视觉结果时只选
`status` 为 `measured` 或 `flow_only` 的行。恢复仍依赖码间相对固定的假设，不能把长时遮挡或完全模糊的图像恢复成真实测量。
图像中的背景同 ID 标签也会造成误跟踪，录制时宜遮住不属于目标腕带的重复标签。

恢复完成后，可以直接用保存的位姿生成视频，无需重新检测：

```bash
python3 scripts/render_ego_recovery.py \
  outputs/EGO-4B_AZLCA610091_20261027_190651_0.03_recovered
```

输出四路拼接的 `tracking_preview.mp4` 和 `camera_previews/cam_0.mp4` 至 `cam_3.mp4`，
均为 H.264 视频，需要系统安装带 libx264 的 FFmpeg。坐标轴由当前帧的整体位姿和固定码布局投影，
画面中的 `image IDs` 单列当前提供图像支持的 ID，插值帧显示 `ESTIMATED` 和橙色边框。

## 跟踪选项

常用选项：

- `--max-frames 120`：先处理前 120 帧检查数据。
- `--reuse-detections`：在相同录制和检测设置下复用保存的角点，重新做位姿拟合。
- `--no-video`：不生成预览视频。
- `--flow-max-frames 15`：最多 15 帧没有重新解码时继续角点跟踪，30 fps 时约 0.5 秒。
- `--flow-max-frames 0`：关闭光流续跟踪。
- `--cutoff 4`：平滑滤波的最低截止频率，单位 Hz。

旧版跟踪缓存的设置校验版本不同，更新代码后首次运行请不带 `--reuse-detections`，重新生成缓存。

## 四路如何融合

1. 通过设备时间戳做最近邻关联，默认最大差 1 ms。
2. 每路解码目标家族和 ID；必要时做局部鱼眼展开后再解码。
3. 对短暂漏解码的角点做 LK 光流跟踪，检查前后向误差与四边形形状。
4. 按 KB 模型将像素恢复为单位视线方向，通过已知外参生成三角测量与单目位姿初值。
5. 用标签四角的已知实际尺寸，在各路原始鱼眼画面中联合最小化重投影残差，求一个 R、t。
6. 使用稳健损失、异常相机剔除和位姿跳变检查，再对位置与 SO(3) 旋转做自适应因果平滑。

KB 模型为 `theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9`。
脚本按具名字段读取内参；四相机外参的平移从毫米换算为米。
外参方向为 `p_cam_i = R_i @ p_cam0 + t_i`。

仅由光流支持的位姿至少需要两路相机一致；当前帧有解码支持时，允许单路退化工作。
未通过检查的帧保持丢失，输出 NaN。

## 坐标和结果

相机坐标：X 向右、Y 向下、Z 向前。标签坐标原点在标签中心。
按标签标准图案正向观察，标签 X 向右、Y 向下、Z 垂直指向标签背面，符合右手系。
标签旋转后，坐标轴随解码出的图案方向旋转，并不固定沿视频画面的右、下方向。
预览中红色为 X、绿色为 Y、蓝色为 Z；叠加使用当前帧的 raw 位姿，避免因果平滑的延迟使坐标轴偏离标签。

```python
import numpy as np

result = np.load("outputs/my_recording/poses.npz")
T = result["T_cam0_tag_filtered"][0]
R = T[:3, :3]
t = T[:3, 3]  # 米
# p_cam0 = R @ p_tag + t
```

- `poses.csv`：时间戳、状态、使用相机数、重投影误差、raw/filtered 位置和四元数。
- `poses.npz`：逐帧四阶变换矩阵与状态。
- `tracking_preview.mp4`：四路角点和当前帧 raw 三维坐标轴叠加视频，`xyz(m)` 也使用 raw 位置。
- `camera_previews/`：各路独立预览。
- `trajectory.png`、`orientation.png`：位置与朝向曲线。
- `summary.json`：覆盖率、观测来源与质量统计。

四元数顺序为 x、y、z、w，并保持相邻符号连续。
`measured` 表示至少一路当前帧解码成功；`flow_only` 表示当前帧由多路光流支持；
`lost` 表示未输出可靠位姿。`raw` 是原始联合估计，`filtered` 是平滑后的联合估计。

这是离线 Python 原型。平滑存在运动相关延迟；重投影误差反映内部一致性，不能直接当作三维绝对精度。
当前结果相对头环 cam_0，未融合 IMU 建立世界坐标轨迹。

## 验证

```bash
python3 -m unittest discover -s test -p test_ego_multicam.py -v
```

测试覆盖 KB 投影与反解、相机外参和三角测量尺度、异常相机剔除、时间戳关联、光流位姿门限，
以及多码共享检测、ID 隔离、缓存复用、多码位姿导出、整体任一码有效的状态规则，
以及码间变换、跨 ID 连续的统一参考位姿、异常位姿剔除和统一位姿导出。

## 同步官方更新

个人仓库使用 `origin`，官方仓库使用 `upstream`。
日常开发提交并推送到 `origin`；同步官方更新时先 `git fetch upstream`，再在干净工作区合并所需分支。
保留官方提交历史和 [BSD 2-Clause 许可证](../LICENSE.md)。
