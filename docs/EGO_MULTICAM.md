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

```bash
python3 scripts/track_ego_apriltag.py /path/to/EGO_recording \
  --family tag36h11 --tag-id 0 --tagsize 0.04 \
  --output outputs/my_recording
```

常用选项：

- `--max-frames 120`：先处理前 120 帧检查数据。
- `--reuse-detections`：在相同录制和检测设置下复用保存的角点，重新做位姿拟合。
- `--no-video`：不生成预览视频。
- `--flow-max-frames 15`：最多 15 帧没有重新解码时继续角点跟踪，30 fps 时约 0.5 秒。
- `--flow-max-frames 0`：关闭光流续跟踪。
- `--cutoff 4`：平滑滤波的最低截止频率，单位 Hz。

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
- `tracking_preview.mp4`：四路角点和三维坐标轴叠加视频。
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

测试覆盖 KB 投影与反解、相机外参和三角测量尺度、异常相机剔除、时间戳关联，以及光流位姿门限。

## 同步官方更新

个人仓库使用 `origin`，官方仓库使用 `upstream`。
日常开发提交并推送到 `origin`；同步官方更新时先 `git fetch upstream`，再在干净工作区合并所需分支。
保留官方提交历史和 [BSD 2-Clause 许可证](../LICENSE.md)。
