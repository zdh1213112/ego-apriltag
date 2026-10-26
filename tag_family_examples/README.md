# 标签家族示例与尺寸说明

本目录覆盖当前仓库的 9 个 AprilTag 家族和 17 个 ArUco 字典。
图案由当前仓库的 `apriltag_to_image()` 根据原始码表生成。每个家族的示例 ID 见下表。
这里展示所有家族的样例，并未导出每个家族的全部 ID。ArUco 是本仓库兼容的另一套标记字典。

## 先看这几张图

- [AprilTag 家族总览](01_apriltag_overview.png)
- [ArUco 字典总览](02_aruco_overview.png)
- [常见家族的检测边界对比](03_tagsize_comparison.png)

## 每个家族文件夹里有什么

- `id_0000.png`、`id_0001.png` 等：纯黑白标签，可用于图片检测或打印。
- `tagsize_id_0000.png`：标出检测正方形、四角和 tagsize 的学习图。
- 根目录 `families.json`：码字数量、编码位数和图像检测边界等元数据。

## family 与 ID

`family` 表示标签家族；`ID` 表示该家族里的一个具体标签。不同家族都可以有 ID 0。
Python 检测时填写文件夹名，例如 `apriltag('tag36h11')`。

## tagsize 怎么量

`tagsize` 是两个相邻检测角点之间的实际长度，单位为米。红色正方形的一条边就是测量位置。
`tag36h11` 的边界在黑色外框外缘；Standard、Circle、Custom 的边界可能位于整个图案内部。
不能统一把整张 PNG 的宽度当作 tagsize。额外白边不计入测量长度。
这些 PNG 没有指定固定的物理打印尺寸。打印后量红框对应边界，80 mm 就填 `0.080`。
每格的打印大小一致时：tagsize = 下表的检测边长格数 × 每格的实际边长。

## 家族清单（来自本地码表）

| 家族 | 类别 | 编码位数 | 可用 ID 数量 | 检测边长 / 格 | 已导出 ID | 尺寸图 |
|---|---|---:|---:|---:|---|---|
| [tag36h10](apriltag/tag36h10/id_0000.png) | apriltag | 36 | 2320 | 8 | 0, 1, 2 | [查看](apriltag/tag36h10/tagsize_id_0000.png) |
| [tag36h11](apriltag/tag36h11/id_0000.png) | apriltag | 36 | 587 | 8 | 0, 1, 2 | [查看](apriltag/tag36h11/tagsize_id_0000.png) |
| [tag25h9](apriltag/tag25h9/id_0000.png) | apriltag | 25 | 35 | 7 | 0, 1, 2 | [查看](apriltag/tag25h9/tagsize_id_0000.png) |
| [tag16h5](apriltag/tag16h5/id_0000.png) | apriltag | 16 | 30 | 6 | 0, 1, 2 | [查看](apriltag/tag16h5/tagsize_id_0000.png) |
| [tagCircle21h7](apriltag/tagCircle21h7/id_0000.png) | apriltag | 21 | 38 | 5 | 0, 1, 2 | [查看](apriltag/tagCircle21h7/tagsize_id_0000.png) |
| [tagCircle49h12](apriltag/tagCircle49h12/id_0000.png) | apriltag | 49 | 65535 | 5 | 0, 1, 2 | [查看](apriltag/tagCircle49h12/tagsize_id_0000.png) |
| [tagStandard41h12](apriltag/tagStandard41h12/id_0000.png) | apriltag | 41 | 2115 | 5 | 0, 1, 2 | [查看](apriltag/tagStandard41h12/tagsize_id_0000.png) |
| [tagStandard52h13](apriltag/tagStandard52h13/id_0000.png) | apriltag | 52 | 48714 | 6 | 0, 1, 2 | [查看](apriltag/tagStandard52h13/tagsize_id_0000.png) |
| [tagCustom48h12](apriltag/tagCustom48h12/id_0000.png) | apriltag | 48 | 42211 | 6 | 0, 1, 2 | [查看](apriltag/tagCustom48h12/tagsize_id_0000.png) |
| [tagAruco4x4_50](aruco/tagAruco4x4_50/id_0000.png) | aruco | 16 | 50 | 6 | 0, 1, 2 | [查看](aruco/tagAruco4x4_50/tagsize_id_0000.png) |
| [tagAruco4x4_100](aruco/tagAruco4x4_100/id_0000.png) | aruco | 16 | 100 | 6 | 0, 1, 2 | [查看](aruco/tagAruco4x4_100/tagsize_id_0000.png) |
| [tagAruco4x4_250](aruco/tagAruco4x4_250/id_0000.png) | aruco | 16 | 250 | 6 | 0, 1, 2 | [查看](aruco/tagAruco4x4_250/tagsize_id_0000.png) |
| [tagAruco4x4_1000](aruco/tagAruco4x4_1000/id_0000.png) | aruco | 16 | 1000 | 6 | 0, 1, 2 | [查看](aruco/tagAruco4x4_1000/tagsize_id_0000.png) |
| [tagAruco5x5_50](aruco/tagAruco5x5_50/id_0000.png) | aruco | 25 | 50 | 7 | 0, 1, 2 | [查看](aruco/tagAruco5x5_50/tagsize_id_0000.png) |
| [tagAruco5x5_100](aruco/tagAruco5x5_100/id_0000.png) | aruco | 25 | 100 | 7 | 0, 1, 2 | [查看](aruco/tagAruco5x5_100/tagsize_id_0000.png) |
| [tagAruco5x5_250](aruco/tagAruco5x5_250/id_0000.png) | aruco | 25 | 250 | 7 | 0, 1, 2 | [查看](aruco/tagAruco5x5_250/tagsize_id_0000.png) |
| [tagAruco5x5_1000](aruco/tagAruco5x5_1000/id_0000.png) | aruco | 25 | 1000 | 7 | 0, 1, 2 | [查看](aruco/tagAruco5x5_1000/tagsize_id_0000.png) |
| [tagAruco6x6_50](aruco/tagAruco6x6_50/id_0000.png) | aruco | 36 | 50 | 8 | 0, 1, 2 | [查看](aruco/tagAruco6x6_50/tagsize_id_0000.png) |
| [tagAruco6x6_100](aruco/tagAruco6x6_100/id_0000.png) | aruco | 36 | 100 | 8 | 0, 1, 2 | [查看](aruco/tagAruco6x6_100/tagsize_id_0000.png) |
| [tagAruco6x6_250](aruco/tagAruco6x6_250/id_0000.png) | aruco | 36 | 250 | 8 | 0, 1, 2 | [查看](aruco/tagAruco6x6_250/tagsize_id_0000.png) |
| [tagAruco6x6_1000](aruco/tagAruco6x6_1000/id_0000.png) | aruco | 36 | 1000 | 8 | 0, 1, 2 | [查看](aruco/tagAruco6x6_1000/tagsize_id_0000.png) |
| [tagAruco7x7_50](aruco/tagAruco7x7_50/id_0000.png) | aruco | 49 | 50 | 9 | 0, 1, 2 | [查看](aruco/tagAruco7x7_50/tagsize_id_0000.png) |
| [tagAruco7x7_100](aruco/tagAruco7x7_100/id_0000.png) | aruco | 49 | 100 | 9 | 0, 1, 2 | [查看](aruco/tagAruco7x7_100/tagsize_id_0000.png) |
| [tagAruco7x7_250](aruco/tagAruco7x7_250/id_0000.png) | aruco | 49 | 250 | 9 | 0, 1, 2 | [查看](aruco/tagAruco7x7_250/tagsize_id_0000.png) |
| [tagAruco7x7_1000](aruco/tagAruco7x7_1000/id_0000.png) | aruco | 49 | 1000 | 9 | 0, 1, 2 | [查看](aruco/tagAruco7x7_1000/tagsize_id_0000.png) |
| [tagArucoMIP36h12](aruco/tagArucoMIP36h12/id_0000.png) | aruco | 36 | 250 | 8 | 0, 1, 2 | [查看](aruco/tagArucoMIP36h12/tagsize_id_0000.png) |

## 运行 Python 检测

在项目根目录执行：

```bash
python3 example/python_demo.py tag_family_examples/apriltag/tag36h10/id_0000.png --family tag36h10
```

## 重新生成或增加 ID

```bash
python3 example/generate_tag_examples.py
python3 example/generate_tag_examples.py --family tag36h11 --count 10 --output /tmp/tag36h11_examples
```

生成脚本使用本仓库 `build/libapriltag.so`，需要 NumPy 和 Pillow。
图案来自本仓库码表，许可信息见上级目录 LICENSE.md。
