#!/usr/bin/env python3
"""Export tag family examples using the repository's native tag renderer."""

import argparse
import ctypes as ct
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
RED = "#d62828"
INK = "#172d45"
NEAREST = getattr(Image, "Resampling", Image).NEAREST


class Family(ct.Structure):
    _fields_ = [
        ("ncodes", ct.c_uint32), ("codes", ct.POINTER(ct.c_uint64)),
        ("width_at_border", ct.c_int), ("total_width", ct.c_int),
        ("reversed_border", ct.c_bool), ("nbits", ct.c_uint32),
        ("bit_x", ct.POINTER(ct.c_uint32)),
        ("bit_y", ct.POINTER(ct.c_uint32)), ("h", ct.c_uint32),
        ("name", ct.c_char_p), ("impl", ct.c_void_p),
    ]


class ImageU8(ct.Structure):
    _fields_ = [
        ("width", ct.c_int32), ("height", ct.c_int32),
        ("stride", ct.c_int32), ("buf", ct.POINTER(ct.c_uint8)),
    ]


def font(size, mono=False):
    return ImageFont.truetype(MONO if mono else FONT, size)


def centered(draw, x, y, text, typeface, fill=INK):
    bounds = draw.textbbox((0, 0), text, font=typeface)
    draw.text((x - (bounds[2] - bounds[0]) / 2, y), text,
              font=typeface, fill=fill)


def known_families():
    source = (ROOT / "apriltag_pywrap.c").read_text()
    macro = source.split("#define SUPPORTED_TAG_FAMILIES(_)", 1)[1]
    macro = macro.split("#define TAG_CREATE_FAMILY", 1)[0]
    return re.findall(r"_\((tag\w+)\)", macro)


def render(lib, family, tag_id, cell_pixels):
    pointer = lib.apriltag_to_image(family, tag_id)
    if not pointer:
        raise RuntimeError(f"Could not render tag ID {tag_id}")
    try:
        image = pointer.contents
        array = np.ctypeslib.as_array(
            image.buf, shape=(image.height * image.stride,),
        ).reshape(image.height, image.stride)[:, :image.width].copy()
    finally:
        lib.image_u8_destroy(pointer)
    # Two extra white cells protect the pattern and provide a quiet margin.
    array = np.pad(array, 2, constant_values=255)
    return Image.fromarray(array).resize(
        (array.shape[1] * cell_pixels, array.shape[0] * cell_pixels),
        NEAREST,
    )


def mark_boundary(draw, bounds, width=4):
    left, top, right, bottom = bounds
    draw.rectangle(bounds, outline=RED, width=width)
    for x, y in ((left, top), (right, top), (right, bottom), (left, bottom)):
        draw.line((x - 7, y - 7, x + 7, y + 7), fill=RED, width=width)
        draw.line((x - 7, y + 7, x + 7, y - 7), fill=RED, width=width)


def size_illustration(record, image, output):
    canvas = Image.new("RGB", (1000, image.height + 410), "white")
    draw = ImageDraw.Draw(canvas)
    centered(draw, 500, 20, record["family"] + " / ID 0", font(30, True))
    centered(draw, 500, 74, "标签尺寸测量示意", font(30))
    x, y = (1000 - image.width) // 2, 145
    canvas.paste(image, (x, y))
    bounds = record["detection_bounds_px"]
    left, top, right, bottom = (
        x + bounds[0], y + bounds[1], x + bounds[2], y + bounds[3],
    )
    mark_boundary(draw, (left, top, right, bottom))
    arrow_y = y + image.height + 25
    for edge in (left, right):
        draw.line((edge, bottom, edge, arrow_y + 10), fill=RED, width=2)
    draw.line((left, arrow_y, right, arrow_y), fill=RED, width=3)
    draw.polygon(((left, arrow_y), (left + 16, arrow_y - 7),
                  (left + 16, arrow_y + 7)), fill=RED)
    draw.polygon(((right, arrow_y), (right - 16, arrow_y - 7),
                  (right - 16, arrow_y + 7)), fill=RED)
    centered(draw, 500, arrow_y + 15,
             f"tagsize = {record['width_at_border']} 个单元格 × 每格实际边长",
             font(26), RED)
    centered(draw, 500, arrow_y + 72,
             "红色方框：检测边界；红色 ×：四个检测角点。", font(25))
    centered(draw, 500, arrow_y + 116,
             "tagsize 使用打印后的实测长度，单位为米。", font(24))
    centered(draw, 500, arrow_y + 158,
             "此图用于学习；检测或打印请使用 id_0000.png 等纯黑白图。", font(23))
    canvas.save(output)


def overview(records, title, output, annotate=False, columns=3):
    card_w, card_h = 370, 380
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * card_w, 140 + rows * card_h), "#eff3f8")
    draw = ImageDraw.Draw(canvas)
    centered(draw, canvas.width / 2, 18, title, font(34))
    subtitle = "红色边界的边长 = tagsize" if annotate else "每个家族展示 ID 0；其他 ID 见对应文件夹"
    centered(draw, canvas.width / 2, 72, subtitle, font(21))
    for index, record in enumerate(records):
        x, y = (index % columns) * card_w, 125 + (index // columns) * card_h
        draw.rounded_rectangle((x + 10, y, x + card_w - 10, y + card_h - 15),
                               radius=12, fill="white")
        centered(draw, x + card_w / 2, y + 12, record["family"], font(20, True))
        centered(draw, x + card_w / 2, y + 48,
                 f"{record['nbits']} bits · {record['ncodes']:,} 个 ID", font(18))
        # Use an integer pixel count per cell, preserving the exact binary layout.
        with Image.open(record["first_image"]) as full:
            modules = record["total_width"] + 4
            side = modules * (220 // modules)
            image = full.resize((side, side), NEAREST)
        left, top = x + (card_w - side) // 2, y + 88
        canvas.paste(image, (left, top))
        if annotate:
            factor = side / record["image_width_px"]
            bounds = record["detection_bounds_px"]
            mark_boundary(draw, (left + bounds[0] * factor,
                                 top + bounds[1] * factor,
                                 left + bounds[2] * factor,
                                 top + bounds[3] * factor), width=2)
        centered(draw, x + card_w / 2, y + 320,
                 f"检测边长：{record['width_at_border']} 格", font(20),
                 RED if annotate else INK)
    canvas.save(output)


def write_guide(output, records):
    apriltags = sum(row["group"] == "apriltag" for row in records)
    arucos = len(records) - apriltags
    lines = [
        "# 标签家族示例与尺寸说明", "",
        f"本目录覆盖当前仓库的 {apriltags} 个 AprilTag 家族和 {arucos} 个 ArUco 字典。",
        "图案由当前仓库的 `apriltag_to_image()` 根据原始码表生成。每个家族的示例 ID 见下表。",
        "这里展示所有家族的样例，并未导出每个家族的全部 ID。ArUco 是本仓库兼容的另一套标记字典。", "",
        "## 先看这几张图", "",
    ]
    for filename, label in (("01_apriltag_overview.png", "AprilTag 家族总览"),
                            ("02_aruco_overview.png", "ArUco 字典总览"),
                            ("03_tagsize_comparison.png", "常见家族的检测边界对比")):
        if (output / filename).exists():
            lines.append(f"- [{label}]({filename})")
    lines += [
        "", "## 每个家族文件夹里有什么", "",
        "- `id_0000.png`、`id_0001.png` 等：纯黑白标签，可用于图片检测或打印。",
        "- `tagsize_id_0000.png`：标出检测正方形、四角和 tagsize 的学习图。",
        "- 根目录 `families.json`：码字数量、编码位数和图像检测边界等元数据。", "",
        "## family 与 ID", "",
        "`family` 表示标签家族；`ID` 表示该家族里的一个具体标签。不同家族都可以有 ID 0。",
        "Python 检测时填写文件夹名，例如 `apriltag('tag36h11')`。", "",
        "## tagsize 怎么量", "",
        "`tagsize` 是两个相邻检测角点之间的实际长度，单位为米。红色正方形的一条边就是测量位置。",
        "`tag36h11` 的边界在黑色外框外缘；Standard、Circle、Custom 的边界可能位于整个图案内部。",
        "不能统一把整张 PNG 的宽度当作 tagsize。额外白边不计入测量长度。",
        "这些 PNG 没有指定固定的物理打印尺寸。打印后量红框对应边界，80 mm 就填 `0.080`。",
        "每格的打印大小一致时：tagsize = 下表的检测边长格数 × 每格的实际边长。", "",
        "## 家族清单（来自本地码表）", "",
        "| 家族 | 类别 | 编码位数 | 可用 ID 数量 | 检测边长 / 格 | 已导出 ID | 尺寸图 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in records:
        folder = f"{row['group']}/{row['family']}"
        ids = ", ".join(map(str, row["ids"]))
        lines.append(f"| [{row['family']}]({folder}/id_0000.png) | "
                     f"{row['group']} | {row['nbits']} | {row['ncodes']} | "
                     f"{row['width_at_border']} | {ids} | "
                     f"[查看]({folder}/tagsize_id_0000.png) |")
    sample = records[0]
    lines += [
        "", "## 运行 Python 检测", "", "在项目根目录执行：", "", "```bash",
        f"python3 example/python_demo.py tag_family_examples/{sample['group']}/{sample['family']}/id_0000.png --family {sample['family']}",
        "```", "", "## 重新生成或增加 ID", "", "```bash",
        "python3 example/generate_tag_examples.py",
        "python3 example/generate_tag_examples.py --family tag36h11 --count 10 --output /tmp/tag36h11_examples",
        "```", "",
        "生成脚本使用本仓库 `build/libapriltag.so`，需要 NumPy 和 Pillow。",
        "图案来自本仓库码表，许可信息见上级目录 LICENSE.md。", "",
    ]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tag_family_examples")
    parser.add_argument("--count", type=int, default=3, help="IDs per family, starting at 0")
    parser.add_argument("--family", nargs="+", choices=known_families())
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    library = ROOT / "build/libapriltag.so"
    if not library.exists():
        parser.error("Build the shared library in build/ first")
    lib = ct.CDLL(str(library))
    lib.apriltag_to_image.argtypes = [ct.POINTER(Family), ct.c_uint32]
    lib.apriltag_to_image.restype = ct.POINTER(ImageU8)
    lib.image_u8_destroy.argtypes = [ct.POINTER(ImageU8)]
    lib.image_u8_destroy.restype = None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.family or known_families():
        create, destroy = getattr(lib, name + "_create"), getattr(lib, name + "_destroy")
        create.argtypes, create.restype = [], ct.POINTER(Family)
        destroy.argtypes, destroy.restype = [ct.POINTER(Family)], None
        pointer = create()
        if not pointer:
            raise RuntimeError(f"Could not create family {name}")
        try:
            family = pointer.contents
            group = "aruco" if name.startswith("tagAruco") else "apriltag"
            folder = output / group / name
            folder.mkdir(parents=True, exist_ok=True)
            count, scale = min(args.count, family.ncodes), 40
            offset = (2 + (family.total_width - family.width_at_border) // 2) * scale
            edge = family.width_at_border * scale
            record = {
                "family": name, "group": group, "ncodes": family.ncodes,
                "nbits": family.nbits, "minimum_hamming_distance": family.h,
                "width_at_border": family.width_at_border,
                "total_width": family.total_width,
                "reversed_border": family.reversed_border,
                "ids": list(range(count)), "cell_pixels": scale,
                "extra_white_margin_cells": 2,
                "image_width_px": (family.total_width + 4) * scale,
                "detection_bounds_px": [offset, offset, offset + edge, offset + edge],
                "first_image": str(folder / "id_0000.png"),
            }
            for tag_id in record["ids"]:
                image = render(lib, pointer, tag_id, scale)
                image.save(folder / f"id_{tag_id:04d}.png")
                if tag_id == 0:
                    size_illustration(record, image, folder / "tagsize_id_0000.png")
            records.append(record)
            print(f"{name}: exported IDs 0..{count - 1} (of {family.ncodes})", flush=True)
        finally:
            destroy(pointer)
    native = [row for row in records if row["group"] == "apriltag"]
    aruco = [row for row in records if row["group"] == "aruco"]
    if native:
        overview(native, "AprilTag 家族总览", output / "01_apriltag_overview.png")
    if aruco:
        overview(aruco, "ArUco 字典总览（本仓库支持）",
                 output / "02_aruco_overview.png", columns=4)
    names = ("tag36h11", "tagStandard41h12", "tagCircle21h7", "tagCustom48h12")
    examples = [row for name in names for row in records if row["family"] == name]
    if examples:
        overview(examples, "tagsize 测量边界对比", output / "03_tagsize_comparison.png",
                 annotate=True, columns=2)
    manifest = []
    for record in records:
        row = dict(record)
        row["first_image"] = str(Path(row["first_image"]).relative_to(output))
        manifest.append(row)
    (output / "families.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_guide(output, records)
    print(f"Saved {len(records)} families in {output}")


if __name__ == "__main__":
    sys.exit(main())
