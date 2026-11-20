#!/usr/bin/env python3
"""Run EGO multi-tag tracking, gap recovery and the final preview in one command."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]


def show_progress(log_path, stopped):
    """Forward frame progress while keeping detailed reports in the log."""
    pattern = re.compile(r"^(CUDA detector:|cam_\d \d+/\d+|pose \d+/\d+|Solving |Selecting rigid |ID \d+:|Re-detected |Re-decoded |Rendered |Saved )")
    with log_path.open(errors="replace") as stream:
        while True:
            for line in stream:
                if pattern.match(line):
                    print(line.rstrip(), flush=True)
            if stopped.wait(.2):
                for line in stream:
                    if pattern.match(line):
                        print(line.rstrip(), flush=True)
                return


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="一条命令完成多码跟踪、码间标定、丢帧恢复和预览视频。")
    parser.add_argument("recording", type=Path, help="原始 EGOViewer 录制目录")
    parser.add_argument("--tagsize", type=float, required=True, help="标签黑框边长，单位为米；20 mm 填 0.020")
    parser.add_argument("--reference", type=int, help="参考标签 ID；未指定时使用 --tag-ids 的第一个 ID")
    parser.add_argument("--tag-ids", type=int, nargs="+", required=True, help="明确指定要跟踪的 ID，例如 0 1 2 3")
    parser.add_argument("--family", required=True, help="明确指定标签家族，例如 tag36h11")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="全图检测后端；cuda 加速检测，光流与位姿拟合仍在 CPU 上执行")
    parser.add_argument("--output", type=Path, help="结果目录，默认自动保存到 outputs/")
    parser.add_argument("--max-frames", type=int, default=0, help="试跑帧数，默认 0 处理全部")
    parser.add_argument("--reuse-detections", action="store_true", help="复用相同录制和检测设置的已有角点缓存")
    parser.add_argument("--trail-seconds", type=float, default=0., help="参考 ID 的轨迹拖影时长，单位秒；0 关闭")
    parser.add_argument("--flow-max-frames", type=int, default=15, help="未解码时最多续跟多少帧；0 关闭主跟踪光流")
    parser.add_argument("--cutoff", type=float, default=4., help="自适应滤波最低截止频率，单位 Hz")
    parser.add_argument("--no-filter", action="store_true", help="跳过位姿滤波，仍导出原始位姿")
    parser.add_argument("--max-reprojection", type=float, default=2.5, help="主跟踪和恢复的重投影误差门限，单位像素")
    parser.add_argument("--max-sync-us", type=int, default=1000, help="相机间最大匹配时间差，单位微秒")
    parser.add_argument("--alignment-max-translation", type=float, default=.01, help="码间位置一致性门限，单位米")
    parser.add_argument("--alignment-max-rotation", type=float, default=10., help="码间朝向一致性门限，单位度")
    parser.add_argument("--recovery", choices=("on", "off"), default="on", help="是否执行离线重检和双向光流恢复")
    parser.add_argument("--patch-recovery", action="store_true", help="从缺口附近已解码的标签启动局部图像块跟踪")
    parser.add_argument("--pattern-recovery", action="store_true", help="在鱼眼原图中重新读取 tag36h11 码格，至少两路相机验证")
    interpolation = parser.add_mutually_exclusive_group()
    interpolation.add_argument("--max-gap-frames", type=int, default=3, help="恢复阶段最多插值补多少帧；0 不插值")
    interpolation.add_argument("--no-interpolation", action="store_true", help="等同 --max-gap-frames 0")
    args = parser.parse_args(argv)
    if args.patch_recovery and args.recovery == "off":
        parser.error("--patch-recovery 需要 --recovery on")
    if args.pattern_recovery and (args.family != "tag36h11" or args.recovery == "off"):
        parser.error("--pattern-recovery 需要 --family tag36h11 和 --recovery on")
    if args.device == "cuda" and args.family != "tag36h11":
        parser.error("当前 CUDA 后端仅支持 --family tag36h11；其他家族使用 --device cpu")
    if not math.isfinite(args.tagsize) or args.tagsize <= 0:
        parser.error("--tagsize 必须是正数，单位为米")
    if not math.isfinite(args.trail_seconds) or args.trail_seconds < 0:
        parser.error("--trail-seconds 必须是非负有限秒数")
    if len(args.tag_ids) < 2 or len(set(args.tag_ids)) != len(args.tag_ids) or min(args.tag_ids) < 0:
        parser.error("--tag-ids 需要至少两个不同的非负 ID；单码请使用 track_ego_apriltag.py")
    if min(args.max_frames, args.flow_max_frames, args.max_sync_us, args.max_gap_frames) < 0:
        parser.error("帧数和同步时间差不能为负数")
    if any(not math.isfinite(value) or value <= 0 for value in (
            args.cutoff, args.max_reprojection, args.alignment_max_translation, args.alignment_max_rotation)):
        parser.error("滤波频率和误差门限必须是有限正数")
    if args.no_interpolation:
        args.max_gap_frames = 0
    args.reference = args.tag_ids[0] if args.reference is None else args.reference
    if args.reference not in args.tag_ids:
        parser.error("参考 ID 必须在跟踪的标签中")
    args.tag_ids = [args.reference] + [tag_id for tag_id in args.tag_ids if tag_id != args.reference]
    args.recording = args.recording.resolve()
    if not args.recording.is_dir() or len(list(args.recording.glob("meta_*.json"))) != 1:
        parser.error("请传入包含一个 meta_*.json 的原始录制目录")
    args.output = (args.output or ROOT / "outputs" / (
        f"{args.recording.name}_{args.tagsize * 1000:g}mm_ref{args.reference}"
        + ("_cuda" if args.device == "cuda" else "")
        + ("_patch" if args.patch_recovery else "")
        + ("_pattern" if args.pattern_recovery else "")
        + ("_no_filter" if args.no_filter else ""))).resolve()
    if args.output == args.recording or args.output.is_file():
        parser.error("结果目录必须与原始录制目录分开，且不能是文件")
    if args.recovery == "on" and not shutil.which("ffmpeg"):
        parser.error("生成预览需要 FFmpeg，请先完成 README 中的安装步骤")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output
    recovery_enabled = args.recovery == "on"
    tracking = output / "tracking" if recovery_enabled else output
    track_command = [
        "track_ego_apriltag.py", str(args.recording), "--family", args.family, "--device", args.device,
        "--tag-ids", *map(str, args.tag_ids), "--tagsize", str(args.tagsize),
        "--max-frames", str(args.max_frames), "--flow-max-frames", str(args.flow_max_frames),
        "--cutoff", str(args.cutoff), "--max-reprojection", str(args.max_reprojection),
        "--max-sync-us", str(args.max_sync_us),
        "--alignment-max-translation", str(args.alignment_max_translation),
        "--alignment-max-rotation", str(args.alignment_max_rotation),
        "--trail-seconds", str(args.trail_seconds), "--output", str(tracking)]
    if recovery_enabled:
        track_command.append("--no-video")
    if args.reuse_detections:
        track_command.append("--reuse-detections")
    if args.no_filter:
        track_command.append("--no-filter")
    commands = [("跟踪标签并计算码间外参", track_command)]
    if recovery_enabled:
        commands.extend([("恢复短暂丢帧", [
            "recover_ego_apriltag.py", str(tracking), "--output", str(output),
            "--max-gap-frames", str(args.max_gap_frames), "--cutoff", str(args.cutoff),
            "--device", args.device,
            *(["--no-filter"] if args.no_filter else []),
            *(["--patch-recovery"] if args.patch_recovery else []),
            *(["--pattern-recovery"] if args.pattern_recovery else []),
            *(["--max-sync-us", str(args.max_sync_us)] if args.patch_recovery or args.pattern_recovery else []),
            "--max-reprojection", str(args.max_reprojection)]),
            ("生成预览视频", ["render_ego_recovery.py", "--trail-seconds", str(args.trail_seconds), str(output)])])
    output.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output / "run_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    log_path = output / "processing.log"
    stopped = threading.Event()
    progress = None
    try:
        with log_path.open("w") as log:
            backend = ("CUDA 全图检测；CPU 码字解码、局部重检、光流与 KB 位姿拟合"
                       if args.device == "cuda" else "CPU（AprilTag / OpenCV / SciPy）")
            print(f"计算后端：{backend}", flush=True)
            progress = threading.Thread(target=show_progress, args=(log_path, stopped), daemon=True)
            progress.start()
            for index, (label, command) in enumerate(commands, 1):
                print(f"{index}/{len(commands)} {label}…", flush=True)
                log.write(f"\n{index}/{len(commands)} {label}\n")
                log.flush()
                subprocess.run([sys.executable, str(ROOT / "scripts" / command[0]), *command[1:]],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        detail = "\n".join(log_path.read_text(errors="replace").splitlines()[-8:])
        raise SystemExit(f"{label}未完成。\n{detail}\n详细日志：{log_path}") from None
    finally:
        stopped.set()
        if progress is not None:
            progress.join()
    summary = json.loads((output / "summary.json").read_text())
    alignment = json.loads((output / "tag_alignment.json").read_text())
    visual = summary["image_supported_frames"] if recovery_enabled else summary["valid_frames"]
    interpolated = summary["interpolated_frames"] if recovery_enabled else 0
    lost = summary["remaining_lost_frames"] if recovery_enabled else summary["lost_frames"]
    print(f"完成：视觉有效 {visual}/{summary['frames']} 帧，插值 {interpolated} 帧，丢失 {lost} 帧。")
    if alignment.get("unaligned_tag_ids"):
        print(f"未能标定的标签：{alignment['unaligned_tag_ids']}，未参与统一位姿。")
    print(f"视频：{output / 'tracking_preview.mp4'}")
    print(f"外参：{output / 'tag_alignment.json'}")
    print(f"位姿：{output / 'global_poses.csv'}")


if __name__ == "__main__":
    main()
