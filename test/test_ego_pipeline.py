"""Checks for the single-command user workflow and failure propagation."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_ego_apriltag import main, parse_args


class PipelineTests(unittest.TestCase):
    def test_cuda_default_output_preserves_cpu_results(self):
        with TemporaryDirectory() as temporary:
            recording = Path(temporary) / "recording"
            recording.mkdir()
            (recording / "meta_test.json").write_text("{}")
            common = [str(recording), "--family", "tag36h11", "--tag-ids", "0", "1", "2", "3",
                      "--tagsize", ".020", "--reference", "0", "--recovery", "off"]
            cpu = parse_args([*common, "--device", "cpu"])
            cuda = parse_args([*common, "--device", "cuda"])
            self.assertEqual(cuda.output.name, cpu.output.name + "_cuda")
            self.assertNotEqual(cpu.output, cuda.output)
            unfiltered = parse_args([*common, "--device", "cuda", "--no-filter"])
            self.assertEqual(unfiltered.output.name, cuda.output.name + "_no_filter")

    def test_complete_workflow_preserves_reference_paths_and_interpolation_choice(self):
        for visual_only in (False, True):
            with self.subTest(visual_only=visual_only), TemporaryDirectory() as temporary:
                recording = Path(temporary) / "recording with spaces"
                recording.mkdir()
                (recording / "meta_test.json").write_text("{}")
                output = Path(temporary) / "results with spaces"
                calls = []

                def run(command, **kwargs):
                    calls.append(command)
                    script = Path(command[1]).name
                    if script == "recover_ego_apriltag.py":
                        (output / "summary.json").write_text(json.dumps({
                            "frames": 10, "image_supported_frames": 8,
                            "interpolated_frames": 0 if visual_only else 1,
                            "remaining_lost_frames": 2 if visual_only else 1}))
                        (output / "tag_alignment.json").write_text(json.dumps({"unaligned_tag_ids": []}))
                    elif script == "render_ego_recovery.py":
                        self.assertTrue((output / "summary.json").exists())
                        (output / "tracking_preview.mp4").touch()
                    self.assertTrue(kwargs["check"])
                    self.assertEqual(kwargs["stderr"], subprocess.STDOUT)

                args = [str(recording), "--family", "tag36h11", "--tag-ids", "0", "1", "2", "3",
                        "--tagsize", ".020", "--reference", "1", "--output", str(output),
                        "--flow-max-frames", "7", "--cutoff", "2", "--max-reprojection", "1.25",
                        "--max-sync-us", "250", "--alignment-max-translation", ".005",
                        "--alignment-max-rotation", "5", "--recovery", "on", "--trail-seconds", "1.5"]
                args.extend(["--device", "cuda" if visual_only else "cpu"])
                if visual_only:
                    args.extend(["--no-interpolation", "--reuse-detections", "--no-filter"])
                else:
                    args.extend(["--max-gap-frames", "3"])
                console = io.StringIO()
                with patch("run_ego_apriltag.shutil.which", return_value="/usr/bin/ffmpeg"), \
                        patch("run_ego_apriltag.subprocess.run", side_effect=run), redirect_stdout(console):
                    main(args)
                self.assertEqual([Path(command[1]).name for command in calls], [
                    "track_ego_apriltag.py", "recover_ego_apriltag.py", "render_ego_recovery.py"])
                start = calls[0].index("--tag-ids") + 1
                self.assertEqual(calls[0][start:start + 4], ["1", "0", "2", "3"])
                self.assertEqual(calls[0][calls[0].index("--family") + 1], "tag36h11")
                for command in calls[:2]:
                    self.assertEqual(command[command.index("--device") + 1], "cuda" if visual_only else "cpu")
                self.assertEqual(calls[0][calls[0].index("--output") + 1], str(output / "tracking"))
                self.assertEqual(float(calls[0][calls[0].index("--trail-seconds") + 1]), 1.5)
                self.assertIn("--no-video", calls[0])
                self.assertEqual("--reuse-detections" in calls[0], visual_only)
                for command in calls[:2]:
                    self.assertEqual("--no-filter" in command, visual_only)
                for flag, value in (("--flow-max-frames", 7), ("--cutoff", 2), ("--max-reprojection", 1.25),
                                    ("--max-sync-us", 250), ("--alignment-max-translation", .005),
                                    ("--alignment-max-rotation", 5)):
                    self.assertEqual(float(calls[0][calls[0].index(flag) + 1]), value)
                self.assertEqual(calls[1][calls[1].index("--max-gap-frames") + 1], "0" if visual_only else "3")
                self.assertEqual(float(calls[1][calls[1].index("--cutoff") + 1]), 2)
                self.assertEqual(float(calls[1][calls[1].index("--max-reprojection") + 1]), 1.25)
                self.assertEqual(float(calls[2][calls[2].index("--trail-seconds") + 1]), 1.5)
                self.assertEqual(calls[2][-1], str(output))
                self.assertTrue((output / "tracking_preview.mp4").exists())
                self.assertIn(str(output / "tracking_preview.mp4"), console.getvalue())
                settings = json.loads((output / "run_settings.json").read_text())
                self.assertEqual(settings["flow_max_frames"], 7)
                self.assertEqual(settings["max_gap_frames"], 0 if visual_only else 3)
                self.assertEqual(settings["trail_seconds"], 1.5)
                self.assertEqual(settings["no_filter"], visual_only)

    def test_recovery_can_be_disabled_without_losing_the_preview(self):
        with TemporaryDirectory() as temporary:
            recording = Path(temporary) / "recording"
            recording.mkdir()
            (recording / "meta_test.json").write_text("{}")
            output = Path(temporary) / "results"

            def run(command, **kwargs):
                self.assertEqual(Path(command[1]).name, "track_ego_apriltag.py")
                self.assertEqual(command[command.index("--output") + 1], str(output))
                self.assertNotIn("--no-video", command)
                self.assertEqual(float(command[command.index("--trail-seconds") + 1]), 1.5)
                (output / "summary.json").write_text(json.dumps({"frames": 10, "valid_frames": 9, "lost_frames": 1}))
                (output / "tag_alignment.json").write_text(json.dumps({"unaligned_tag_ids": []}))
                (output / "tracking_preview.mp4").touch()

            console = io.StringIO()
            with patch("run_ego_apriltag.shutil.which", return_value=None), \
                    patch("run_ego_apriltag.subprocess.run", side_effect=run) as execute, redirect_stdout(console):
                main([str(recording), "--family", "tag36h11", "--tag-ids", "0", "1", "2", "3",
                      "--tagsize", ".020", "--output", str(output), "--recovery", "off", "--max-gap-frames", "0",
                      "--trail-seconds", "1.5"])
            self.assertEqual(execute.call_count, 1)
            self.assertIn("1/1", console.getvalue())
            self.assertIn("视觉有效 9/10 帧，插值 0 帧，丢失 1 帧", console.getvalue())
            self.assertTrue((output / "tracking_preview.mp4").exists())

    def test_failed_tracking_stops_recovery_and_rendering(self):
        with TemporaryDirectory() as temporary:
            recording = Path(temporary) / "recording"
            recording.mkdir()
            (recording / "meta_test.json").write_text("{}")
            output = Path(temporary) / "results"
            with patch("run_ego_apriltag.shutil.which", return_value="/usr/bin/ffmpeg"), \
                    patch("run_ego_apriltag.subprocess.run", side_effect=subprocess.CalledProcessError(1, "tracking")) as run, \
                    redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(SystemExit, "跟踪标签并计算码间外参未完成"):
                main([str(recording), "--family", "tag36h11", "--tag-ids", "0", "1", "2", "3",
                      "--tagsize", ".020", "--output", str(output)])
            self.assertEqual(run.call_count, 1)
            self.assertFalse((output / "tracking_preview.mp4").exists())
