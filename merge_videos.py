#!/usr/bin/env python3
"""
merge_videos.py

Merges every video found in the "input" folder into a single Full HD
(1920x1080) video, in filename order, using FFmpeg.

Usage:
    python merge_videos.py

Put your clips in the "input" folder, name them so they sort in the
order you want (e.g. 01_intro.mp4, 02_main.mp4, 03_outro.mp4), and run
the script. The result is written to output/merged_output.mp4.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm",
    ".flv", ".wmv", ".ts", ".mts", ".m2ts",
}

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

INSTALL_INSTRUCTIONS = """
FFmpeg was not found on this system.

To install it on Windows, open PowerShell and run:

    winget install --id Gyan.FFmpeg -e

Then CLOSE and REOPEN your terminal (PowerShell/VS Code) so the
updated PATH takes effect, and run this script again.

If you don't have winget, you can install manually instead:
  1. Go to https://www.gyan.dev/ffmpeg/builds/ and download the
     "release full" build (a .zip file).
  2. Extract it, e.g. to C:\\ffmpeg
  3. Add C:\\ffmpeg\\bin to your PATH:
       Windows Settings -> System -> About -> Advanced system settings
       -> Environment Variables -> edit the "Path" variable (User or
       System) -> New -> C:\\ffmpeg\\bin -> OK
  4. Open a NEW terminal window and run "ffmpeg -version" to confirm.
"""


class MergeError(Exception):
    """Raised when a specific input video could not be processed."""


def find_binary(name: str) -> str | None:
    """Find an executable on PATH, or in a fresh winget install location."""
    found = shutil.which(name)
    if found:
        return found

    # winget installs to a versioned folder under LOCALAPPDATA that may not
    # be on PATH yet until the terminal is restarted. Look there too so the
    # script works right after installation without requiring a restart.
    packages_dir = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if packages_dir.exists():
        matches = sorted(packages_dir.glob(f"**/{name}.exe"))
        if matches:
            return str(matches[0])

    return None


def locate_ffmpeg() -> tuple[str, str]:
    ffmpeg_path = find_binary("ffmpeg")
    ffprobe_path = find_binary("ffprobe")

    if not ffmpeg_path or not ffprobe_path:
        print("ERROR: FFmpeg/FFprobe could not be found.")
        print(INSTALL_INSTRUCTIONS)
        sys.exit(1)

    return ffmpeg_path, ffprobe_path


def natural_sort_key(path: Path):
    """Sort '2' before '10' by splitting the filename into text/number chunks."""
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def discover_input_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        print(f"ERROR: Input folder not found: {input_dir}")
        sys.exit(1)

    videos = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if len(videos) < 2:
        print(f"ERROR: Found {len(videos)} video(s) in '{input_dir}'. "
              f"Add at least 2 videos to merge (supported: {', '.join(sorted(VIDEO_EXTENSIONS))}).")
        sys.exit(1)

    return sorted(videos, key=natural_sort_key)


def run_ffprobe(ffprobe_path: str, path: Path) -> dict:
    cmd = [
        ffprobe_path, "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise MergeError(f"Could not read '{path.name}' with ffprobe:\n{result.stderr.strip()}")
    return json.loads(result.stdout)


def parse_frame_rate(rate_str: str) -> float:
    try:
        return float(Fraction(rate_str))
    except (ValueError, ZeroDivisionError):
        return 30.0


def get_rotation(stream: dict) -> int:
    tags = stream.get("tags", {})
    if "rotate" in tags:
        try:
            return int(tags["rotate"]) % 360
        except ValueError:
            pass
    for sd in stream.get("side_data_list", []):
        if "rotation" in sd:
            try:
                return int(sd["rotation"]) % 360
            except (ValueError, TypeError):
                pass
    return 0


def probe_video(ffprobe_path: str, path: Path) -> dict:
    info = run_ffprobe(ffprobe_path, path)

    video_stream = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise MergeError(f"'{path.name}' has no video stream — is it a valid video file?")

    audio_stream = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    rotation = abs(get_rotation(video_stream))
    display_width, display_height = (height, width) if rotation in (90, 270) else (width, height)

    fps = parse_frame_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate", "30/1"))

    return {
        "path": path,
        "width": width,
        "height": height,
        "display_width": display_width,
        "display_height": display_height,
        "fps": fps,
        "codec": video_stream.get("codec_name", "?"),
        "pix_fmt": video_stream.get("pix_fmt", "?"),
        "has_audio": audio_stream is not None,
    }


def choose_target_fps(probed: list[dict]) -> float:
    fps_values = [p["fps"] for p in probed if p["fps"] > 0]
    return round(max(fps_values), 2) if fps_values else 30.0


def build_video_filters(info: dict, target_fps: float) -> str:
    filters = []

    needs_resize = not (
        info["display_width"] == TARGET_WIDTH and info["display_height"] == TARGET_HEIGHT
    )
    if needs_resize:
        filters.append(
            f"scale=w={TARGET_WIDTH}:h={TARGET_HEIGHT}:force_original_aspect_ratio=decrease"
        )
        filters.append(
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    filters.append("setsar=1")

    if abs(info["fps"] - target_fps) > 0.01:
        filters.append(f"fps={target_fps}")

    return ",".join(filters)


def normalize_video(
    ffmpeg_path: str,
    info: dict,
    dst: Path,
    target_fps: float,
    crf: int,
    preset: str,
) -> None:
    src = info["path"]
    vf = build_video_filters(info, target_fps)

    cmd = [ffmpeg_path, "-y", "-i", str(src)]

    if info["has_audio"]:
        cmd += ["-map", "0:v:0", "-map", "0:a:0"]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]

    cmd += [
        "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(dst),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        raise MergeError(f"FFmpeg failed while processing '{src.name}':\n{tail}")


def write_concat_list(paths: list[Path], list_file: Path) -> None:
    with open(list_file, "w", encoding="utf-8") as f:
        for p in paths:
            escaped = str(p.resolve()).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")


def concat_videos(ffmpeg_path: str, list_file: Path, output_path: Path) -> None:
    cmd = [
        ffmpeg_path, "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-15:])
        raise MergeError(f"FFmpeg failed while merging the normalized clips:\n{tail}")


def main():
    parser = argparse.ArgumentParser(description="Merge videos in a folder into one Full HD video.")
    parser.add_argument("--input-dir", default="input", help="Folder containing the source videos (default: input)")
    parser.add_argument("--output-dir", default="output", help="Folder to write the merged video to (default: output)")
    parser.add_argument("--output-name", default="merged_output.mp4", help="Output filename (default: merged_output.mp4)")
    parser.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better quality, default: 18)")
    parser.add_argument("--preset", default="slow", help="x264 encoding preset (default: slow)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_dir = (script_dir / args.input_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_path = output_dir / args.output_name

    print("Checking for FFmpeg...")
    ffmpeg_path, ffprobe_path = locate_ffmpeg()
    print(f"  Using ffmpeg:  {ffmpeg_path}")
    print(f"  Using ffprobe: {ffprobe_path}")

    videos = discover_input_videos(input_dir)
    print(f"\nFound {len(videos)} video(s) in '{input_dir}', in this order:")
    for v in videos:
        print(f"  - {v.name}")

    try:
        print("\nInspecting videos...")
        probed = [probe_video(ffprobe_path, v) for v in videos]
        for info in probed:
            orientation = "portrait" if info["display_height"] > info["display_width"] else "landscape"
            audio_note = "with audio" if info["has_audio"] else "no audio track"
            print(
                f"  - {info['path'].name}: {info['display_width']}x{info['display_height']} "
                f"({orientation}), {info['fps']:.2f} fps, {info['codec']}, {audio_note}"
            )

        target_fps = choose_target_fps(probed)
        print(f"\nTarget output: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {target_fps:.2f} fps, H.264 + AAC")

        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="merge_videos_") as tmp:
            tmp_dir = Path(tmp)
            normalized_paths = []

            print("\nNormalizing clips (resolution, aspect ratio, frame rate, codecs)...")
            for i, info in enumerate(probed, start=1):
                dst = tmp_dir / f"norm_{i:03d}.mp4"
                print(f"  [{i}/{len(probed)}] Processing '{info['path'].name}'...")
                normalize_video(ffmpeg_path, info, dst, target_fps, args.crf, args.preset)
                normalized_paths.append(dst)
            print("Normalization complete.")

            list_file = tmp_dir / "concat_list.txt"
            write_concat_list(normalized_paths, list_file)

            print(f"\nMerging {len(normalized_paths)} clips into '{output_path.name}'...")
            concat_videos(ffmpeg_path, list_file, output_path)

    except MergeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"\nDone! Merged video saved to:\n  {output_path}")


if __name__ == "__main__":
    main()
