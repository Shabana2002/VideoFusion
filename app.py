"""
VideoFusion - Streamlit web UI for merge_videos.py

Upload clips, set the merge order, and download a single Full HD
(1920x1080) H.264/AAC video. Optionally add/replace the audio track on
the merged video. Reuses the exact same FFmpeg pipeline as the
command-line tool (merge_videos.py).
"""

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from merge_videos import (
    AUDIO_EXTENSIONS,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    MergeError,
    find_binary,
    natural_sort_key,
    probe_video,
    probe_audio,
    choose_target_fps,
    normalize_video,
    write_concat_list,
    concat_videos,
    get_duration,
    merge_audio_into_video,
)

SUPPORTED_TYPES = ["mp4", "mov", "mkv", "avi", "m4v", "webm", "flv", "wmv", "ts", "mts", "m2ts"]
SUPPORTED_AUDIO_TYPES = sorted(ext.lstrip(".") for ext in AUDIO_EXTENSIONS)

st.set_page_config(page_title="VideoFusion", page_icon="🎬", layout="centered")
st.title("🎬 VideoFusion")
st.caption("Select Videos → Merge Videos → Select Audio → Merge Audio → Final Video")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "video_uploader_key": 0,
    "audio_uploader_key": 0,
    "merged_video_bytes": None,
    "merged_output_name": None,
    "final_video_bytes": None,
    "final_output_name": None,
}
for _key, _value in _DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


def clear_selected_videos():
    """Reset the app's current selection/state so a new merge can begin.

    This only resets in-memory app state and the upload widgets. It never
    touches any files on the user's computer - not the original source
    videos, the selected audio file, or any video already downloaded.
    """
    st.session_state.video_uploader_key += 1
    st.session_state.audio_uploader_key += 1
    st.session_state.merged_video_bytes = None
    st.session_state.merged_output_name = None
    st.session_state.final_video_bytes = None
    st.session_state.final_output_name = None


# ---------------------------------------------------------------------------
# Step 1 — Select Videos
# ---------------------------------------------------------------------------
st.header("Step 1 — Select Videos")
uploaded_files = st.file_uploader(
    "Upload video files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    key=f"video_uploader_{st.session_state.video_uploader_key}",
)

if not uploaded_files:
    st.info("Upload two or more videos to get started.")
    st.stop()

if len(uploaded_files) < 2:
    st.warning("Add at least one more video — need 2 or more to merge.")
    st.stop()

# ---------------------------------------------------------------------------
# Step 2 — Merge Videos
# ---------------------------------------------------------------------------
st.header("Step 2 — Merge Videos")

default_order = sorted(
    range(len(uploaded_files)),
    key=lambda i: natural_sort_key(Path(uploaded_files[i].name)),
)
order_lookup = {file_index: position + 1 for position, file_index in enumerate(default_order)}

order_df = pd.DataFrame(
    {
        "Order": [order_lookup[i] for i in range(len(uploaded_files))],
        "File": [f.name for f in uploaded_files],
    }
)

st.caption("Edit the Order column to control the sequence (lower number = earlier in the video).")
edited_df = st.data_editor(
    order_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Order": st.column_config.NumberColumn(min_value=1, step=1),
        "File": st.column_config.TextColumn(disabled=True),
    },
    key=f"order_editor_{st.session_state.video_uploader_key}",
)

with st.expander("Advanced settings"):
    crf = st.slider("Quality (CRF) — lower is better quality, larger file", 14, 28, 18)
    preset = st.selectbox(
        "Encoding preset (slower = better compression at same quality)",
        ["slow", "medium", "fast", "veryfast"],
        index=0,
    )
    output_name = st.text_input("Output filename", "merged_output.mp4").strip() or "merged_output.mp4"
    if not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"

ffmpeg_path = find_binary("ffmpeg")
ffprobe_path = find_binary("ffprobe")
if not ffmpeg_path or not ffprobe_path:
    st.error(
        "FFmpeg is not available in this environment. "
        "If you're deploying to Streamlit Community Cloud, make sure a "
        "`packages.txt` file with `ffmpeg` is present in the repo root."
    )
    st.stop()

merge_clicked = st.button("Merge videos", type="primary")

if merge_clicked:
    ordered_positions = edited_df.sort_values("Order", kind="stable").index.tolist()
    ordered_files = [uploaded_files[i] for i in ordered_positions]

    with tempfile.TemporaryDirectory(prefix="videofusion_") as tmp:
        tmp_dir = Path(tmp)
        input_paths = []
        for f in ordered_files:
            p = tmp_dir / f.name
            p.write_bytes(f.getbuffer())
            input_paths.append(p)

        status = st.status("Starting merge...", expanded=True)
        try:
            status.write("Inspecting videos...")
            probed = [probe_video(ffprobe_path, p) for p in input_paths]
            for info in probed:
                orientation = "portrait" if info["display_height"] > info["display_width"] else "landscape"
                audio_note = "with audio" if info["has_audio"] else "no audio"
                status.write(
                    f"- {info['path'].name}: {info['display_width']}x{info['display_height']} "
                    f"({orientation}), {info['fps']:.2f} fps, {audio_note}"
                )

            target_fps = choose_target_fps(probed)
            status.write(f"Target output: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {target_fps:.2f} fps, H.264 + AAC")

            progress_bar = st.progress(0.0)
            normalized_paths = []
            for idx, info in enumerate(probed, start=1):
                dst = tmp_dir / f"norm_{idx:03d}.mp4"
                status.write(f"Normalizing {info['path'].name} ({idx}/{len(probed)})...")
                normalize_video(ffmpeg_path, info, dst, target_fps, crf, preset)
                normalized_paths.append(dst)
                progress_bar.progress(idx / len(probed))

            list_file = tmp_dir / "concat_list.txt"
            write_concat_list(normalized_paths, list_file)

            output_path = tmp_dir / output_name
            status.write("Merging normalized clips...")
            concat_videos(ffmpeg_path, list_file, output_path)

            status.update(label="Merge complete", state="complete")

            st.session_state.merged_video_bytes = output_path.read_bytes()
            st.session_state.merged_output_name = output_name
            # A fresh video merge invalidates any previously-generated final video.
            st.session_state.final_video_bytes = None
            st.session_state.final_output_name = None

        except MergeError as e:
            status.update(label="Merge failed", state="error")
            st.error(f"Video merge failed: {e}")
        except Exception as e:  # noqa: BLE001 - surface a friendly message, not a raw traceback
            status.update(label="Merge failed", state="error")
            st.error(f"Unexpected error while merging videos: {e}")

# ---------------------------------------------------------------------------
# Merged video result + Step 3 — Add Audio to Merged Video
# ---------------------------------------------------------------------------
if st.session_state.merged_video_bytes:
    st.success(f"Merged {len(uploaded_files)} videos into {st.session_state.merged_output_name}.")
    st.video(st.session_state.merged_video_bytes)
    st.download_button(
        "Download merged video",
        data=st.session_state.merged_video_bytes,
        file_name=st.session_state.merged_output_name,
        mime="video/mp4",
        type="primary",
        key="download_merged",
    )

    st.header("Step 3 — Add Audio to Merged Video")
    audio_file = st.file_uploader(
        "Select an audio file",
        type=SUPPORTED_AUDIO_TYPES,
        accept_multiple_files=False,
        key=f"audio_uploader_{st.session_state.audio_uploader_key}",
    )

    audio_mode = st.radio(
        "Audio handling",
        ["Replace the video's audio with this track", "Keep the merged video's existing audio (no change)"],
        index=0,
    )

    st.caption(
        "If the audio is shorter than the video, it's padded with silence to fill the rest. "
        "If it's longer, it's trimmed to the video's length. The video's visuals and length are never changed."
    )

    final_name = st.text_input(
        "Final output filename",
        Path(st.session_state.merged_output_name).stem + "_with_audio.mp4",
    ).strip() or "final_output.mp4"
    if not final_name.lower().endswith(".mp4"):
        final_name += ".mp4"

    merge_audio_clicked = st.button("Merge Audio", type="primary")

    if merge_audio_clicked:
        if audio_mode.startswith("Keep"):
            st.session_state.final_video_bytes = st.session_state.merged_video_bytes
            st.session_state.final_output_name = st.session_state.merged_output_name
            st.success("Kept the merged video's existing audio — no audio merge needed.")
        elif audio_file is None:
            st.error("Please select an audio file before merging.")
        else:
            with tempfile.TemporaryDirectory(prefix="videofusion_audio_") as tmp:
                tmp_dir = Path(tmp)
                video_path = tmp_dir / "merged_input.mp4"
                video_path.write_bytes(st.session_state.merged_video_bytes)

                audio_suffix = Path(audio_file.name).suffix or ".audio"
                audio_path = tmp_dir / f"selected_audio{audio_suffix}"
                audio_path.write_bytes(audio_file.getbuffer())

                status = st.status("Starting audio merge...", expanded=True)
                try:
                    status.write("Validating audio file...")
                    audio_info = probe_audio(ffprobe_path, audio_path)
                    status.write(f"- {audio_file.name}: {audio_info['codec']}, {audio_info['duration']:.1f}s")

                    status.write("Reading merged video duration...")
                    video_duration = get_duration(ffprobe_path, video_path)

                    output_path = tmp_dir / final_name
                    status.write("Merging audio into video...")
                    merge_audio_into_video(ffmpeg_path, video_path, video_duration, audio_path, output_path)

                    status.update(label="Audio merge complete", state="complete")

                    st.session_state.final_video_bytes = output_path.read_bytes()
                    st.session_state.final_output_name = final_name

                except MergeError as e:
                    status.update(label="Audio merge failed", state="error")
                    st.error(f"Audio merge failed: {e}")
                except Exception as e:  # noqa: BLE001 - surface a friendly message, not a raw traceback
                    status.update(label="Audio merge failed", state="error")
                    st.error(f"Unexpected error while merging audio: {e}")

# ---------------------------------------------------------------------------
# Step 4 — Final Video
# ---------------------------------------------------------------------------
if st.session_state.final_video_bytes:
    st.header("Step 4 — Final Video")
    st.success(f"Final video ready: {st.session_state.final_output_name}")
    st.video(st.session_state.final_video_bytes)
    st.download_button(
        "Download final video",
        data=st.session_state.final_video_bytes,
        file_name=st.session_state.final_output_name,
        mime="video/mp4",
        type="primary",
        key="download_final",
    )

# ---------------------------------------------------------------------------
# Step 5 — Start Over
# ---------------------------------------------------------------------------
if st.session_state.merged_video_bytes:
    st.header("Step 5 — Start Over")
    st.caption(
        "Clears the videos, audio, and results currently loaded in this app so you can start a new "
        "project. This does not delete any files on your computer - not the source videos, the "
        "selected audio, or anything you've already downloaded."
    )
    st.button("Clear Selected Videos", on_click=clear_selected_videos)
