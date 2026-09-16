"""
VideoFusion - Streamlit web UI for merge_videos.py

Upload clips, set the merge order, and download a single Full HD
(1920x1080) H.264/AAC video. Reuses the exact same FFmpeg pipeline as
the command-line tool (merge_videos.py).
"""

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from merge_videos import (
    TARGET_WIDTH,
    TARGET_HEIGHT,
    MergeError,
    find_binary,
    natural_sort_key,
    probe_video,
    choose_target_fps,
    normalize_video,
    write_concat_list,
    concat_videos,
)

SUPPORTED_TYPES = ["mp4", "mov", "mkv", "avi", "m4v", "webm", "flv", "wmv", "ts", "mts", "m2ts"]

st.set_page_config(page_title="VideoFusion", page_icon="🎬", layout="centered")
st.title("🎬 VideoFusion")
st.caption("Merge multiple clips into one Full HD (1920x1080) H.264/AAC video.")

uploaded_files = st.file_uploader(
    "Upload video files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload two or more videos to get started.")
    st.stop()

if len(uploaded_files) < 2:
    st.warning("Add at least one more video — need 2 or more to merge.")
    st.stop()

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

st.subheader("Merge order")
st.caption("Edit the Order column to control the sequence (lower number = earlier in the video).")
edited_df = st.data_editor(
    order_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Order": st.column_config.NumberColumn(min_value=1, step=1),
        "File": st.column_config.TextColumn(disabled=True),
    },
    key="order_editor",
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

merge_clicked = st.button("Merge videos", type="primary")

if not merge_clicked:
    st.stop()

ordered_positions = edited_df.sort_values("Order", kind="stable").index.tolist()
ordered_files = [uploaded_files[i] for i in ordered_positions]

ffmpeg_path = find_binary("ffmpeg")
ffprobe_path = find_binary("ffprobe")
if not ffmpeg_path or not ffprobe_path:
    st.error(
        "FFmpeg is not available in this environment. "
        "If you're deploying to Streamlit Community Cloud, make sure a "
        "`packages.txt` file with `ffmpeg` is present in the repo root."
    )
    st.stop()

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

        video_bytes = output_path.read_bytes()

    except MergeError as e:
        status.update(label="Merge failed", state="error")
        st.error(str(e))
        st.stop()

st.success(f"Merged {len(input_paths)} videos into {output_name}.")
st.video(video_bytes)
st.download_button(
    "Download merged video",
    data=video_bytes,
    file_name=output_name,
    mime="video/mp4",
    type="primary",
)
