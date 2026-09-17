"""
VideoFusion - Streamlit web UI for merge_videos.py

A flexible video/audio utility: the user picks one independent operation
from a dropdown (Merge Videos / Add or Replace Audio / Merge Videos +
Audio) instead of being forced through a fixed multi-step flow. Reuses the
exact same FFmpeg pipeline as the command-line tool (merge_videos.py).
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

SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "mkv", "avi", "m4v", "webm", "flv", "wmv", "ts", "mts", "m2ts"]
SUPPORTED_AUDIO_TYPES = sorted(ext.lstrip(".") for ext in AUDIO_EXTENSIONS)

st.set_page_config(page_title="VideoFusion", page_icon="🎬", layout="centered")
st.title("🎬 VideoFusion")
st.caption("A simple video/audio utility — pick an operation and go.")


# ---------------------------------------------------------------------------
# Shared FFmpeg pipeline helpers (used by more than one operation)
# ---------------------------------------------------------------------------
def run_video_merge(ordered_files, crf, preset, output_name, ffmpeg_path, ffprobe_path, status) -> bytes:
    """Normalize + concatenate the given uploaded video files. Returns the merged video bytes."""
    with tempfile.TemporaryDirectory(prefix="videofusion_merge_") as tmp:
        tmp_dir = Path(tmp)
        input_paths = []
        for f in ordered_files:
            p = tmp_dir / f.name
            p.write_bytes(f.getbuffer())
            input_paths.append(p)

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

        normalized_paths = []
        for idx, info in enumerate(probed, start=1):
            dst = tmp_dir / f"norm_{idx:03d}.mp4"
            status.write(f"Normalizing {info['path'].name} ({idx}/{len(probed)})...")
            normalize_video(ffmpeg_path, info, dst, target_fps, crf, preset)
            normalized_paths.append(dst)

        list_file = tmp_dir / "concat_list.txt"
        write_concat_list(normalized_paths, list_file)

        output_path = tmp_dir / output_name
        status.write("Merging normalized clips...")
        concat_videos(ffmpeg_path, list_file, output_path)

        return output_path.read_bytes()


def run_audio_merge(video_bytes, audio_file, output_name, ffmpeg_path, ffprobe_path, status) -> bytes:
    """Replace a video's audio track with the given uploaded audio file. Returns the final video bytes."""
    with tempfile.TemporaryDirectory(prefix="videofusion_audio_") as tmp:
        tmp_dir = Path(tmp)

        video_path = tmp_dir / "input_video.mp4"
        video_path.write_bytes(video_bytes)

        audio_suffix = Path(audio_file.name).suffix or ".audio"
        audio_path = tmp_dir / f"selected_audio{audio_suffix}"
        audio_path.write_bytes(audio_file.getbuffer())

        status.write("Validating audio file...")
        audio_info = probe_audio(ffprobe_path, audio_path)
        status.write(f"- {audio_file.name}: {audio_info['codec']}, {audio_info['duration']:.1f}s")

        status.write("Reading video duration...")
        video_duration = get_duration(ffprobe_path, video_path)

        output_path = tmp_dir / output_name
        status.write("Merging audio into video...")
        merge_audio_into_video(ffmpeg_path, video_path, video_duration, audio_path, output_path)

        return output_path.read_bytes()


def show_result(label: str, video_bytes: bytes, file_name: str, download_key: str):
    st.success(label)
    st.video(video_bytes)
    st.download_button(
        "Download video",
        data=video_bytes,
        file_name=file_name,
        mime="video/mp4",
        type="primary",
        key=download_key,
    )


# ---------------------------------------------------------------------------
# FFmpeg availability (required by every operation)
# ---------------------------------------------------------------------------
ffmpeg_path = find_binary("ffmpeg")
ffprobe_path = find_binary("ffprobe")
if not ffmpeg_path or not ffprobe_path:
    st.error(
        "FFmpeg is not available in this environment. "
        "If you're deploying to Streamlit Community Cloud, make sure a "
        "`packages.txt` file with `ffmpeg` is present in the repo root."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Operation picker
# ---------------------------------------------------------------------------
st.subheader("Choose an operation")
operation = st.selectbox(
    "Operation",
    ["Merge Videos", "Add / Replace Audio", "Merge Videos + Audio"],
    label_visibility="collapsed",
)
st.divider()


# ---------------------------------------------------------------------------
# Operation: Merge Videos
# ---------------------------------------------------------------------------
if operation == "Merge Videos":
    for key, default in {"mv_uploader_key": 0, "mv_result_bytes": None, "mv_result_name": None}.items():
        st.session_state.setdefault(key, default)

    st.markdown("### Select video files")
    mv_files = st.file_uploader(
        "Upload videos",
        type=SUPPORTED_VIDEO_TYPES,
        accept_multiple_files=True,
        key=f"mv_uploader_{st.session_state.mv_uploader_key}",
        label_visibility="collapsed",
    )

    if mv_files:
        st.write("Selected files:")
        for f in mv_files:
            st.write(f"✓ {f.name}")

        default_order = sorted(
            range(len(mv_files)), key=lambda i: natural_sort_key(Path(mv_files[i].name))
        )
        order_lookup = {file_index: position + 1 for position, file_index in enumerate(default_order)}
        order_df = pd.DataFrame(
            {
                "Order": [order_lookup[i] for i in range(len(mv_files))],
                "File": [f.name for f in mv_files],
            }
        )
        st.caption("Edit the Order column to control the sequence (lower number = earlier in the video).")
        mv_edited_df = st.data_editor(
            order_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Order": st.column_config.NumberColumn(min_value=1, step=1),
                "File": st.column_config.TextColumn(disabled=True),
            },
            key=f"mv_order_editor_{st.session_state.mv_uploader_key}",
        )

        with st.expander("Advanced settings"):
            mv_crf = st.slider("Quality (CRF) — lower is better quality, larger file", 14, 28, 18, key="mv_crf")
            mv_preset = st.selectbox(
                "Encoding preset (slower = better compression at same quality)",
                ["slow", "medium", "fast", "veryfast"],
                index=0,
                key="mv_preset",
            )
            mv_output_name = st.text_input("Output filename", "merged_output.mp4", key="mv_output_name").strip() or "merged_output.mp4"
            if not mv_output_name.lower().endswith(".mp4"):
                mv_output_name += ".mp4"

        if st.button("Merge Videos", type="primary", key="mv_merge_btn"):
            if len(mv_files) < 2:
                st.error("Please select at least 2 videos to merge.")
            else:
                ordered_positions = mv_edited_df.sort_values("Order", kind="stable").index.tolist()
                ordered_files = [mv_files[i] for i in ordered_positions]
                status = st.status("Merging videos...", expanded=True)
                try:
                    result_bytes = run_video_merge(
                        ordered_files, mv_crf, mv_preset, mv_output_name, ffmpeg_path, ffprobe_path, status
                    )
                    status.update(label="Merge complete", state="complete")
                    st.session_state.mv_result_bytes = result_bytes
                    st.session_state.mv_result_name = mv_output_name
                except MergeError as e:
                    status.update(label="Merge failed", state="error")
                    st.error(f"Video merge failed: {e}")
                except Exception as e:  # noqa: BLE001 - friendly message instead of a raw traceback
                    status.update(label="Merge failed", state="error")
                    st.error(f"Unexpected error while merging videos: {e}")
    else:
        st.info("Upload one or more videos to get started.")

    if st.session_state.mv_result_bytes:
        show_result("✅ Videos merged successfully", st.session_state.mv_result_bytes, st.session_state.mv_result_name, "mv_download")
        if st.button("Clear", key="mv_clear_btn"):
            st.session_state.mv_uploader_key += 1
            st.session_state.mv_result_bytes = None
            st.session_state.mv_result_name = None
            st.rerun()


# ---------------------------------------------------------------------------
# Operation: Add / Replace Audio
# ---------------------------------------------------------------------------
elif operation == "Add / Replace Audio":
    for key, default in {
        "aa_video_uploader_key": 0, "aa_audio_uploader_key": 0,
        "aa_result_bytes": None, "aa_result_name": None,
    }.items():
        st.session_state.setdefault(key, default)

    st.markdown("### Select video")
    aa_video = st.file_uploader(
        "Upload a video",
        type=SUPPORTED_VIDEO_TYPES,
        accept_multiple_files=False,
        key=f"aa_video_uploader_{st.session_state.aa_video_uploader_key}",
        label_visibility="collapsed",
    )
    if aa_video:
        st.write(f"✓ {aa_video.name}")

    st.markdown("### Select audio")
    aa_audio = st.file_uploader(
        "Upload an audio file",
        type=SUPPORTED_AUDIO_TYPES,
        accept_multiple_files=False,
        key=f"aa_audio_uploader_{st.session_state.aa_audio_uploader_key}",
        label_visibility="collapsed",
    )
    if aa_audio:
        st.write(f"✓ {aa_audio.name}")

    st.caption(
        "The video's visuals and length are never changed. If the audio is shorter than the video, "
        "it's padded with silence; if it's longer, it's trimmed to the video's length."
    )

    aa_output_name = st.text_input(
        "Output filename",
        (Path(aa_video.name).stem + "_with_audio.mp4") if aa_video else "video_with_audio.mp4",
        key="aa_output_name",
    ).strip() or "video_with_audio.mp4"
    if not aa_output_name.lower().endswith(".mp4"):
        aa_output_name += ".mp4"

    if st.button("Add / Replace Audio", type="primary", key="aa_merge_btn"):
        if not aa_video:
            st.error("Please select a video.")
        elif not aa_audio:
            st.error("Please select an audio file.")
        else:
            status = st.status("Merging audio into video...", expanded=True)
            try:
                result_bytes = run_audio_merge(
                    aa_video.getvalue(), aa_audio, aa_output_name, ffmpeg_path, ffprobe_path, status
                )
                status.update(label="Audio merge complete", state="complete")
                st.session_state.aa_result_bytes = result_bytes
                st.session_state.aa_result_name = aa_output_name
            except MergeError as e:
                status.update(label="Audio merge failed", state="error")
                st.error(f"Audio merge failed: {e}")
            except Exception as e:  # noqa: BLE001 - friendly message instead of a raw traceback
                status.update(label="Audio merge failed", state="error")
                st.error(f"Unexpected error while merging audio: {e}")

    if st.session_state.aa_result_bytes:
        show_result("✅ Audio added successfully", st.session_state.aa_result_bytes, st.session_state.aa_result_name, "aa_download")
        if st.button("Clear", key="aa_clear_btn"):
            st.session_state.aa_video_uploader_key += 1
            st.session_state.aa_audio_uploader_key += 1
            st.session_state.aa_result_bytes = None
            st.session_state.aa_result_name = None
            st.rerun()


# ---------------------------------------------------------------------------
# Operation: Merge Videos + Audio
# ---------------------------------------------------------------------------
elif operation == "Merge Videos + Audio":
    for key, default in {
        "mva_video_uploader_key": 0, "mva_audio_uploader_key": 0,
        "mva_result_bytes": None, "mva_result_name": None,
    }.items():
        st.session_state.setdefault(key, default)

    st.markdown("### Select video files")
    mva_files = st.file_uploader(
        "Upload videos",
        type=SUPPORTED_VIDEO_TYPES,
        accept_multiple_files=True,
        key=f"mva_video_uploader_{st.session_state.mva_video_uploader_key}",
        label_visibility="collapsed",
    )

    mva_edited_df = None
    if mva_files:
        st.write("Selected files:")
        for f in mva_files:
            st.write(f"✓ {f.name}")

        default_order = sorted(
            range(len(mva_files)), key=lambda i: natural_sort_key(Path(mva_files[i].name))
        )
        order_lookup = {file_index: position + 1 for position, file_index in enumerate(default_order)}
        order_df = pd.DataFrame(
            {
                "Order": [order_lookup[i] for i in range(len(mva_files))],
                "File": [f.name for f in mva_files],
            }
        )
        st.caption("Edit the Order column to control the sequence (lower number = earlier in the video).")
        mva_edited_df = st.data_editor(
            order_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Order": st.column_config.NumberColumn(min_value=1, step=1),
                "File": st.column_config.TextColumn(disabled=True),
            },
            key=f"mva_order_editor_{st.session_state.mva_video_uploader_key}",
        )
    else:
        st.info("Upload two or more videos to get started.")

    st.markdown("### Select audio")
    mva_audio = st.file_uploader(
        "Upload an audio file",
        type=SUPPORTED_AUDIO_TYPES,
        accept_multiple_files=False,
        key=f"mva_audio_uploader_{st.session_state.mva_audio_uploader_key}",
        label_visibility="collapsed",
    )
    if mva_audio:
        st.write(f"✓ {mva_audio.name}")

    st.caption(
        "Videos are merged first, keeping their combined length unchanged. If the audio is shorter, "
        "it's padded with silence; if it's longer, it's trimmed to match."
    )

    with st.expander("Advanced settings"):
        mva_crf = st.slider("Quality (CRF) — lower is better quality, larger file", 14, 28, 18, key="mva_crf")
        mva_preset = st.selectbox(
            "Encoding preset (slower = better compression at same quality)",
            ["slow", "medium", "fast", "veryfast"],
            index=0,
            key="mva_preset",
        )
        mva_output_name = st.text_input("Output filename", "final_output.mp4", key="mva_output_name").strip() or "final_output.mp4"
        if not mva_output_name.lower().endswith(".mp4"):
            mva_output_name += ".mp4"

    if st.button("Merge Videos + Audio", type="primary", key="mva_merge_btn"):
        if not mva_files or len(mva_files) < 2:
            st.error("Please select at least 2 videos to merge.")
        elif not mva_audio:
            st.error("Please select an audio file.")
        else:
            ordered_positions = mva_edited_df.sort_values("Order", kind="stable").index.tolist()
            ordered_files = [mva_files[i] for i in ordered_positions]
            status = st.status("Merging videos and audio...", expanded=True)
            try:
                merged_bytes = run_video_merge(
                    ordered_files, mva_crf, mva_preset, "merged_intermediate.mp4", ffmpeg_path, ffprobe_path, status
                )
                result_bytes = run_audio_merge(
                    merged_bytes, mva_audio, mva_output_name, ffmpeg_path, ffprobe_path, status
                )
                status.update(label="Merge complete", state="complete")
                st.session_state.mva_result_bytes = result_bytes
                st.session_state.mva_result_name = mva_output_name
            except MergeError as e:
                status.update(label="Merge failed", state="error")
                st.error(f"Merge failed: {e}")
            except Exception as e:  # noqa: BLE001 - friendly message instead of a raw traceback
                status.update(label="Merge failed", state="error")
                st.error(f"Unexpected error while merging: {e}")

    if st.session_state.mva_result_bytes:
        show_result("✅ Videos merged and audio added successfully", st.session_state.mva_result_bytes, st.session_state.mva_result_name, "mva_download")
        if st.button("Clear", key="mva_clear_btn"):
            st.session_state.mva_video_uploader_key += 1
            st.session_state.mva_audio_uploader_key += 1
            st.session_state.mva_result_bytes = None
            st.session_state.mva_result_name = None
            st.rerun()
