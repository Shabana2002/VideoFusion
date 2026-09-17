"""
VideoFusion - Streamlit web UI for merge_videos.py

A polished, public-facing video/audio tool. Presentation layer only: all
processing still goes through the exact same FFmpeg pipeline used by the
command-line tool (merge_videos.py) - normalize+concat for merging clips,
and apad/-t duration matching for replacing a video's audio track.
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
MAX_FILE_MB = 1024  # matches .streamlit/config.toml [server] maxUploadSize

OPERATIONS = ["Merge Videos", "Add / Replace Audio", "Merge Videos + Audio"]

CARDS = [
    {
        "key": "Merge Videos",
        "icon": "🎬",
        "title": "Merge Videos",
        "desc": "Combine multiple video files into one video.",
    },
    {
        "key": "Add / Replace Audio",
        "icon": "🎵",
        "title": "Add / Replace Audio",
        "desc": "Add music, narration, or another audio track to a video.",
    },
    {
        "key": "Merge Videos + Audio",
        "icon": "🎬🎵",
        "title": "Merge Videos + Audio",
        "desc": "Combine multiple videos and add an audio track.",
    },
]

st.set_page_config(
    page_title="Free Video & Audio Tools | VideoFusion",
    page_icon="🎬",
    layout="wide",
)

# Best-effort meta description. Streamlit does not give scripts a supported
# way to write into <head>, so this is not guaranteed to be read by every
# crawler - the page title set above (which Streamlit does put in <head>)
# is the reliable SEO signal here.
st.markdown(
    '<meta name="description" content="VideoFusion is a free online video and audio tool for '
    'merging videos, adding or replacing audio, and creating combined video files.">',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --vf-accent: #4F46E5;
        --vf-accent-dark: #4338CA;
        --vf-text: #0F172A;
        --vf-muted: #64748B;
        --vf-border: #E2E8F0;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { max-width: 1050px; padding-top: 1.5rem; padding-bottom: 2rem; margin: 0 auto; }

    .vf-hero { text-align: center; padding: 0 0 0.75rem 0; }
    .vf-hero-icon { font-size: 2.2rem; line-height: 1; margin-bottom: 0.1rem; }
    .vf-hero h1 {
        font-size: 2rem; font-weight: 800; color: var(--vf-text);
        margin: 0.1rem 0 0.15rem 0; letter-spacing: -0.02em;
    }
    .vf-tagline {
        font-size: 1rem; font-weight: 600; color: var(--vf-accent);
        margin: 0 0 0.4rem 0;
    }
    .vf-sub { font-size: 0.95rem; color: var(--vf-muted); max-width: 520px; margin: 0 auto; }

    h2.vf-section-title {
        text-align: center; font-size: 1.2rem; font-weight: 700;
        color: var(--vf-text); letter-spacing: 0.02em; margin: 0 0 0.9rem 0;
    }
    .vf-section { margin-top: 2.6rem; }

    .vf-card-icon { font-size: 1.7rem; margin-bottom: 0.2rem; }
    .vf-card-title { font-size: 1.05rem; font-weight: 700; color: var(--vf-text); margin-bottom: 0.2rem; }
    .vf-card-desc { font-size: 0.87rem; color: var(--vf-muted); margin-bottom: 0.5rem; min-height: 2.3rem; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border: 1px solid var(--vf-border) !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }

    .stButton > button {
        border-radius: 8px; font-weight: 600; padding: 0.45rem 1rem;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--vf-accent); border-color: var(--vf-accent);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--vf-accent-dark); border-color: var(--vf-accent-dark);
    }

    .vf-op-title { font-size: 1.35rem; font-weight: 700; color: var(--vf-text); margin-bottom: 0.1rem; }
    .vf-op-desc { color: var(--vf-muted); margin-bottom: 0.6rem; font-size: 0.95rem; }
    .vf-meta { font-size: 0.85rem; color: var(--vf-muted); }

    .vf-info-card {
        background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 10px;
        padding: 0.75rem 1rem; font-size: 0.88rem; color: var(--vf-text); margin: 0.6rem 0;
    }
    .vf-info-card b { color: var(--vf-accent-dark); }

    .vf-benefit-title { font-weight: 700; color: var(--vf-text); font-size: 0.95rem; margin-bottom: 0.15rem; }
    .vf-benefit-desc { font-size: 0.85rem; color: var(--vf-muted); }

    .vf-footer { text-align: center; color: var(--vf-muted); font-size: 0.85rem; padding: 0.5rem 0 0.5rem 0; }
    .vf-footer b { color: var(--vf-text); }

    @media (max-width: 640px) {
        .vf-hero h1 { font-size: 1.7rem; }
        .block-container { padding-top: 1rem; }
        .vf-section { margin-top: 1.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def show_error(user_message: str, technical_detail: str):
    """Friendly headline error for everyone; full FFmpeg/technical detail tucked away for debugging."""
    st.error(user_message)
    with st.expander("Technical details"):
        st.code(technical_detail)


def show_output(video_bytes: bytes, file_name: str, download_key: str):
    st.markdown("#### ✓ Video created successfully")
    st.caption("Your video is ready — preview it below.")
    st.video(video_bytes)
    st.markdown(f'<span class="vf-meta">File: <code>{file_name}</code></span>', unsafe_allow_html=True)
    st.download_button(
        "⬇ Download Video",
        data=video_bytes,
        file_name=file_name,
        mime="video/mp4",
        type="primary",
        key=download_key,
    )


def file_list(files):
    for f in files:
        size_mb = len(f.getbuffer()) / (1024 * 1024)
        st.markdown(f"✓ **{f.name}** &nbsp;<span class='vf-meta'>({size_mb:.1f} MB)</span>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="vf-hero">
        <div class="vf-hero-icon">🎬</div>
        <h1>VideoFusion</h1>
        <p class="vf-tagline">Free Video & Audio Tools</p>
        <p class="vf-sub">Merge videos, add audio, and create your final video with a simple workflow.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# FFmpeg availability (required by every operation)
# ---------------------------------------------------------------------------
ffmpeg_path = find_binary("ffmpeg")
ffprobe_path = find_binary("ffprobe")
if not ffmpeg_path or not ffprobe_path:
    st.error(
        "This tool is temporarily unavailable because FFmpeg isn't installed in this environment. "
        "If you're deploying to Streamlit Community Cloud, make sure a `packages.txt` file with "
        "`ffmpeg` is present in the repo root."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Operation picker: the three cards are the ONLY visible selector.
# The dropdown value still lives in session_state internally (no visible
# widget), so the rest of the app can keep branching on one variable.
# ---------------------------------------------------------------------------
st.session_state.setdefault("current_operation", OPERATIONS[0])


def set_operation(op: str):
    st.session_state.current_operation = op
    st.session_state["_scroll_to_workspace"] = True
    st.toast(f"Switched to {op}", icon="✅")


if st.session_state.pop("_scroll_to_workspace", False):
    # The "Use Tool" cards further down the page change the same operation
    # state as the cards up here - without this, clicking one changes the
    # workspace above without moving the user's scroll position, which
    # looks like the button did nothing.
    st.components.v1.html(
        "<script>window.parent.scrollTo({top: 0, behavior: 'smooth'});</script>",
        height=0,
    )

st.markdown('<h2 class="vf-section-title">WHAT DO YOU WANT TO DO?</h2>', unsafe_allow_html=True)

card_cols = st.columns(3, gap="medium")
for col, card in zip(card_cols, CARDS):
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="vf-card-icon">{card["icon"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="vf-card-title">{card["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="vf-card-desc">{card["desc"]}</div>', unsafe_allow_html=True)
            is_active = st.session_state.current_operation == card["key"]
            st.button(
                f'{card["key"]} →',
                key=f"card_{card['key']}",
                on_click=set_operation,
                args=(card["key"],),
                use_container_width=True,
                type="primary" if is_active else "secondary",
            )

operation = st.session_state.current_operation


# ---------------------------------------------------------------------------
# Operation: Merge Videos
# ---------------------------------------------------------------------------
if operation == "Merge Videos":
    for key, default in {"mv_uploader_key": 0, "mv_result_bytes": None, "mv_result_name": None}.items():
        st.session_state.setdefault(key, default)

    st.markdown('<div class="vf-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="vf-op-title">Merge Videos</div>', unsafe_allow_html=True)
    st.markdown('<div class="vf-op-desc">Combine multiple video files into one video.</div>', unsafe_allow_html=True)

    mv_files = st.file_uploader(
        "Drag & drop your videos here",
        type=SUPPORTED_VIDEO_TYPES,
        accept_multiple_files=True,
        key=f"mv_uploader_{st.session_state.mv_uploader_key}",
        help=f"Supported: {', '.join(t.upper() for t in SUPPORTED_VIDEO_TYPES)}. Maximum {MAX_FILE_MB} MB per file.",
    )

    mv_ready = bool(mv_files) and len(mv_files) >= 2
    mv_edited_df = None

    if mv_files:
        st.write("Your videos:")
        file_list(mv_files)

    if not mv_ready:
        st.info("Add at least 2 videos to merge.")
    else:
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
        st.caption("The order below is the order clips appear in the merged video — edit the Order "
                    "column to rearrange. To remove a file, use the × on its entry above.")
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

        if st.button("🎬 Merge Videos", type="primary", key="mv_merge_btn"):
            ordered_positions = mv_edited_df.sort_values("Order", kind="stable").index.tolist()
            ordered_files = [mv_files[i] for i in ordered_positions]
            status = st.status("Processing your videos... please wait while VideoFusion creates your file.", expanded=True)
            try:
                result_bytes = run_video_merge(
                    ordered_files, mv_crf, mv_preset, mv_output_name, ffmpeg_path, ffprobe_path, status
                )
                status.update(label="Merge complete", state="complete")
                st.session_state.mv_result_bytes = result_bytes
                st.session_state.mv_result_name = mv_output_name
            except MergeError as e:
                status.update(label="Merge failed", state="error")
                show_error("We couldn't merge these videos. Please check the files and try again.", str(e))
            except Exception as e:  # noqa: BLE001 - friendly message instead of a raw traceback
                status.update(label="Merge failed", state="error")
                show_error("Something went wrong while processing your files. Please try again.", str(e))

    if st.session_state.mv_result_bytes:
        show_output(st.session_state.mv_result_bytes, st.session_state.mv_result_name, "mv_download")

    if mv_files or st.session_state.mv_result_bytes:
        if st.button("↻ Clear Selection", key="mv_clear_btn"):
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

    st.markdown('<div class="vf-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="vf-op-title">Add / Replace Audio</div>', unsafe_allow_html=True)
    st.markdown('<div class="vf-op-desc">Add music, narration, or another audio track to your video.</div>', unsafe_allow_html=True)

    col_v, col_a = st.columns(2, gap="medium")
    with col_v:
        st.markdown("**Video**")
        aa_video = st.file_uploader(
            "Drag & drop your video here",
            type=SUPPORTED_VIDEO_TYPES,
            accept_multiple_files=False,
            key=f"aa_video_uploader_{st.session_state.aa_video_uploader_key}",
            help=f"Supported: {', '.join(t.upper() for t in SUPPORTED_VIDEO_TYPES)}. Maximum {MAX_FILE_MB} MB.",
        )
        if aa_video:
            st.markdown(f"✓ **{aa_video.name}**")
    with col_a:
        st.markdown("**Audio**")
        aa_audio = st.file_uploader(
            "Drag & drop your audio here",
            type=SUPPORTED_AUDIO_TYPES,
            accept_multiple_files=False,
            key=f"aa_audio_uploader_{st.session_state.aa_audio_uploader_key}",
            help=f"Supported: {', '.join(t.upper() for t in SUPPORTED_AUDIO_TYPES)}. Maximum {MAX_FILE_MB} MB.",
        )
        if aa_audio:
            st.markdown(f"✓ **{aa_audio.name}**")

    st.markdown(
        """
        <div class="vf-info-card">
        <b>Your video stays unchanged.</b> We preserve the video's visuals and duration.
        If the audio is shorter than the video, silence is added to fill the remaining time.
        If the audio is longer, it is trimmed to match the video's duration.
        </div>
        """,
        unsafe_allow_html=True,
    )

    aa_ready = bool(aa_video) and bool(aa_audio)

    if not aa_ready:
        st.info("Add 1 video and 1 audio file to continue.")
    else:
        aa_output_name = st.text_input(
            "Output filename",
            Path(aa_video.name).stem + "_with_audio.mp4",
            key="aa_output_name",
        ).strip() or "video_with_audio.mp4"
        if not aa_output_name.lower().endswith(".mp4"):
            aa_output_name += ".mp4"

        if st.button("🎵 Add / Replace Audio", type="primary", key="aa_merge_btn"):
            status = st.status("Processing your video... please wait while VideoFusion creates your file.", expanded=True)
            try:
                result_bytes = run_audio_merge(
                    aa_video.getvalue(), aa_audio, aa_output_name, ffmpeg_path, ffprobe_path, status
                )
                status.update(label="Audio merge complete", state="complete")
                st.session_state.aa_result_bytes = result_bytes
                st.session_state.aa_result_name = aa_output_name
            except MergeError as e:
                status.update(label="Audio merge failed", state="error")
                show_error("We couldn't add this audio to your video. Please check the files and try again.", str(e))
            except Exception as e:  # noqa: BLE001 - friendly message instead of a raw traceback
                status.update(label="Audio merge failed", state="error")
                show_error("Something went wrong while processing your files. Please try again.", str(e))

    if st.session_state.aa_result_bytes:
        show_output(st.session_state.aa_result_bytes, st.session_state.aa_result_name, "aa_download")

    if aa_video or aa_audio or st.session_state.aa_result_bytes:
        if st.button("↻ Clear Selection", key="aa_clear_btn"):
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

    st.markdown('<div class="vf-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="vf-op-title">Merge Videos + Audio</div>', unsafe_allow_html=True)
    st.markdown('<div class="vf-op-desc">Combine multiple videos and add an audio track, in one step.</div>', unsafe_allow_html=True)

    st.markdown("**Video files**")
    mva_files = st.file_uploader(
        "Drag & drop your videos here",
        type=SUPPORTED_VIDEO_TYPES,
        accept_multiple_files=True,
        key=f"mva_video_uploader_{st.session_state.mva_video_uploader_key}",
        help=f"Supported: {', '.join(t.upper() for t in SUPPORTED_VIDEO_TYPES)}. Maximum {MAX_FILE_MB} MB per file.",
    )

    if mva_files:
        st.write("Your videos:")
        file_list(mva_files)

    st.markdown("**Audio file**")
    mva_audio = st.file_uploader(
        "Drag & drop your audio here",
        type=SUPPORTED_AUDIO_TYPES,
        accept_multiple_files=False,
        key=f"mva_audio_uploader_{st.session_state.mva_audio_uploader_key}",
        help=f"Supported: {', '.join(t.upper() for t in SUPPORTED_AUDIO_TYPES)}. Maximum {MAX_FILE_MB} MB.",
    )
    if mva_audio:
        st.markdown(f"✓ **{mva_audio.name}**")

    st.markdown(
        """
        <div class="vf-info-card">
        Videos are merged first, keeping their combined length unchanged. If the audio is shorter,
        silence is added to fill the remaining time. If it's longer, it is trimmed to match.
        </div>
        """,
        unsafe_allow_html=True,
    )

    mva_ready = bool(mva_files) and len(mva_files) >= 2 and bool(mva_audio)
    mva_edited_df = None

    if not mva_ready:
        st.info("Add at least 2 videos and 1 audio file to continue.")
    else:
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
        st.caption("The order below is the order clips appear in the merged video — edit the Order "
                    "column to rearrange. To remove a file, use the × on its entry above.")
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

        if st.button("🎬🎵 Merge Videos + Audio", type="primary", key="mva_merge_btn"):
            ordered_positions = mva_edited_df.sort_values("Order", kind="stable").index.tolist()
            ordered_files = [mva_files[i] for i in ordered_positions]
            status = st.status("Processing your videos... please wait while VideoFusion creates your file.", expanded=True)
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
                show_error("We couldn't create your video. Please check the files and try again.", str(e))
            except Exception as e:  # noqa: BLE001 - friendly message instead of a raw traceback
                status.update(label="Merge failed", state="error")
                show_error("Something went wrong while processing your files. Please try again.", str(e))

    if st.session_state.mva_result_bytes:
        show_output(st.session_state.mva_result_bytes, st.session_state.mva_result_name, "mva_download")

    if mva_files or mva_audio or st.session_state.mva_result_bytes:
        if st.button("↻ Clear Selection", key="mva_clear_btn"):
            st.session_state.mva_video_uploader_key += 1
            st.session_state.mva_audio_uploader_key += 1
            st.session_state.mva_result_bytes = None
            st.session_state.mva_result_name = None
            st.rerun()


# ---------------------------------------------------------------------------
# Trust section
# ---------------------------------------------------------------------------
st.markdown('<div class="vf-section"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="vf-section-title">Why use VideoFusion?</h2>', unsafe_allow_html=True)

benefit_cols = st.columns(4, gap="medium")
benefits = [
    ("🆓", "Free to Use", "Use the available VideoFusion tools without a paid subscription."),
    ("⚡", "Simple Workflow", "Upload your files, choose an operation, and create your result."),
    ("🎬", "Video Tools", "Merge multiple video files into one video."),
    ("🎵", "Audio Tools", "Add or replace an audio track in your video."),
]
for col, (icon, title, desc) in zip(benefit_cols, benefits):
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="vf-card-icon">{icon}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="vf-benefit-title">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="vf-benefit-desc">{desc}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="vf-info-card">
    <b>🔒 Clear file handling.</b> Uploaded files and generated videos are processed in a temporary
    working area for your session and are not kept in a database. Download your result before
    starting a new operation or closing the tab.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Video & Audio Tools
# ---------------------------------------------------------------------------
st.markdown('<div class="vf-section"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="vf-section-title">Video & Audio Tools</h2>', unsafe_allow_html=True)

tool_cols = st.columns(3, gap="medium")
for col, card in zip(tool_cols, CARDS):
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="vf-card-icon">{card["icon"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="vf-card-title">{card["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="vf-card-desc">{card["desc"]}</div>', unsafe_allow_html=True)
            st.button(
                "Use Tool →",
                key=f"tool_{card['key']}",
                on_click=set_operation,
                args=(card["key"],),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# FAQ
# ---------------------------------------------------------------------------
st.markdown('<div class="vf-section"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="vf-section-title">Frequently Asked Questions</h2>', unsafe_allow_html=True)

faqs = [
    (
        "What video formats does VideoFusion support?",
        f"VideoFusion supports {', '.join(t.upper() for t in SUPPORTED_VIDEO_TYPES)}.",
    ),
    (
        "How many videos can I merge?",
        "You can merge two or more videos at once — there's no fixed maximum, though very large "
        f"batches take longer to process, and each file is limited to {MAX_FILE_MB} MB.",
    ),
    (
        "Can I add audio to a video?",
        "Yes. Use Add / Replace Audio to attach an audio file to a single video, or "
        "Merge Videos + Audio to do it as part of a merge.",
    ),
    (
        "Can I replace the existing audio?",
        "Yes. Add / Replace Audio always replaces the video's existing audio track with the "
        "audio file you select.",
    ),
    (
        "Does VideoFusion preserve video quality?",
        "When merging, clips are re-encoded to a consistent Full HD format using an adjustable "
        "quality setting (CRF, available under Advanced settings). When only replacing audio on a "
        "single video, the video stream is copied without re-encoding, so its quality is unchanged.",
    ),
    (
        "How are uploaded files handled?",
        "Uploaded files and generated videos are processed in a temporary working area for your "
        "session and are not kept in a database. Download your result before starting a new "
        "operation or closing the tab.",
    ),
    (
        "Is VideoFusion free?",
        "Yes, VideoFusion is currently free to use with no account or payment required.",
    ),
]
for question, answer in faqs:
    with st.expander(question):
        st.write(answer)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div class="vf-footer">
        🎬 <b>VideoFusion</b><br>
        Free Video &amp; Audio Tools<br>
        Merge Videos • Add / Replace Audio • Merge Videos + Audio<br>
        © 2026 VideoFusion
    </div>
    """,
    unsafe_allow_html=True,
)
