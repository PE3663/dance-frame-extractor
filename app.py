"""
Dance Frame Extractor — Streamlit app
Uploads a dance video, extracts 10–12 high-energy frames,
shows them in a gallery, and lets the user download selected frames.
"""

import io
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_WIDTH = 1280          # Max pixel width when downscaling frames
TARGET_FRAMES = 11        # How many frames to surface (10–12 range)
MIN_MOTION_PERCENTILE = 30  # Only consider frames above this motion percentile


# ─── Core helpers ─────────────────────────────────────────────────────────────

def load_video(uploaded_file) -> tuple[cv2.VideoCapture, str]:
    """
    Write the uploaded file to a named temp file and open it with OpenCV.
    Returns (VideoCapture, temp_file_path). Caller should delete the temp file.
    Raises ValueError on unsupported extensions or if OpenCV can't open the file.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Please upload an MP4, MOV, AVI, MKV, or WEBM file."
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        os.unlink(tmp.name)
        raise ValueError("Could not open the video file. It may be corrupt or in an unsupported codec.")

    return cap, tmp.name


def _compute_motion_scores(cap: cv2.VideoCapture, sample_step: int) -> list[tuple[int, float]]:
    """
    Sample every `sample_step` frames, compute per-frame motion as the mean
    absolute difference from the previous sampled frame.
    Returns a list of (frame_index, motion_score).
    """
    scores: list[tuple[int, float]] = []
    prev_gray: np.ndarray | None = None
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))  # small for speed

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(np.mean(diff))
            else:
                score = 0.0

            scores.append((frame_idx, score))
            prev_gray = gray

        frame_idx += 1

    return scores


def select_frames(cap: cv2.VideoCapture, n: int = TARGET_FRAMES) -> list[Image.Image]:
    """
    Select `n` high-energy frames from the video.

    Strategy:
    1. Sample every Nth frame and score by frame-to-frame motion.
    2. Discard very-low-motion frames (bottom percentile).
    3. Divide the video into `n` equal segments; pick the highest-motion
       candidate in each segment so coverage spans the whole clip.
    4. Decode selected frame indices and return as PIL Images (downscaled).

    Returns a list of PIL.Image objects.
    Raises ValueError if the video has too few frames.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames < n:
        raise ValueError(
            f"Video is too short ({total_frames} frames). "
            f"Need at least {n} frames to extract."
        )

    # Step 1 – score sampled frames
    # Aim for roughly 200–500 candidates to score
    sample_step = max(1, total_frames // 400)
    scores = _compute_motion_scores(cap, sample_step)

    if len(scores) < n:
        # Fallback: uniform sampling when not enough scored frames
        step = total_frames // n
        selected_indices = [i * step for i in range(n)]
    else:
        # Step 2 – filter low-motion frames
        motion_values = np.array([s for _, s in scores])
        threshold = np.percentile(motion_values, MIN_MOTION_PERCENTILE)
        filtered = [(idx, score) for idx, score in scores if score >= threshold or score == 0.0]

        if len(filtered) < n:
            filtered = scores  # too aggressive — fall back to all

        # Step 3 – segment-based selection for spread coverage
        min_idx = filtered[0][0]
        max_idx = filtered[-1][0]
        span = max_idx - min_idx or 1
        segment_size = span / n

        selected_indices: list[int] = []
        for seg in range(n):
            seg_start = min_idx + seg * segment_size
            seg_end = seg_start + segment_size
            candidates = [
                (idx, score)
                for idx, score in filtered
                if seg_start <= idx < seg_end
            ]
            if candidates:
                best_idx = max(candidates, key=lambda x: x[1])[0]
            else:
                # Pick globally best not yet selected
                used = set(selected_indices)
                remaining = [(i, s) for i, s in filtered if i not in used]
                best_idx = max(remaining, key=lambda x: x[1])[0] if remaining else filtered[0][0]
            selected_indices.append(best_idx)

        selected_indices = sorted(set(selected_indices))

    # Step 4 – decode and downscale
    images: list[Image.Image] = []
    for idx in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        # Downscale if wider than MAX_WIDTH
        if img.width > MAX_WIDTH:
            ratio = MAX_WIDTH / img.width
            img = img.resize(
                (MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS
            )
        images.append(img)

    if not images:
        raise ValueError("Could not decode any frames from the video.")

    return images


def image_to_jpeg_bytes(img: Image.Image, quality: int = 88) -> bytes:
    """Encode a PIL Image to JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def create_zip(frames: list[Image.Image], indices: list[int]) -> bytes:
    """
    Bundle the given frames (by index into `frames`) into an in-memory ZIP.
    Returns the ZIP as bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, frame_idx in enumerate(indices):
            jpeg_bytes = image_to_jpeg_bytes(frames[frame_idx])
            zf.writestr(f"dance_frame_{frame_idx + 1:02d}.jpg", jpeg_bytes)
    return buf.getvalue()


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── Page chrome ── */
        [data-testid="stAppViewContainer"] {
            background: #0a0a0f;
        }
        [data-testid="stHeader"] { background: transparent; }

        /* ── Typography ── */
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            color: #e8e2d9;
        }

        /* ── Hero title ── */
        .hero-title {
            font-family: 'Bebas Neue', 'Impact', sans-serif;
            font-size: clamp(2.8rem, 6vw, 5.5rem);
            letter-spacing: 0.04em;
            line-height: 0.92;
            background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 45%, #ffe0c8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.15em;
        }
        .hero-sub {
            font-size: 1.05rem;
            color: #9e9688;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 2rem;
        }

        /* ── Upload zone ── */
        [data-testid="stFileUploaderDropzoneInstructions"] { color: #9e9688; }
        section[data-testid="stFileUploadDropzone"] {
            background: #13131a !important;
            border: 2px dashed #2e2e3a !important;
            border-radius: 12px !important;
            transition: border-color 0.2s;
        }
        section[data-testid="stFileUploadDropzone"]:hover {
            border-color: #ff6b35 !important;
        }

        /* ── Frame cards ── */
        .frame-card {
            background: #13131a;
            border: 1px solid #1e1e28;
            border-radius: 10px;
            padding: 8px;
            margin-bottom: 12px;
            transition: border-color 0.18s, transform 0.18s;
        }
        .frame-card:hover {
            border-color: #ff6b35;
            transform: translateY(-2px);
        }
        .frame-label {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #5c5650;
            margin-top: 4px;
            text-align: center;
        }

        /* ── Buttons ── */
        .stButton > button, .stDownloadButton > button {
            background: #ff6b35 !important;
            color: #0a0a0f !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
            font-size: 0.82rem !important;
            padding: 0.45rem 1rem !important;
            transition: background 0.15s, opacity 0.15s !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #e55a26 !important;
            opacity: 1 !important;
        }

        /* ── Checkboxes ── */
        [data-testid="stCheckbox"] label {
            font-size: 0.82rem;
            color: #c8c0b4;
            letter-spacing: 0.04em;
        }

        /* ── Progress / spinner ── */
        .stProgress > div > div { background: #ff6b35 !important; }

        /* ── Divider ── */
        hr { border-color: #1e1e28; }

        /* ── Info / error boxes ── */
        [data-testid="stAlert"] {
            background: #13131a !important;
            border-radius: 8px !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] { background: #0d0d14; }
        </style>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Dance Frame Extractor",
        page_icon="🕺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="hero-title">DANCE FRAME<br>EXTRACTOR</p>'
        '<p class="hero-sub">Drop your video · Get the best moments · Download in one click</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload your dance video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Accepted formats: MP4, MOV, AVI, MKV, WEBM · Max size depends on your Streamlit deployment.",
    )

    if not uploaded:
        st.info("👆 Upload a dance video above to get started.")
        st.stop()

    # ── Process video ─────────────────────────────────────────────────────────
    if "frames" not in st.session_state or st.session_state.get("last_file") != uploaded.name:
        with st.spinner("Analysing video and selecting high-energy frames…"):
            try:
                cap, tmp_path = load_video(uploaded)
                try:
                    frames = select_frames(cap, n=TARGET_FRAMES)
                finally:
                    cap.release()
                    os.unlink(tmp_path)

                st.session_state["frames"] = frames
                st.session_state["last_file"] = uploaded.name
                # Reset checkbox state when a new file is uploaded
                st.session_state["selected"] = [True] * len(frames)

            except ValueError as exc:
                st.error(f"⚠️ {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"⚠️ Unexpected error while processing the video: {exc}")
                st.stop()

    frames: list[Image.Image] = st.session_state["frames"]

    # Initialise selection state if missing
    if "selected" not in st.session_state or len(st.session_state["selected"]) != len(frames):
        st.session_state["selected"] = [True] * len(frames)

    # ── Gallery ───────────────────────────────────────────────────────────────
    st.markdown(
        f"### 🎞 {len(frames)} frames extracted — pick yours",
        unsafe_allow_html=False,
    )

    cols_per_row = 4
    rows = [frames[i : i + cols_per_row] for i in range(0, len(frames), cols_per_row)]
    flat_idx = 0

    for row_frames in rows:
        cols = st.columns(len(row_frames))
        for col, img in zip(cols, row_frames):
            with col:
                frame_num = flat_idx  # capture for closure
                st.markdown('<div class="frame-card">', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown(
                    f'<p class="frame-label">Frame {frame_num + 1}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                # Checkbox
                checked = st.checkbox(
                    f"Select",
                    key=f"chk_{frame_num}",
                    value=st.session_state["selected"][frame_num],
                )
                st.session_state["selected"][frame_num] = checked

                # Per-frame download
                jpeg = image_to_jpeg_bytes(img)
                st.download_button(
                    label="⬇ JPEG",
                    data=jpeg,
                    file_name=f"dance_frame_{frame_num + 1:02d}.jpg",
                    mime="image/jpeg",
                    key=f"dl_{frame_num}",
                )

            flat_idx += 1

    # ── Bulk download ─────────────────────────────────────────────────────────
    st.divider()
    selected_indices = [i for i, v in enumerate(st.session_state["selected"]) if v]
    n_selected = len(selected_indices)

    col_a, col_b, col_c = st.columns([3, 2, 2])
    with col_a:
        st.markdown(
            f"**{n_selected} / {len(frames)}** frames selected",
            unsafe_allow_html=False,
        )

    with col_b:
        if st.button("✅ Select All", use_container_width=True):
            st.session_state["selected"] = [True] * len(frames)
            st.rerun()

    with col_c:
        if st.button("☐ Deselect All", use_container_width=True):
            st.session_state["selected"] = [False] * len(frames)
            st.rerun()

    st.markdown("")  # spacer

    if n_selected == 0:
        st.warning("Select at least one frame to enable the ZIP download.")
    else:
        zip_bytes = create_zip(frames, selected_indices)
        st.download_button(
            label=f"📦 Download {n_selected} frame{'s' if n_selected != 1 else ''} as ZIP",
            data=zip_bytes,
            file_name="dance_frames.zip",
            mime="application/zip",
            use_container_width=False,
        )

    st.markdown(
        '<p style="color:#3a3830;font-size:0.75rem;margin-top:2rem;text-align:center;">'
        "Dance Frame Extractor · built with Streamlit &amp; OpenCV"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
