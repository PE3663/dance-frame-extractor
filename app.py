"""
Dance Frame Extractor — Streamlit app
Uploads a dance video, extracts 10–12 high-energy frames,
detects standout poses (jumps, extensions, fun moments),
shows them in a gallery, and lets the user download selected frames.
"""

import io
import math
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_WIDTH = 3840              # Max pixel width — preserve full HD / 4K quality
TARGET_FRAMES = 11            # How many motion frames to surface
DEFAULT_POSE_PICKS = 5        # Default number of pose picks
MIN_MOTION_PERCENTILE = 30    # Only consider frames above this motion percentile
POSE_SAMPLE_STEP_DIVIDER = 300  # Sample ~300 frames for pose analysis
MIN_POSE_GAP_SECONDS = 1.5   # Minimum gap between selected pose frames
DEFAULT_SPOTLIGHT_PICKS = 4  # Default number of solo spotlight picks
CROP_PADDING = 0.40          # Extra padding around cropped dancer (fraction of bbox)
MIN_DANCERS_FOR_SPOTLIGHT = 2  # Only spotlight when this many dancers detected
POSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")


# ─── Core helpers ─────────────────────────────────────────────────────────────

def load_video(uploaded_file) -> tuple[cv2.VideoCapture, str]:
    """
    Write the uploaded file to a named temp file and open it with OpenCV.
    Returns (VideoCapture, temp_file_path). Caller should delete the temp file.
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
            gray = cv2.resize(gray, (320, 180))

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(np.mean(diff))
            else:
                score = 0.0

            scores.append((frame_idx, score))
            prev_gray = gray

        frame_idx += 1

    return scores


# ─── Pose detection helpers ──────────────────────────────────────────────────

def _compute_pose_extension(landmarks: list) -> float:
    """
    Score how 'extended' a pose is — high scores mean big jumps, wide splits,
    arms thrown out, etc.
    Uses the new MediaPipe Tasks API landmark list (list of NormalizedLandmark).
    """
    lm = landmarks  # list indexed by landmark ID

    # Key landmark indices (MediaPipe Pose)
    NOSE = 0
    L_WRIST = 15
    R_WRIST = 16
    L_ANKLE = 27
    R_ANKLE = 28
    L_HIP = 23
    R_HIP = 24
    L_SHOULDER = 11
    R_SHOULDER = 12
    L_KNEE = 25
    R_KNEE = 26

    def dist(a, b):
        return math.sqrt((lm[a].x - lm[b].x) ** 2 +
                         (lm[a].y - lm[b].y) ** 2 +
                         (lm[a].z - lm[b].z) ** 2)

    def midpoint_y(a, b):
        return (lm[a].y + lm[b].y) / 2.0

    score = 0.0

    # 1. Arm spread — distance between wrists relative to shoulder width
    shoulder_width = dist(L_SHOULDER, R_SHOULDER)
    wrist_spread = dist(L_WRIST, R_WRIST)
    if shoulder_width > 0.01:
        score += (wrist_spread / shoulder_width) * 2.0

    # 2. Leg spread — distance between ankles relative to hip width
    hip_width = dist(L_HIP, R_HIP)
    ankle_spread = dist(L_ANKLE, R_ANKLE)
    if hip_width > 0.01:
        score += (ankle_spread / hip_width) * 2.0

    # 3. Jump detection — hips above normal standing position
    #    In MediaPipe coords, y=0 is top, y=1 is bottom.
    #    If hip midpoint is unusually high (low y value), it's a jump.
    hip_y = midpoint_y(L_HIP, R_HIP)
    # Hips typically at y~0.55-0.65; above 0.45 is elevated
    if hip_y < 0.45:
        score += (0.45 - hip_y) * 15.0  # big bonus for airborne

    # 4. Hands above head — wrists above nose
    nose_y = lm[NOSE].y
    for wrist_idx in [L_WRIST, R_WRIST]:
        if lm[wrist_idx].y < nose_y - 0.05:
            score += 1.5

    # 5. Deep bend / floor work — knees significantly bent or body low
    for knee_idx, hip_idx in [(L_KNEE, L_HIP), (R_KNEE, R_HIP)]:
        knee_y = lm[knee_idx].y
        h_y = lm[hip_idx].y
        if knee_y > 0.85 and h_y > 0.7:
            score += 1.0

    # 6. Asymmetry bonus — one arm up one down, or split legs = dynamic pose
    wrist_y_diff = abs(lm[L_WRIST].y - lm[R_WRIST].y)
    ankle_y_diff = abs(lm[L_ANKLE].y - lm[R_ANKLE].y)
    score += wrist_y_diff * 3.0
    score += ankle_y_diff * 3.0

    return score


def _detect_motion_spikes(motion_scores: list[tuple[int, float]], fps: float) -> list[tuple[int, float]]:
    """
    Find frames where high motion is immediately followed by stillness —
    the 'freeze' at the peak of a jump or the hold of a pose.
    Returns list of (frame_index, spike_score).
    """
    if len(motion_scores) < 3:
        return []

    spikes: list[tuple[int, float]] = []
    scores_only = [s for _, s in motion_scores]
    mean_motion = np.mean(scores_only)
    std_motion = np.std(scores_only)

    if std_motion < 0.01:
        return []

    for i in range(1, len(motion_scores) - 1):
        prev_score = motion_scores[i - 1][1]
        curr_score = motion_scores[i][1]
        next_score = motion_scores[i + 1][1]

        # High motion followed by sudden drop = freeze after movement
        if prev_score > mean_motion + std_motion * 0.5:
            drop = prev_score - curr_score
            if drop > std_motion * 0.8:
                spike_score = drop * (prev_score / (mean_motion + 0.001))
                spikes.append((motion_scores[i][0], spike_score))

        # Sudden burst after stillness = start of explosive move
        if curr_score > mean_motion + std_motion * 1.5 and prev_score < mean_motion:
            spike_score = curr_score - prev_score
            spikes.append((motion_scores[i][0], spike_score))

    return spikes


def select_pose_frames(
    cap: cv2.VideoCapture,
    motion_scores: list[tuple[int, float]],
    n: int,
    already_selected: set[int],
) -> list[tuple[Image.Image, str]]:
    """
    Select `n` standout pose frames using both MediaPipe pose detection
    and motion spike analysis. Returns list of (PIL.Image, reason_label).
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    min_gap_frames = int(fps * MIN_POSE_GAP_SECONDS)

    if total_frames < 10 or n <= 0:
        return []

    # ── Phase 1: Motion spike analysis ────────────────────────────────────
    spikes = _detect_motion_spikes(motion_scores, fps)

    # ── Phase 2: MediaPipe pose scoring on sampled frames ─────────────────
    sample_step = max(1, total_frames // POSE_SAMPLE_STEP_DIVIDER)
    pose_scores: list[tuple[int, float]] = []

    # New Tasks API — create PoseLandmarker
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    import mediapipe as _mp

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                # Resize for speed
                small = cv2.resize(frame, (320, 180))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect(mp_image)

                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    ext_score = _compute_pose_extension(results.pose_landmarks[0])
                    pose_scores.append((frame_idx, ext_score))

            frame_idx += 1

    # ── Phase 3: Combine scores ───────────────────────────────────────────
    # Normalize both score sets to 0-1 range and merge
    combined: dict[int, float] = {}

    if pose_scores:
        pose_vals = [s for _, s in pose_scores]
        p_min, p_max = min(pose_vals), max(pose_vals)
        p_range = p_max - p_min if p_max > p_min else 1.0
        for idx, score in pose_scores:
            combined[idx] = combined.get(idx, 0) + ((score - p_min) / p_range) * 0.7

    if spikes:
        spike_vals = [s for _, s in spikes]
        s_min, s_max = min(spike_vals), max(spike_vals)
        s_range = s_max - s_min if s_max > s_min else 1.0
        for idx, score in spikes:
            # Find nearest pose-scored frame to add spike bonus
            nearest = idx
            best_dist = float("inf")
            for pidx, _ in pose_scores:
                d = abs(pidx - idx)
                if d < best_dist:
                    best_dist = d
                    nearest = pidx
            if best_dist <= sample_step * 2:
                combined[nearest] = combined.get(nearest, 0) + ((score - s_min) / s_range) * 0.3
            else:
                combined[idx] = combined.get(idx, 0) + ((score - s_min) / s_range) * 0.3

    if not combined:
        return []

    # Sort by score descending
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # ── Phase 4: Select top N with minimum gap, skipping already-selected ─
    selected_indices: list[int] = []
    for idx, score in ranked:
        if idx in already_selected:
            continue
        too_close = False
        for sel_idx in selected_indices:
            if abs(idx - sel_idx) < min_gap_frames:
                too_close = True
                break
        # Also check against already_selected motion frames
        for sel_idx in already_selected:
            if abs(idx - sel_idx) < min_gap_frames:
                too_close = True
                break
        if not too_close:
            selected_indices.append(idx)
        if len(selected_indices) >= n:
            break

    selected_indices.sort()

    # ── Phase 5: Decode frames ────────────────────────────────────────────
    results_out: list[tuple[Image.Image, str]] = []
    for idx in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        if img.width > MAX_WIDTH:
            ratio = MAX_WIDTH / img.width
            img = img.resize((MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)

        # Determine label
        has_pose = idx in dict(pose_scores)
        has_spike = any(abs(idx - sidx) <= sample_step * 2 for sidx, _ in spikes)
        if has_pose and has_spike:
            label = "Pose + Freeze"
        elif has_pose:
            label = "Standout Pose"
        else:
            label = "Motion Spike"

        # Add timestamp
        timestamp = idx / fps
        mins = int(timestamp // 60)
        secs = timestamp % 60
        time_str = f"{mins}:{secs:04.1f}"
        label = f"{label} @ {time_str}"

        results_out.append((img, label))

    return results_out


def _get_dancer_bbox(landmarks: list, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """
    Compute a bounding box (x1, y1, x2, y2) around a single dancer's landmarks.
    Adds CROP_PADDING around the extremes.
    """
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    w = x2 - x1
    h = y2 - y1
    pad_x = w * CROP_PADDING
    pad_y = h * CROP_PADDING

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(img_w, int(x2 + pad_x))
    y2 = min(img_h, int(y2 + pad_y))

    return x1, y1, x2, y2


def _get_dancer_center(landmarks: list) -> tuple[float, float]:
    """Get the hip midpoint of a dancer (normalized 0-1 coords)."""
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    return hip_x, hip_y


def _score_standout(dancer_landmarks: list, all_dancers: list[list], dancer_idx: int) -> float:
    """
    Score how much a single dancer stands out from the group.
    Prioritises spatial isolation — a dancer physically separated from everyone else.
    Also rewards dynamic poses (extension, jump height).
    """
    ext = _compute_pose_extension(dancer_landmarks)
    my_cx, my_cy = _get_dancer_center(dancer_landmarks)

    # ── Spatial isolation: min distance to any other dancer ──
    min_dist = float("inf")
    other_centers = []
    for i, dlm in enumerate(all_dancers):
        if i == dancer_idx:
            continue
        ox, oy = _get_dancer_center(dlm)
        other_centers.append((ox, oy))
        dist = math.hypot(my_cx - ox, my_cy - oy)
        if dist < min_dist:
            min_dist = dist

    if not other_centers:
        isolation = 0.0
    else:
        # Also compute average distance to group centroid
        gx = np.mean([c[0] for c in other_centers])
        gy = np.mean([c[1] for c in other_centers])
        dist_to_centroid = math.hypot(my_cx - gx, my_cy - gy)
        # Isolation combines nearest-neighbour and centroid distance
        isolation = min_dist * 0.6 + dist_to_centroid * 0.4

    # ── Pose deviation from group average ──
    if len(all_dancers) >= MIN_DANCERS_FOR_SPOTLIGHT:
        lm = dancer_landmarks
        my_metrics = np.array([
            lm[15].y, lm[16].y,
            lm[27].y, lm[28].y,
            abs(lm[15].x - lm[16].x),
            abs(lm[27].x - lm[28].x),
        ])
        all_metrics = []
        for i, dlm in enumerate(all_dancers):
            if i == dancer_idx:
                continue
            all_metrics.append(np.array([
                dlm[15].y, dlm[16].y,
                dlm[27].y, dlm[28].y,
                abs(dlm[15].x - dlm[16].x),
                abs(dlm[27].x - dlm[28].x),
            ]))
        deviation = float(np.linalg.norm(my_metrics - np.mean(all_metrics, axis=0))) if all_metrics else 0.0
    else:
        deviation = 0.0

    # Combined: 45% isolation, 30% extension, 25% pose deviation
    return isolation * 10.0 * 0.45 + ext * 0.30 + deviation * 5.0 * 0.25


def select_spotlight_frames(
    cap: cv2.VideoCapture,
    n: int,
    already_selected: set[int],
) -> list[tuple[Image.Image, Image.Image, str]]:
    """
    Find frames where one dancer stands out from a group.
    Returns list of (full_frame, cropped_closeup, label).
    Uses multi-person pose detection.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    min_gap_frames = int(fps * MIN_POSE_GAP_SECONDS)

    if total_frames < 10 or n <= 0:
        return []

    # Sample more densely for spotlight — we need to catch fleeting solo moments
    sample_step = max(1, total_frames // (POSE_SAMPLE_STEP_DIVIDER * 2))

    # Multi-person pose detection
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        num_poses=10,  # detect up to 10 dancers
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    import mediapipe as _mp

    # Collect: (frame_idx, standout_score, best_dancer_idx, num_dancers, all_landmarks)
    candidates: list[tuple[int, float, int, int, list]] = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                small = cv2.resize(frame, (960, 540))  # higher res for reliable multi-person detection
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect(mp_image)

                if results.pose_landmarks and len(results.pose_landmarks) >= MIN_DANCERS_FOR_SPOTLIGHT:
                    all_landmarks = results.pose_landmarks
                    best_score = -1.0
                    best_idx = 0

                    for i, dancer_lm in enumerate(all_landmarks):
                        score = _score_standout(dancer_lm, all_landmarks, i)
                        if score > best_score:
                            best_score = score
                            best_idx = i

                    candidates.append((
                        frame_idx,
                        best_score,
                        best_idx,
                        len(all_landmarks),
                        all_landmarks,
                    ))

            frame_idx += 1

    if not candidates:
        return []

    # Rank by standout score
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Select top N with gap enforcement
    selected: list[tuple[int, float, int, int]] = []
    for frame_idx, score, best_dancer, num_dancers, _ in candidates:
        if frame_idx in already_selected:
            continue
        too_close = any(
            abs(frame_idx - s[0]) < min_gap_frames
            for s in selected
        ) or any(
            abs(frame_idx - si) < min_gap_frames
            for si in already_selected
        )
        if not too_close:
            selected.append((frame_idx, score, best_dancer, num_dancers))
        if len(selected) >= n:
            break

    selected.sort(key=lambda x: x[0])

    # Decode frames and crop
    # We need to re-detect on full-res frames for accurate cropping
    results_out: list[tuple[Image.Image, Image.Image, str]] = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_idx, score, best_dancer_hint, num_dancers in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame_rgb.shape[:2]

            # Re-detect at full res (capped at 1280) for accurate bbox
            if img_w > 1280:
                scale = 1280 / img_w
                detect_rgb = cv2.resize(frame_rgb, (1280, int(img_h * scale)))
            else:
                detect_rgb = frame_rgb.copy()
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=detect_rgb)
            results = landmarker.detect(mp_image)

            full_img = Image.fromarray(frame_rgb)
            if full_img.width > MAX_WIDTH:
                ratio = MAX_WIDTH / full_img.width
                full_img = full_img.resize(
                    (MAX_WIDTH, int(full_img.height * ratio)), Image.LANCZOS
                )
                img_w, img_h = full_img.size

            if results.pose_landmarks and len(results.pose_landmarks) >= MIN_DANCERS_FOR_SPOTLIGHT:
                # Re-score to find the best dancer on the full-res detection
                all_lm = results.pose_landmarks
                best_score = -1.0
                best_idx = 0
                for i, dlm in enumerate(all_lm):
                    s = _score_standout(dlm, all_lm, i)
                    if s > best_score:
                        best_score = s
                        best_idx = i

                dancer_lm = all_lm[best_idx]
                x1, y1, x2, y2 = _get_dancer_bbox(dancer_lm, img_w, img_h)

                # Ensure minimum crop size for a usable close-up
                crop_w = x2 - x1
                crop_h = y2 - y1
                min_crop_w = max(300, img_w // 4)
                min_crop_h = max(400, img_h // 3)
                if crop_w < min_crop_w:
                    cx = (x1 + x2) // 2
                    x1 = max(0, cx - min_crop_w // 2)
                    x2 = min(img_w, cx + min_crop_w // 2)
                if crop_h < min_crop_h:
                    cy = (y1 + y2) // 2
                    y1 = max(0, cy - min_crop_h // 2)
                    y2 = min(img_h, cy + min_crop_h // 2)

                cropped = full_img.crop((x1, y1, x2, y2))
            else:
                # Fallback: center crop
                cx, cy = img_w // 2, img_h // 2
                crop_size = min(img_w, img_h) // 2
                cropped = full_img.crop((
                    max(0, cx - crop_size // 2),
                    max(0, cy - crop_size),
                    min(img_w, cx + crop_size // 2),
                    min(img_h, cy + crop_size),
                ))

            timestamp = frame_idx / fps
            mins = int(timestamp // 60)
            secs = timestamp % 60
            label = f"Solo Spotlight ({num_dancers} dancers) @ {mins}:{secs:04.1f}"

            results_out.append((full_img, cropped, label))

    return results_out


def select_frames(cap: cv2.VideoCapture, n: int = TARGET_FRAMES) -> tuple[list[Image.Image], list[tuple[int, float]], set[int]]:
    """
    Select `n` high-energy frames from the video.
    Returns (images, motion_scores, selected_frame_indices).
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames < n:
        raise ValueError(
            f"Video is too short ({total_frames} frames). "
            f"Need at least {n} frames to extract."
        )

    sample_step = max(1, total_frames // 400)
    scores = _compute_motion_scores(cap, sample_step)

    if len(scores) < n:
        step = total_frames // n
        selected_indices = [i * step for i in range(n)]
    else:
        motion_values = np.array([s for _, s in scores])
        threshold = np.percentile(motion_values, MIN_MOTION_PERCENTILE)
        filtered = [(idx, score) for idx, score in scores if score >= threshold or score == 0.0]

        if len(filtered) < n:
            filtered = scores

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
                used = set(selected_indices)
                remaining = [(i, s) for i, s in filtered if i not in used]
                best_idx = max(remaining, key=lambda x: x[1])[0] if remaining else filtered[0][0]
            selected_indices.append(best_idx)

        selected_indices = sorted(set(selected_indices))

    images: list[Image.Image] = []
    final_indices: list[int] = []
    for idx in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        if img.width > MAX_WIDTH:
            ratio = MAX_WIDTH / img.width
            img = img.resize((MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)
        images.append(img)
        final_indices.append(idx)

    if not images:
        raise ValueError("Could not decode any frames from the video.")

    return images, scores, set(final_indices)


def image_to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    """Encode a PIL Image to JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def create_zip(frames: list[Image.Image], indices: list[int], prefix: str = "dance_frame") -> bytes:
    """Bundle the given frames into an in-memory ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, frame_idx in enumerate(indices):
            jpeg_bytes = image_to_jpeg_bytes(frames[frame_idx])
            zf.writestr(f"{prefix}_{frame_idx + 1:02d}.jpg", jpeg_bytes)
    return buf.getvalue()


def create_combined_zip(
    motion_frames: list[Image.Image],
    motion_indices: list[int],
    pose_frames: list[tuple[Image.Image, str]],
    pose_indices: list[int],
    spotlight_frames: list[tuple[Image.Image, Image.Image, str]] | None = None,
    spotlight_indices: list[int] | None = None,
) -> bytes:
    """Bundle selected motion, pose, and spotlight frames into one ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for frame_idx in motion_indices:
            jpeg_bytes = image_to_jpeg_bytes(motion_frames[frame_idx])
            zf.writestr(f"motion/frame_{frame_idx + 1:02d}.jpg", jpeg_bytes)
        for i, pidx in enumerate(pose_indices):
            img, label = pose_frames[pidx]
            jpeg_bytes = image_to_jpeg_bytes(img)
            safe_label = label.replace(" ", "_").replace(":", "-").replace("/", "-")
            zf.writestr(f"poses/pose_{i + 1:02d}_{safe_label}.jpg", jpeg_bytes)
        if spotlight_frames and spotlight_indices:
            for j, sidx in enumerate(spotlight_indices):
                full_img, crop_img, label = spotlight_frames[sidx]
                safe_label = label.replace(" ", "_").replace(":", "-").replace("/", "-").replace("@", "at").replace("(", "").replace(")", "")
                zf.writestr(f"spotlights/spotlight_{j + 1:02d}_wide_{safe_label}.jpg", image_to_jpeg_bytes(full_img))
                zf.writestr(f"spotlights/spotlight_{j + 1:02d}_closeup_{safe_label}.jpg", image_to_jpeg_bytes(crop_img))
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

        /* ── Section headers ── */
        .section-header {
            font-family: 'Bebas Neue', 'Impact', sans-serif;
            font-size: 1.6rem;
            letter-spacing: 0.05em;
            color: #ff6b35;
            margin: 1.5rem 0 0.5rem 0;
        }
        .section-desc {
            font-size: 0.85rem;
            color: #7a746c;
            margin-bottom: 1rem;
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

        /* ── Pose cards (accent border) ── */
        .pose-card {
            background: #13131a;
            border: 1px solid #2a1f14;
            border-radius: 10px;
            padding: 8px;
            margin-bottom: 12px;
            transition: border-color 0.18s, transform 0.18s;
        }
        .pose-card:hover {
            border-color: #ff6b35;
            transform: translateY(-2px);
        }
        .pose-label {
            font-size: 0.68rem;
            letter-spacing: 0.06em;
            color: #ff6b35;
            margin-top: 4px;
            text-align: center;
            font-weight: 500;
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

        /* ── Spotlight cards ── */
        .spotlight-card {
            background: #13131a;
            border: 1px solid #1a2a1a;
            border-radius: 10px;
            padding: 8px;
            margin-bottom: 12px;
            transition: border-color 0.18s, transform 0.18s;
        }
        .spotlight-card:hover {
            border-color: #35c9ff;
            transform: translateY(-2px);
        }
        .spotlight-label {
            font-size: 0.68rem;
            letter-spacing: 0.06em;
            color: #35c9ff;
            margin-top: 4px;
            text-align: center;
            font-weight: 500;
        }
        .spotlight-pair {
            display: flex;
            gap: 8px;
            align-items: stretch;
        }

        /* ── Slider ── */
        [data-testid="stSlider"] label {
            color: #c8c0b4 !important;
        }
        </style>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True,
    )


def _render_gallery(
    frames,
    session_key: str,
    card_class: str,
    label_class: str,
    label_prefix: str,
    cols_per_row: int = 4,
    labels: list[str] | None = None,
):
    """Render a frame gallery with checkboxes and download buttons."""
    if f"{session_key}_selected" not in st.session_state or \
       len(st.session_state[f"{session_key}_selected"]) != len(frames):
        st.session_state[f"{session_key}_selected"] = [True] * len(frames)

    rows = [frames[i : i + cols_per_row] for i in range(0, len(frames), cols_per_row)]
    flat_idx = 0

    for row_frames in rows:
        cols = st.columns(len(row_frames))
        for col, item in zip(cols, row_frames):
            with col:
                frame_num = flat_idx
                img = item[0] if isinstance(item, tuple) else item
                display_label = labels[frame_num] if labels else f"{label_prefix} {frame_num + 1}"

                st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown(
                    f'<p class="{label_class}">{display_label}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                checked = st.checkbox(
                    "Select",
                    key=f"{session_key}_chk_{frame_num}",
                    value=st.session_state[f"{session_key}_selected"][frame_num],
                )
                st.session_state[f"{session_key}_selected"][frame_num] = checked

                jpeg = image_to_jpeg_bytes(img)
                st.download_button(
                    label="⬇ JPEG",
                    data=jpeg,
                    file_name=f"{session_key}_{frame_num + 1:02d}.jpg",
                    mime="image/jpeg",
                    key=f"{session_key}_dl_{frame_num}",
                )

            flat_idx += 1


def main() -> None:
    st.set_page_config(
        page_title="Dance Frame Extractor",
        page_icon="🕺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="hero-title">DANCE FRAME<br>EXTRACTOR</p>'
        '<p class="hero-sub">Drop your video · Get the best moments · Download in one click</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Settings ──────────────────────────────────────────────────────────
    settings_col1, settings_col2 = st.columns(2)
    with settings_col1:
        pose_picks = st.slider(
            "Pose picks — jumps, extensions & standout moments",
            min_value=0,
            max_value=15,
            value=DEFAULT_POSE_PICKS,
            step=1,
            help="How many extra frames to find using AI pose detection + motion spike analysis. "
                 "Set to 0 to disable.",
        )
    with settings_col2:
        spotlight_picks = st.slider(
            "Solo spotlights — individuals standing out from the group",
            min_value=0,
            max_value=10,
            value=DEFAULT_SPOTLIGHT_PICKS,
            step=1,
            help="Find moments where one dancer stands out from a group. "
                 "Shows full frame + cropped close-up. Set to 0 to disable.",
        )

    # ── Upload ────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload your dance video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Accepted formats: MP4, MOV, AVI, MKV, WEBM · Max size depends on your Streamlit deployment.",
    )

    if not uploaded:
        st.info("👆 Upload a dance video above to get started.")
        st.stop()

    # ── Process video (motion frames) ─────────────────────────────────────
    needs_reprocess = (
        "frames" not in st.session_state
        or st.session_state.get("last_file") != uploaded.name
    )

    if needs_reprocess:
        with st.spinner("Analysing video and selecting high-energy frames…"):
            try:
                cap, tmp_path = load_video(uploaded)
                try:
                    frames, motion_scores, motion_indices = select_frames(cap, n=TARGET_FRAMES)
                    st.session_state["frames"] = frames
                    st.session_state["motion_scores"] = motion_scores
                    st.session_state["motion_indices"] = motion_indices
                    st.session_state["last_file"] = uploaded.name
                    st.session_state["motion_selected"] = [True] * len(frames)
                    # Clear pose frames so they get re-generated
                    st.session_state.pop("pose_frames", None)
                    st.session_state.pop("pose_n", None)
                    st.session_state["tmp_path"] = tmp_path
                    # Keep cap open for pose detection
                    st.session_state["_cap_path"] = tmp_path
                finally:
                    cap.release()

            except ValueError as exc:
                st.error(f"⚠️ {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"⚠️ Unexpected error while processing the video: {exc}")
                st.stop()

    frames: list[Image.Image] = st.session_state["frames"]

    # ── Process video (pose frames) ───────────────────────────────────────
    needs_pose_reprocess = (
        pose_picks > 0
        and (
            "pose_frames" not in st.session_state
            or st.session_state.get("pose_n") != pose_picks
            or st.session_state.get("pose_file") != uploaded.name
        )
    )

    if needs_pose_reprocess:
        with st.spinner("Detecting standout poses and jumps with AI…"):
            try:
                tmp_path = st.session_state.get("_cap_path") or st.session_state.get("tmp_path")
                if not tmp_path or not os.path.exists(tmp_path):
                    # Re-load the video
                    uploaded.seek(0)
                    cap, tmp_path = load_video(uploaded)
                else:
                    cap = cv2.VideoCapture(tmp_path)

                try:
                    pose_results = select_pose_frames(
                        cap,
                        st.session_state["motion_scores"],
                        pose_picks,
                        st.session_state["motion_indices"],
                    )
                    st.session_state["pose_frames"] = pose_results
                    st.session_state["pose_n"] = pose_picks
                    st.session_state["pose_file"] = uploaded.name
                    st.session_state["pose_selected"] = [True] * len(pose_results)
                finally:
                    cap.release()

            except Exception as exc:
                st.warning(f"Pose detection encountered an issue: {exc}")
                st.session_state["pose_frames"] = []
                st.session_state["pose_n"] = pose_picks
                st.session_state["pose_file"] = uploaded.name

    # ── Process video (solo spotlights) ─────────────────────────────────
    needs_spotlight_reprocess = (
        spotlight_picks > 0
        and (
            "spotlight_frames" not in st.session_state
            or st.session_state.get("spotlight_n") != spotlight_picks
            or st.session_state.get("spotlight_file") != uploaded.name
        )
    )

    if needs_spotlight_reprocess:
        with st.spinner("Scanning for solo spotlight moments in group formations…"):
            try:
                tmp_path = st.session_state.get("_cap_path") or st.session_state.get("tmp_path")
                if not tmp_path or not os.path.exists(tmp_path):
                    uploaded.seek(0)
                    cap, tmp_path = load_video(uploaded)
                else:
                    cap = cv2.VideoCapture(tmp_path)

                try:
                    # Combine motion + pose indices for gap enforcement
                    all_selected = set(st.session_state.get("motion_indices", set()))
                    spotlight_results = select_spotlight_frames(
                        cap,
                        spotlight_picks,
                        all_selected,
                    )
                    st.session_state["spotlight_frames"] = spotlight_results
                    st.session_state["spotlight_n"] = spotlight_picks
                    st.session_state["spotlight_file"] = uploaded.name
                    st.session_state["spotlight_selected"] = [True] * len(spotlight_results)
                finally:
                    cap.release()

            except Exception as exc:
                st.warning(f"Solo spotlight detection encountered an issue: {exc}")
                st.session_state["spotlight_frames"] = []
                st.session_state["spotlight_n"] = spotlight_picks
                st.session_state["spotlight_file"] = uploaded.name

    # Clean up temp file if all passes are done
    tmp_path = st.session_state.pop("_cap_path", None) or st.session_state.pop("tmp_path", None)
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Motion Gallery ────────────────────────────────────────────────────
    st.markdown(
        '<p class="section-header">🎞 HIGH-ENERGY FRAMES</p>'
        f'<p class="section-desc">{len(frames)} frames selected by motion analysis</p>',
        unsafe_allow_html=True,
    )

    _render_gallery(
        frames,
        session_key="motion",
        card_class="frame-card",
        label_class="frame-label",
        label_prefix="Frame",
    )

    # ── Motion bulk controls ──────────────────────────────────────────────
    st.divider()
    motion_sel = [i for i, v in enumerate(st.session_state.get("motion_selected", [])) if v]
    n_motion_sel = len(motion_sel)

    col_a, col_b, col_c = st.columns([3, 2, 2])
    with col_a:
        st.markdown(f"**{n_motion_sel} / {len(frames)}** motion frames selected")
    with col_b:
        if st.button("✅ Select All", key="motion_sel_all", use_container_width=True):
            st.session_state["motion_selected"] = [True] * len(frames)
            st.rerun()
    with col_c:
        if st.button("☐ Deselect All", key="motion_desel_all", use_container_width=True):
            st.session_state["motion_selected"] = [False] * len(frames)
            st.rerun()

    if n_motion_sel > 0:
        zip_bytes = create_zip(frames, motion_sel, prefix="motion_frame")
        st.download_button(
            label=f"📦 Download {n_motion_sel} motion frame{'s' if n_motion_sel != 1 else ''} as ZIP",
            data=zip_bytes,
            file_name="motion_frames.zip",
            mime="application/zip",
            key="dl_motion_zip",
        )

    # ── Pose Gallery ──────────────────────────────────────────────────────
    pose_frames: list[tuple[Image.Image, str]] = st.session_state.get("pose_frames", [])

    if pose_picks > 0 and pose_frames:
        st.divider()
        st.markdown(
            '<p class="section-header">⚡ STANDOUT POSES</p>'
            f'<p class="section-desc">{len(pose_frames)} moments detected — jumps, extensions & freezes</p>',
            unsafe_allow_html=True,
        )

        pose_labels = [label for _, label in pose_frames]
        _render_gallery(
            pose_frames,
            session_key="pose",
            card_class="pose-card",
            label_class="pose-label",
            label_prefix="Pose",
            labels=pose_labels,
        )

        # Pose bulk controls
        st.divider()
        pose_sel = [i for i, v in enumerate(st.session_state.get("pose_selected", [])) if v]
        n_pose_sel = len(pose_sel)

        col_d, col_e, col_f = st.columns([3, 2, 2])
        with col_d:
            st.markdown(f"**{n_pose_sel} / {len(pose_frames)}** pose frames selected")
        with col_e:
            if st.button("✅ Select All", key="pose_sel_all", use_container_width=True):
                st.session_state["pose_selected"] = [True] * len(pose_frames)
                st.rerun()
        with col_f:
            if st.button("☐ Deselect All", key="pose_desel_all", use_container_width=True):
                st.session_state["pose_selected"] = [False] * len(pose_frames)
                st.rerun()

        if n_pose_sel > 0:
            # Build pose-only zip
            pose_zip_buf = io.BytesIO()
            with zipfile.ZipFile(pose_zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, pidx in enumerate(pose_sel):
                    img, label = pose_frames[pidx]
                    jpeg_bytes = image_to_jpeg_bytes(img)
                    safe_label = label.replace(" ", "_").replace(":", "-").replace("/", "-").replace("@", "at")
                    zf.writestr(f"pose_{i + 1:02d}_{safe_label}.jpg", jpeg_bytes)

            st.download_button(
                label=f"📦 Download {n_pose_sel} pose frame{'s' if n_pose_sel != 1 else ''} as ZIP",
                data=pose_zip_buf.getvalue(),
                file_name="pose_frames.zip",
                mime="application/zip",
                key="dl_pose_zip",
            )

    elif pose_picks > 0 and not pose_frames:
        st.divider()
        st.info("No standout poses detected. Try a video with more dynamic movement.")

    # ── Spotlight Gallery ───────────────────────────────────────────────
    spotlight_frames: list[tuple[Image.Image, Image.Image, str]] = st.session_state.get("spotlight_frames", [])
    spotlight_sel: list[int] = []

    if spotlight_picks > 0 and spotlight_frames:
        st.divider()
        st.markdown(
            '<p class="section-header">🔦 SOLO SPOTLIGHTS</p>'
            f'<p class="section-desc">{len(spotlight_frames)} individuals standing out from the group — wide shot + close-up</p>',
            unsafe_allow_html=True,
        )

        if "spotlight_selected" not in st.session_state or \
           len(st.session_state["spotlight_selected"]) != len(spotlight_frames):
            st.session_state["spotlight_selected"] = [True] * len(spotlight_frames)

        for i, (full_img, crop_img, label) in enumerate(spotlight_frames):
            st.markdown(f'<div class="spotlight-card">', unsafe_allow_html=True)
            col_wide, col_crop = st.columns([3, 2])
            with col_wide:
                st.image(full_img, caption="Wide shot", use_container_width=True)
            with col_crop:
                st.image(crop_img, caption="Close-up", use_container_width=True)
            st.markdown(
                f'<p class="spotlight-label">{label}</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
            with ctrl_col1:
                checked = st.checkbox(
                    "Select",
                    key=f"spotlight_chk_{i}",
                    value=st.session_state["spotlight_selected"][i],
                )
                st.session_state["spotlight_selected"][i] = checked
            with ctrl_col2:
                wide_jpeg = image_to_jpeg_bytes(full_img)
                st.download_button(
                    label="⬇ Wide",
                    data=wide_jpeg,
                    file_name=f"spotlight_{i + 1:02d}_wide.jpg",
                    mime="image/jpeg",
                    key=f"spotlight_dl_wide_{i}",
                )
            with ctrl_col3:
                crop_jpeg = image_to_jpeg_bytes(crop_img)
                st.download_button(
                    label="⬇ Close-up",
                    data=crop_jpeg,
                    file_name=f"spotlight_{i + 1:02d}_closeup.jpg",
                    mime="image/jpeg",
                    key=f"spotlight_dl_crop_{i}",
                )

        # Spotlight bulk controls
        st.divider()
        spotlight_sel = [i for i, v in enumerate(st.session_state.get("spotlight_selected", [])) if v]
        n_spotlight_sel = len(spotlight_sel)

        col_g, col_h, col_i = st.columns([3, 2, 2])
        with col_g:
            st.markdown(f"**{n_spotlight_sel} / {len(spotlight_frames)}** spotlight frames selected")
        with col_h:
            if st.button("✅ Select All", key="spotlight_sel_all", use_container_width=True):
                st.session_state["spotlight_selected"] = [True] * len(spotlight_frames)
                st.rerun()
        with col_i:
            if st.button("☐ Deselect All", key="spotlight_desel_all", use_container_width=True):
                st.session_state["spotlight_selected"] = [False] * len(spotlight_frames)
                st.rerun()

        if n_spotlight_sel > 0:
            spot_zip_buf = io.BytesIO()
            with zipfile.ZipFile(spot_zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for j, sidx in enumerate(spotlight_sel):
                    full_img, crop_img, label = spotlight_frames[sidx]
                    safe_label = label.replace(" ", "_").replace(":", "-").replace("/", "-").replace("@", "at").replace("(", "").replace(")", "")
                    zf.writestr(f"spotlight_{j + 1:02d}_wide_{safe_label}.jpg", image_to_jpeg_bytes(full_img))
                    zf.writestr(f"spotlight_{j + 1:02d}_closeup_{safe_label}.jpg", image_to_jpeg_bytes(crop_img))

            st.download_button(
                label=f"📦 Download {n_spotlight_sel} spotlight{'s' if n_spotlight_sel != 1 else ''} (wide + close-up) as ZIP",
                data=spot_zip_buf.getvalue(),
                file_name="spotlight_frames.zip",
                mime="application/zip",
                key="dl_spotlight_zip",
            )

    elif spotlight_picks > 0 and not spotlight_frames:
        st.divider()
        st.info("No solo spotlight moments found. This works best with group dance videos where dancers take turns standing out.")

    # ── Combined download ─────────────────────────────────────────────────
    active_pose_sel = pose_sel if (pose_picks > 0 and pose_frames) else []
    active_spotlight_sel = spotlight_sel if (spotlight_picks > 0 and spotlight_frames) else []
    total_selected = n_motion_sel + len(active_pose_sel) + len(active_spotlight_sel)
    has_extras = len(active_pose_sel) > 0 or len(active_spotlight_sel) > 0

    if total_selected > 0 and has_extras:
        st.divider()
        combined_zip = create_combined_zip(
            frames,
            motion_sel,
            pose_frames if active_pose_sel else [],
            active_pose_sel,
            spotlight_frames if active_spotlight_sel else None,
            active_spotlight_sel if active_spotlight_sel else None,
        )
        st.download_button(
            label=f"📦 Download ALL {total_selected} selected frames as ZIP",
            data=combined_zip,
            file_name="dance_frames_all.zip",
            mime="application/zip",
            key="dl_combined_zip",
            use_container_width=True,
        )

    st.markdown(
        '<p style="color:#3a3830;font-size:0.75rem;margin-top:2rem;text-align:center;">'
        "Dance Frame Extractor · built with Streamlit, OpenCV &amp; MediaPipe"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
