"""Memory-safe, model-guided forearm isolation for close-range NIR frames."""

import os
import tempfile
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

# MediaPipe imports matplotlib transitively. Keep its caches in a writable,
# disposable location on hosted workers instead of depending on a home folder.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "veinz-matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "veinz-cache"),
)

try:
    import mediapipe as mp
    from mediapipe.tasks.python.components.containers.keypoint import (
        NormalizedKeypoint,
    )
except ImportError:
    mp = None
    NormalizedKeypoint = None


MODEL_PATH = Path(__file__).resolve().parent / "models" / "magic_touch.tflite"
MODEL_INPUT_LONG_EDGE = 512
_SEGMENTER = None
_SEGMENTER_UNAVAILABLE = False
_SEGMENTER_LOCK = Lock()
logger = logging.getLogger("veinz.arm-segmentation")


@dataclass(frozen=True)
class ArmSegmentation:
    mask: np.ndarray
    confidence: float
    method: str


def _segmenter():
    global _SEGMENTER, _SEGMENTER_UNAVAILABLE
    if _SEGMENTER is not None:
        return _SEGMENTER
    if _SEGMENTER_UNAVAILABLE or mp is None or not MODEL_PATH.exists():
        return None

    with _SEGMENTER_LOCK:
        if _SEGMENTER is None:
            try:
                options = mp.tasks.vision.InteractiveSegmenterOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(MODEL_PATH),
                        delegate=mp.tasks.BaseOptions.Delegate.CPU,
                    ),
                    output_confidence_masks=True,
                    output_category_mask=True,
                )
                _SEGMENTER = (
                    mp.tasks.vision.InteractiveSegmenter.create_from_options(options)
                )
            except Exception:
                _SEGMENTER_UNAVAILABLE = True
                logger.exception(
                    "MediaPipe could not initialize; using geometric arm fallback"
                )
                return None
    return _SEGMENTER


def _component_at(mask, point):
    """Retain the prompted component and discard disconnected model speckle."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary

    x, y = point
    x = int(np.clip(x, 0, labels.shape[1] - 1))
    y = int(np.clip(y, 0, labels.shape[0] - 1))
    label = int(labels[y, x])
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8)


def _shape_metrics(mask):
    ys, xs = np.where(mask > 0)
    if xs.size < 8:
        return 0.0, 0.0, 0.0

    points = np.column_stack((xs, ys)).astype(np.float32)
    eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
    elongation = float(
        np.sqrt((eigenvalues[-1] + 1e-6) / (eigenvalues[0] + 1e-6))
    )
    height, width = mask.shape
    span = max(
        float((xs.max() - xs.min() + 1) / width),
        float((ys.max() - ys.min() + 1) / height),
    )
    return float(mask.mean()), elongation, span


def _broad_candidate_score(mask, confidence):
    area, elongation, span = _shape_metrics(mask)
    if area < 0.045 or area > 0.62 or elongation < 1.25 or span < 0.25:
        return -1e6

    area_preference = 1.0 - min(1.0, abs(area - 0.25) / 0.38)
    return (
        0.35 * float(confidence)
        + 1.05 * min(elongation / 4.0, 1.0)
        + 0.85 * min(span, 1.0)
        + 0.20 * area_preference
    )


def _model_arm_mask(image_bgr):
    """Prompt the lightweight object model on both sides of the frame center."""
    segmenter = _segmenter()
    if segmenter is None:
        return None, None, 0.0

    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    media_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )
    roi_type = mp.tasks.vision.InteractiveSegmenterRegionOfInterest

    best_mask = None
    best_point = None
    best_confidence = 0.0
    best_score = -1e6

    for normalized_x in (0.35, 0.65):
        point = (round(normalized_x * width), round(0.50 * height))
        roi = roi_type(
            format=roi_type.Format.KEYPOINT,
            keypoint=NormalizedKeypoint(x=normalized_x, y=0.50),
        )

        try:
            with _SEGMENTER_LOCK:
                result = segmenter.segment(media_image, roi)
        except Exception:
            continue

        category = np.squeeze(result.category_mask.numpy_view())
        # MagicTouch encodes the prompted foreground as category 0.
        candidate = _component_at(category == 0, point)
        confidence_map = np.squeeze(result.confidence_masks[0].numpy_view())
        confidence = (
            float(np.mean(confidence_map[candidate > 0]))
            if np.any(candidate)
            else 0.0
        )
        score = _broad_candidate_score(candidate, confidence)
        if score > best_score:
            best_mask = candidate
            best_point = point
            best_confidence = confidence
            best_score = score

    return best_mask, best_point, best_confidence


def _refine_broad_mask(image_bgr, broad, point):
    """Preserve the broad arm proposal while trimming table and dark hardware."""
    height, width = broad.shape
    broad = (broad > 0).astype(np.uint8)

    erosion_size = max(7, round(min(height, width) * 0.055))
    if erosion_size % 2 == 0:
        erosion_size += 1
    core = cv2.erode(
        broad,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_size, erosion_size),
        ),
    )
    core = _component_at(core, point)
    if int(core.sum()) < 24:
        core = _component_at(broad, point)

    ys, xs = np.where(core > 0)
    if xs.size < 8:
        return _component_at(broad, point) * 255

    points = np.column_stack((xs, ys)).astype(np.float32)
    center = points.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(points.T))
    minor_axis = eigenvectors[:, 0]
    yy, xx = np.mgrid[:height, :width]
    perpendicular_distance = np.abs(
        (xx - center[0]) * minor_axis[0]
        + (yy - center[1]) * minor_axis[1]
    )
    tube_radius = max(18.0, min(height, width) * 0.27)
    refined = (broad > 0) & (perpendicular_distance <= tube_radius)

    # Dark watch bands separate the usable forearm from the hand. Removing only
    # the darkest broad-mask pixels avoids suppressing faint vessels in skin.
    smoothed = cv2.GaussianBlur(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
        (0, 0),
        sigmaX=2.0,
    )
    sample = smoothed[refined]
    if sample.size:
        dark_cutoff = max(8.0, float(np.percentile(sample, 6.0)))
        tissue = smoothed > dark_cutoff
        tissue_close = max(5, round(min(height, width) * 0.018))
        if tissue_close % 2 == 0:
            tissue_close += 1
        tissue = cv2.morphologyEx(
            tissue.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (tissue_close, tissue_close),
            ),
        ).astype(bool)
        refined &= tissue

    refined = _component_at(refined, point)
    close_size = max(5, round(min(height, width) * 0.014))
    if close_size % 2 == 0:
        close_size += 1
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    return refined * 255


def _geometric_fallback(image_bgr):
    """Keep analyzing a centered arm when the hosted model is unavailable."""
    height, width = image_bgr.shape[:2]
    gray = cv2.GaussianBlur(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
        (0, 0),
        sigmaX=3.0,
    )
    horizontal_band = max(3, round(height * 0.04))
    vertical_band = max(3, round(width * 0.04))
    horizontal_profile = np.median(
        gray[
            max(0, height // 2 - horizontal_band):
            min(height, height // 2 + horizontal_band + 1),
            :,
        ],
        axis=0,
    )
    vertical_profile = np.median(
        gray[
            :,
            max(0, width // 2 - vertical_band):
            min(width, width // 2 + vertical_band + 1),
        ],
        axis=1,
    )
    horizontal_change = float(
        np.mean(np.abs(np.diff(horizontal_profile.astype(np.float32))))
    )
    vertical_change = float(
        np.mean(np.abs(np.diff(vertical_profile.astype(np.float32))))
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    if horizontal_change <= vertical_change:
        axes = (
            max(2, round(width * 0.49)),
            max(2, round(height * 0.22)),
        )
    else:
        axes = (
            max(2, round(width * 0.22)),
            max(2, round(height * 0.49)),
        )
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        axes,
        0,
        0,
        360,
        255,
        -1,
    )
    return mask


def segment_arm(image_bgr):
    """Return a full-resolution mask; fall back to the complete input frame."""
    height, width = image_bgr.shape[:2]
    scale = min(1.0, MODEL_INPUT_LONG_EDGE / max(height, width))
    working = cv2.resize(
        image_bgr,
        (max(32, round(width * scale)), max(32, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

    try:
        broad, point, confidence = _model_arm_mask(working)
    except Exception:
        # Some headless hosts cannot initialize MediaPipe's vision runtime.
        # Segmentation is an enhancement, not a reason to abort vessel analysis.
        logger.exception(
            "Arm model unavailable; continuing with the complete input frame"
        )
        broad, point, confidence = None, None, 0.0
    if broad is None or point is None:
        mask = _geometric_fallback(working)
        method = "geometric-fallback"
        confidence = 0.0
    else:
        mask = _refine_broad_mask(working, broad, point)
        method = "magic-touch-refined"

    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    if float(np.mean(mask > 0)) < 0.025:
        mask = _geometric_fallback(image_bgr)
        method = "geometric-fallback"
        confidence = 0.0

    return ArmSegmentation(
        mask=mask,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        method=method,
    )
