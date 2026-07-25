"""Memory-safe, model-guided forearm isolation for close-range NIR frames."""

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        Interpreter = None


MODEL_PATH = Path(__file__).resolve().parent / "models" / "magic_touch.tflite"
MODEL_INPUT_LONG_EDGE = 512
MODEL_TENSOR_SIZE = 512
_SEGMENTER = None
_SEGMENTER_UNAVAILABLE = False
_SEGMENTER_LOCK = Lock()
logger = logging.getLogger("veinz.arm-segmentation")


@dataclass(frozen=True)
class ArmSegmentation:
    mask: np.ndarray
    confidence: float
    method: str
    model_name: str
    model_tier: str
    runtime: str
    fallback_used: bool


def _segmenter():
    global _SEGMENTER, _SEGMENTER_UNAVAILABLE
    if _SEGMENTER is not None:
        return _SEGMENTER
    if _SEGMENTER_UNAVAILABLE or Interpreter is None or not MODEL_PATH.exists():
        return None

    with _SEGMENTER_LOCK:
        if _SEGMENTER is None:
            try:
                interpreter = Interpreter(
                    model_path=str(MODEL_PATH),
                    num_threads=4,
                )
                interpreter.allocate_tensors()
                input_detail = interpreter.get_input_details()[0]
                output_detail = interpreter.get_output_details()[0]
                expected_input = (1, MODEL_TENSOR_SIZE, MODEL_TENSOR_SIZE, 4)
                if tuple(input_detail["shape"]) != expected_input:
                    raise RuntimeError(
                        "Unexpected MagicTouch input shape "
                        f"{tuple(input_detail['shape'])}"
                    )
                _SEGMENTER = (
                    interpreter,
                    int(input_detail["index"]),
                    int(output_detail["index"]),
                )
            except Exception:
                _SEGMENTER_UNAVAILABLE = True
                logger.exception(
                    "LiteRT could not initialize MagicTouch; "
                    "arm isolation will use the full image"
                )
                return None
    return _SEGMENTER


def _predict_foreground(image_bgr, normalized_x):
    """Run MagicTouch directly, avoiding MediaPipe's headless GPU/GL runtime."""
    segmenter = _segmenter()
    if segmenter is None:
        return None

    interpreter, input_index, output_index = segmenter
    rgb = cv2.cvtColor(
        cv2.resize(
            image_bgr,
            (MODEL_TENSOR_SIZE, MODEL_TENSOR_SIZE),
            interpolation=cv2.INTER_LINEAR,
        ),
        cv2.COLOR_BGR2RGB,
    )
    model_input = np.zeros(
        (1, MODEL_TENSOR_SIZE, MODEL_TENSOR_SIZE, 4),
        dtype=np.float32,
    )
    model_input[0, :, :, :3] = rgb.astype(np.float32) / 255.0

    guide = np.zeros(
        (MODEL_TENSOR_SIZE, MODEL_TENSOR_SIZE),
        dtype=np.float32,
    )
    guide_x = int(round(normalized_x * (MODEL_TENSOR_SIZE - 1)))
    guide_y = int(round(0.50 * (MODEL_TENSOR_SIZE - 1)))
    cv2.circle(guide, (guide_x, guide_y), 1, 1.0, -1)
    model_input[0, :, :, 3] = guide

    with _SEGMENTER_LOCK:
        interpreter.set_tensor(input_index, model_input)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index).copy()

    foreground = np.squeeze(prediction).astype(np.float32)
    height, width = image_bgr.shape[:2]
    return cv2.resize(
        foreground,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


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
    """Prompt the lightweight model at the center and two flanking locations."""
    if _segmenter() is None:
        return None, None, 0.0

    height, width = image_bgr.shape[:2]

    best_mask = None
    best_point = None
    best_confidence = 0.0
    best_score = -1e6

    for normalized_x in (0.50, 0.35, 0.65):
        point = (round(normalized_x * width), round(0.50 * height))

        try:
            confidence_map = _predict_foreground(image_bgr, normalized_x)
        except Exception:
            logger.exception(
                "MagicTouch inference failed for prompt x=%.2f",
                normalized_x,
            )
            continue
        if confidence_map is None:
            continue

        candidate = _component_at(confidence_map >= 0.50, point)
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

    if best_score <= -1e5:
        return None, None, 0.0
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


def segment_arm(image_bgr):
    """Return a full-resolution arm mask without any guide-shaped fallback."""
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
        logger.exception(
            "Arm model unavailable; continuing with the complete input frame"
        )
        broad, point, confidence = None, None, 0.0
    if broad is None or point is None:
        mask = np.full(working.shape[:2], 255, dtype=np.uint8)
        method = "full-image-no-segmentation"
        confidence = 0.0
        model_name = "No arm-isolation model result"
        model_tier = "Fallback"
        runtime = "CPU"
        fallback_used = True
    else:
        mask = _refine_broad_mask(working, broad, point)
        method = "magic-touch-litert-refined"
        model_name = "Google MagicTouch"
        model_tier = "Lightweight"
        runtime = "LiteRT CPU"
        fallback_used = False

    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    if float(np.mean(mask > 0)) < 0.025:
        mask = np.full((height, width), 255, dtype=np.uint8)
        method = "full-image-invalid-model-mask"
        confidence = 0.0
        model_name = "No arm-isolation model result"
        model_tier = "Fallback"
        runtime = "CPU"
        fallback_used = True

    return ArmSegmentation(
        mask=mask,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        method=method,
        model_name=model_name,
        model_tier=model_tier,
        runtime=runtime,
        fallback_used=fallback_used,
    )
