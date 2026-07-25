"""Model-guided forearm isolation for close-range NIR camera frames."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # The geometric fallback keeps local diagnostics usable.
    ort = None


MODEL_PATH = Path(__file__).resolve().parent / "models" / "efficientsam_ti.onnx"
MODEL_INPUT_LONG_EDGE = 512
_SESSION = None
_SESSION_LOCK = Lock()


@dataclass(frozen=True)
class ArmSegmentation:
    mask: np.ndarray
    confidence: float
    method: str


def _session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    if ort is None or not MODEL_PATH.exists():
        return None

    with _SESSION_LOCK:
        if _SESSION is None:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 4
            _SESSION = ort.InferenceSession(
                str(MODEL_PATH),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
    return _SESSION


def _component_at(mask, point):
    """Retain the prompted component and discard disconnected model speckle."""
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary

    x, y = point
    label = int(labels[np.clip(y, 0, labels.shape[0] - 1), np.clip(x, 0, labels.shape[1] - 1)])
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


def _candidate_score(mask, model_quality):
    area, elongation, span = _shape_metrics(mask)
    if area < 0.025 or area > 0.28 or elongation < 1.55 or span < 0.24:
        return -1e6

    area_preference = 1.0 - min(1.0, abs(area - 0.12) / 0.16)
    return (
        float(model_quality)
        + 0.95 * min(elongation / 5.5, 1.0)
        + 0.75 * min(span, 1.0)
        + 0.20 * area_preference
    )


def _model_arm_core(image_bgr):
    """Prompt EfficientSAM from both sides of the centered acquisition guide."""
    session = _session()
    if session is None:
        return None, None, 0.0

    height, width = image_bgr.shape[:2]
    scale = MODEL_INPUT_LONG_EDGE / max(height, width)
    small_width = max(32, round(width * scale))
    small_height = max(32, round(height * scale))
    small = cv2.resize(
        image_bgr,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    model_image = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    best_mask = None
    best_broad_mask = None
    best_score = -1e6
    best_quality = 0.0
    # The Arducam frame is landscape and the capture guide crosses the middle.
    # Two prompts avoid selecting a watch or hand when either side is occluded.
    for normalized_x in (0.35, 0.65):
        point = (round(normalized_x * small_width), round(0.50 * small_height))
        point_coords = np.array(
            [[[[float(point[0]), float(point[1])]]]],
            dtype=np.float32,
        )
        point_labels = np.ones((1, 1, 1), dtype=np.float32)

        try:
            with _SESSION_LOCK:
                logits, qualities, *_ = session.run(
                    None,
                    {
                        "batched_images": model_image,
                        "batched_point_coords": point_coords,
                        "batched_point_labels": point_labels,
                    },
                )
        except Exception:
            continue

        broad_candidate = cv2.resize(
            (logits[0, 0, 0] >= 0).astype(np.uint8),
            (small_width, small_height),
            interpolation=cv2.INTER_NEAREST,
        )
        broad_candidate = _component_at(broad_candidate, point)

        for index in range(logits.shape[2]):
            candidate = cv2.resize(
                (logits[0, 0, index] >= 0).astype(np.uint8),
                (small_width, small_height),
                interpolation=cv2.INTER_NEAREST,
            )
            candidate = _component_at(candidate, point)
            quality = float(qualities[0, 0, index])
            score = _candidate_score(candidate, quality)
            if score > best_score:
                best_mask = candidate
                best_broad_mask = broad_candidate
                best_score = score
                best_quality = quality

    return best_mask, best_broad_mask, best_quality


def _refine_core(image_bgr, core, broad):
    """Keep the broad model arm while trimming table spill around its core axis."""
    height, width = core.shape
    core = (core > 0).astype(np.uint8)
    broad = (broad > 0).astype(np.uint8)

    ys, xs = np.where(core > 0)
    if xs.size < 8:
        return core * 255

    points = np.column_stack((xs, ys)).astype(np.float32)
    center = points.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov(points.T))
    minor_axis = eigenvectors[:, 0]

    yy, xx = np.mgrid[:height, :width]
    perpendicular_distance = np.abs(
        (xx - center[0]) * minor_axis[0]
        + (yy - center[1]) * minor_axis[1]
    )
    # This intentionally follows the user's preferred broad proposal. The core
    # establishes only the forearm axis; a generous band preserves the full arm
    # thickness while dropping the proposal's distant floor/table lobe.
    tube_radius = max(18.0, min(height, width) * 0.24)
    arm_tube = perpendicular_distance <= tube_radius

    open_size = max(5, round(min(height, width) * 0.018))
    if open_size % 2 == 0:
        open_size += 1
    broad = cv2.morphologyEx(
        broad,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )
    refined = ((broad > 0) & arm_tube) | (core > 0)
    refined = refined.astype(np.uint8)

    count, labels, _, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    if count > 1:
        overlaps = [
            (int(np.sum((labels == label) & (core > 0))), label)
            for label in range(1, count)
        ]
        selected_label = max(overlaps)[1]
        refined = (labels == selected_label).astype(np.uint8)

    close_size = max(5, round(min(height, width) * 0.015))
    if close_size % 2 == 0:
        close_size += 1
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    return refined * 255


def _fallback_mask(image_bgr):
    """Conservative centered capsule used only when the model is unavailable."""
    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if width >= height:
        axes = (round(width * 0.46), round(height * 0.17))
    else:
        axes = (round(width * 0.17), round(height * 0.46))
    cv2.ellipse(mask, (width // 2, height // 2), axes, 0, 0, 360, 255, -1)
    return mask


def segment_arm(image_bgr):
    """Return a full-resolution binary forearm mask."""
    height, width = image_bgr.shape[:2]
    scale = min(1.0, MODEL_INPUT_LONG_EDGE / max(height, width))
    working = cv2.resize(
        image_bgr,
        (max(32, round(width * scale)), max(32, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

    core, broad, quality = _model_arm_core(working)
    if core is None:
        mask = _fallback_mask(working)
        method = "geometric-fallback"
        quality = 0.0
    else:
        mask = _refine_core(working, core, broad)
        method = "efficientsam-refined"

    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    if float(np.mean(mask > 0)) < 0.025:
        mask = _fallback_mask(image_bgr)
        method = "geometric-fallback"
        quality = 0.0

    return ArmSegmentation(
        mask=mask,
        confidence=float(np.clip(quality, 0.0, 1.0)),
        method=method,
    )
