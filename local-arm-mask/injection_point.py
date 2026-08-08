"""Best-injection-point marking: antecubital-fossa-aware vein-segment scoring.

The antecubital fossa (inner elbow crease) is the standard, gold-standard
venipuncture target in real phlebotomy -- the median cubital vein there is
typically large, stable, and superficial. CUBITAL's multi-task SavedModel
(edgeai/models/unet_multi) was trained specifically to locate it, alongside
vein segmentation. We use that model's fossa-coordinate head to bias where
we look, then decompose the vein skeleton into individual segments and score
each by length, straightness, and model confidence -- preferring segments
near the predicted fossa, and falling back to the best segment anywhere on
the arm if nothing qualifies there.

Preprocessing matches the multi-task reference exactly (same as the
single-task vein model): grayscale -> CLAHE(2.0, 8x8) -> resize 512x512 ->
normalize to [-1, 1]. See
https://github.com/EdwinTSalcedo/CUBITAL/blob/master/edgeai/final_interface_multitask.py
"""

import numpy as np
import tensorflow as tf

from vein_extraction import _cubital_preprocess

MINIMUM_SEGMENT_LENGTH_FRACTION = 0.025
FOSSA_SEARCH_RADIUS_FRACTION = 0.22


def load_fossa_model(path):
    saved_model = tf.saved_model.load(path)
    return saved_model.signatures["serving_default"]


def predict_fossa(infer_fn, image_bgr):
    """Return (x, y) in image pixel coordinates and arm angle in degrees."""
    height, width = image_bgr.shape[:2]
    model_input = _cubital_preprocess(image_bgr)
    tensor = tf.convert_to_tensor(model_input, dtype=tf.float32)
    results = infer_fn(tensor)
    values = results["output_1"].numpy()
    x = int(np.clip(float(values[0][0]), 0.0, 1.0) * width)
    y = int(np.clip(float(values[0][1]), 0.0, 1.0) * height)
    angle = float(values[0][2]) * 180.0
    return x, y, angle


def _neighbors(y, x, shape):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                yield ny, nx


def _trace_segments(skeleton, nodes):
    """Decompose the skeleton into pixel paths between node pixels (endpoints
    or junctions), walking each branch once."""
    node_coords = set(map(tuple, np.argwhere(nodes)))
    visited_edges = set()
    segments = []

    for start in node_coords:
        for first_step in _neighbors(start[0], start[1], skeleton.shape):
            if not skeleton[first_step]:
                continue
            edge_key = frozenset((start, first_step))
            if edge_key in visited_edges:
                continue

            path = [start, first_step]
            visited_edges.add(edge_key)
            previous, current = start, first_step

            while current not in node_coords:
                candidates = [
                    n
                    for n in _neighbors(current[0], current[1], skeleton.shape)
                    if skeleton[n]
                    and n != previous
                    and frozenset((current, n)) not in visited_edges
                ]
                if not candidates:
                    break
                next_pixel = candidates[0]
                visited_edges.add(frozenset((current, next_pixel)))
                path.append(next_pixel)
                previous, current = current, next_pixel

            if len(path) >= 2:
                segments.append(path)

    return segments


def _segment_metrics(path, response):
    length_traveled = sum(
        float(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]))
        for i in range(len(path) - 1)
    )
    straight_dist = float(np.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1]))
    straightness = straight_dist / max(length_traveled, 1e-6)
    confidence = float(np.mean([response[y, x] for y, x in path]))
    midpoint = path[len(path) // 2]
    return length_traveled, straightness, confidence, midpoint


def _quality_score(length_traveled, straightness, confidence, diagonal):
    length_score = min(length_traveled / max(diagonal * 0.08, 1e-6), 1.0)
    return 0.40 * length_score + 0.35 * straightness + 0.25 * confidence


def find_injection_point(skeleton, endpoints, junctions, response, fossa_point=None):
    """Return (x, y) pixel coordinates for the recommended injection mark, or
    None if no vein segment is good enough. Prefers segments near the
    predicted antecubital fossa; falls back to the best segment anywhere on
    the arm if nothing usable is found there."""
    nodes = endpoints | junctions
    segments = _trace_segments(skeleton, nodes)
    if not segments:
        return None

    diagonal = float(np.hypot(*skeleton.shape))
    minimum_length = diagonal * MINIMUM_SEGMENT_LENGTH_FRACTION

    scored = []
    for path in segments:
        length_traveled, straightness, confidence, midpoint = _segment_metrics(path, response)
        if length_traveled < minimum_length:
            continue
        score = _quality_score(length_traveled, straightness, confidence, diagonal)
        scored.append((score, midpoint))

    if not scored:
        return None

    if fossa_point is not None:
        fossa_x, fossa_y = fossa_point
        search_radius = diagonal * FOSSA_SEARCH_RADIUS_FRACTION
        near_fossa = [
            (score, midpoint)
            for score, midpoint in scored
            if np.hypot(midpoint[1] - fossa_x, midpoint[0] - fossa_y) <= search_radius
        ]
        if near_fossa:
            _, best_midpoint = max(near_fossa, key=lambda item: item[0])
            best_y, best_x = best_midpoint
            return int(best_x), int(best_y)

    _, best_midpoint = max(scored, key=lambda item: item[0])
    best_y, best_x = best_midpoint
    return int(best_x), int(best_y)
