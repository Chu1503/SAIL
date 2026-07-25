"""Presentation renderers for probable vessel centerlines and topology."""

import cv2
import numpy as np


def _scaled_radius(shape, fraction, minimum):
    return max(minimum, round(min(shape) * fraction))


def make_overlay(original_bgr, skeleton, junctions):
    """Draw a clean neon centerline overlay on the isolated input."""
    base = original_bgr.copy()
    line_radius = _scaled_radius(base.shape[:2], 0.0022, 1)
    core = cv2.dilate(
        skeleton.astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (line_radius * 2 + 1, line_radius * 2 + 1)
        ),
    )

    glow = cv2.GaussianBlur(core, (0, 0), sigmaX=max(2.5, line_radius * 1.8))
    glow_alpha = (glow.astype(np.float32) / 255.0 * 0.62)[..., None]
    green = np.zeros_like(base, dtype=np.float32)
    green[..., 0] = 95
    green[..., 1] = 255
    green[..., 2] = 118

    output = base.astype(np.float32) * (1.0 - glow_alpha) + green * glow_alpha
    output[core > 0] = (62, 255, 104)

    return np.clip(output, 0, 255).astype(np.uint8)


def make_graph_mask(skeleton, endpoints, junctions):
    """Render a standalone topology map: paths, endpoints, and junctions."""
    height, width = skeleton.shape
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    line_radius = _scaled_radius((height, width), 0.0016, 1)
    paths = cv2.dilate(
        skeleton.astype(np.uint8) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (line_radius * 2 + 1, line_radius * 2 + 1)
        ),
    )
    graph[paths > 0] = (86, 245, 118)

    endpoint_radius = _scaled_radius((height, width), 0.004, 3)
    junction_radius = _scaled_radius((height, width), 0.005, 4)
    for y, x in np.argwhere(endpoints):
        cv2.circle(graph, (int(x), int(y)), endpoint_radius, (255, 185, 70), -1)
    for y, x in np.argwhere(junctions):
        cv2.circle(graph, (int(x), int(y)), junction_radius, (0, 190, 255), -1)
        cv2.circle(graph, (int(x), int(y)), max(1, junction_radius // 3), (255, 255, 255), -1)
    return graph
