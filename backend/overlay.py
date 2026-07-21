# Paints a binary vein mask onto a display image as a glowing green "laser" overlay.
import cv2
import numpy as np

OVERLAY = {
    "green_color": (0, 255, 0),
    "base_dim": 0.85,
    "glow_sigma": 4.0,
    "glow_gain": 1.0,
    "core_opacity": 0.95,
}


def make_overlay(gray_display, mask):
    p = OVERLAY
    base = cv2.cvtColor(gray_display, cv2.COLOR_GRAY2BGR).astype(np.float32) * p["base_dim"]
    green = np.zeros_like(base)
    green[mask > 0] = p["green_color"]
    glow = cv2.GaussianBlur(green, (0, 0), sigmaX=p["glow_sigma"]) * p["glow_gain"]
    out = base + glow
    sel = mask > 0
    for c in range(3):
        ch = out[..., c]
        ch[sel] = (1.0 - p["core_opacity"]) * ch[sel] + p["core_opacity"] * p["green_color"][c]
    return np.clip(out, 0, 255).astype(np.uint8)
