# PLACEHOLDER MODEL: classical Frangi vein detection returning a binary mask. Replace predict() with your trained model's inference later.
import cv2
import numpy as np
from skimage.filters import frangi

DETECT = {
    "black_ridges": True,
    "pre_blur": 1.0,
    "scale_min": 1.0,
    "scale_max": 6.0,
    "scale_step": 1.0,
    "rel_threshold": 0.20,
    "close_ksize": 3,
    "max_straightness": 10.0,
    "max_halfwidth": 22,
    "min_area": 60,
}


def _filter_shapes(mask, max_straightness, max_halfwidth, min_area):
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(lbl == i)
        if float(dist[ys, xs].max()) > max_halfwidth:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        pts -= pts.mean(0)
        if len(pts) > 2:
            ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
            straightness = float(np.sqrt((ev[0] + 1e-6) / (ev[1] + 1e-6)))
            if straightness > max_straightness:
                continue
        out[lbl == i] = 255
    return out


def predict(gray):
    p = DETECT
    g = cv2.GaussianBlur(gray, (0, 0), sigmaX=p["pre_blur"]) if p["pre_blur"] > 0 else gray
    f = g.astype(np.float32) / 255.0
    sigmas = np.arange(p["scale_min"], p["scale_max"] + 1e-6, p["scale_step"])
    vness = frangi(f, sigmas=sigmas, black_ridges=p["black_ridges"])
    vness = np.nan_to_num(vness, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip(vness / (np.percentile(vness, 99.5) + 1e-8), 0.0, 1.0)
    mask = (norm >= p["rel_threshold"]).astype(np.uint8) * 255
    if p["close_ksize"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["close_ksize"], p["close_ksize"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return _filter_shapes(mask, p["max_straightness"], p["max_halfwidth"], p["min_area"])
