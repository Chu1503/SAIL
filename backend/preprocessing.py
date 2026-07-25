"""Training-free preprocessing for monochrome reflected-NIR arm images."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreparedImage:
    """Images and masks shared by vessel detection and presentation."""

    original: np.ndarray
    gray: np.ndarray
    corrected: np.ndarray
    enhanced: np.ndarray
    analysis_mask: np.ndarray
    signal_quality: float
    warnings: tuple[str, ...]


def _robust_rescale(image, mask=None, low=1.0, high=99.0):
    sample = image[mask > 0] if mask is not None and np.any(mask) else image.ravel()
    lo, hi = np.percentile(sample, (low, high))
    if hi <= lo + 1e-6:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image.astype(np.float32) - float(lo)) * (255.0 / float(hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _foreground_prior(image_bgr):
    """Fast, low-resolution foreground extraction with a centered-arm prior."""
    height, width = image_bgr.shape[:2]
    scale = min(1.0, 360.0 / max(height, width))
    small = cv2.resize(
        image_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    small_height, small_width = small.shape[:2]
    mask = np.full((small_height, small_width), cv2.GC_PR_BGD, dtype=np.uint8)

    border = max(3, round(min(small_height, small_width) * 0.02))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    if small_width >= small_height:
        probable_axes = (round(small_width * 0.48), round(small_height * 0.23))
        certain_axes = (round(small_width * 0.28), round(small_height * 0.085))
    else:
        probable_axes = (round(small_width * 0.23), round(small_height * 0.48))
        certain_axes = (round(small_width * 0.085), round(small_height * 0.28))

    center = (small_width // 2, small_height // 2)
    cv2.ellipse(mask, center, probable_axes, 0, 0, 360, cv2.GC_PR_FGD, -1)
    cv2.ellipse(mask, center, certain_axes, 0, 0, 360, cv2.GC_FGD, -1)

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            small,
            mask,
            None,
            background_model,
            foreground_model,
            3,
            cv2.GC_INIT_WITH_MASK,
        )
        foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    except cv2.error:
        foreground = np.zeros((small_height, small_width), dtype=np.uint8)
        cv2.ellipse(foreground, center, probable_axes, 0, 0, 360, 255, -1)

    foreground = cv2.resize(
        foreground, (width, height), interpolation=cv2.INTER_NEAREST
    )
    if np.mean(foreground > 0) < 0.06:
        foreground.fill(0)
        if width >= height:
            axes = (round(width * 0.46), round(height * 0.21))
        else:
            axes = (round(width * 0.21), round(height * 0.46))
        cv2.ellipse(foreground, (width // 2, height // 2), axes, 0, 0, 360, 255, -1)

    # Acquisition is intentionally centered. This geometric guard prevents a
    # foreground classifier from expanding into walls, tabletops, or clothing
    # with luminance similar to skin.
    centered_guard = np.zeros((height, width), dtype=np.uint8)
    if width >= height:
        guard_axes = (round(width * 0.47), round(height * 0.23))
    else:
        guard_axes = (round(width * 0.23), round(height * 0.47))
    cv2.ellipse(
        centered_guard,
        (width // 2, height // 2),
        guard_axes,
        0,
        0,
        360,
        255,
        -1,
    )
    return cv2.bitwise_and(foreground, centered_guard)


def _analysis_region(image_bgr, gray):
    """Keep centered plausible tissue while suppressing room edges and hardware."""
    height, width = gray.shape
    margin = max(8, round(min(height, width) * 0.018))

    safe = np.zeros_like(gray, dtype=np.uint8)
    safe[margin : height - margin, margin : width - margin] = 255

    foreground = _foreground_prior(image_bgr)
    foreground_sample = foreground > 0
    smoothed = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=max(3.0, min(height, width) / 100)
    )
    low_cutoff = max(
        8.0,
        float(np.percentile(smoothed[foreground_sample], 10))
        if np.any(foreground_sample)
        else 8.0,
    )
    tissue = (smoothed > low_cutoff).astype(np.uint8) * 255

    close_size = max(5, round(min(height, width) * 0.008))
    if close_size % 2 == 0:
        close_size += 1
    tissue = cv2.morphologyEx(
        tissue,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )

    # Hardware, jewelry, and the image edge create strong false ridges. Eroding a
    # small amount keeps the detector focused on the interior of tissue regions.
    erode_size = max(15, round(min(height, width) * 0.060))
    if erode_size % 2 == 0:
        erode_size += 1
    tissue = cv2.erode(
        tissue,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size)),
    )

    region = cv2.bitwise_and(cv2.bitwise_and(tissue, foreground), safe)

    # Suppress the neighborhood of the strongest physical edges (watch bands,
    # jewelry, clothing, table seams). Vein contrast is normally substantially
    # softer than these object boundaries in reflected NIR.
    gradient_source = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.4)
    gradient_x = cv2.Sobel(gradient_source, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gradient_source, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    foreground_gradients = gradient[foreground > 0]
    strong_cutoff = (
        max(28.0, float(np.percentile(foreground_gradients, 97.0)))
        if foreground_gradients.size
        else 28.0
    )
    strong_edges = (gradient >= strong_cutoff).astype(np.uint8)
    edge_guard_size = max(9, round(min(height, width) * 0.022))
    if edge_guard_size % 2 == 0:
        edge_guard_size += 1
    edge_guard = cv2.dilate(
        strong_edges,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (edge_guard_size, edge_guard_size)
        ),
    )
    region[edge_guard > 0] = 0
    return region


def _illumination_correct(gray, analysis_mask):
    """Divide out slowly varying NIR illumination and camera vignetting."""
    sigma = max(18.0, min(gray.shape) / 14.0)
    background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=sigma)

    valid = analysis_mask > 0
    reference = float(np.median(background[valid])) if np.any(valid) else 128.0
    corrected = (gray.astype(np.float32) + 1.0) / (background + 1.0) * reference
    return _robust_rescale(corrected, analysis_mask, low=1.0, high=99.0)


def _quality_metrics(gray, corrected, analysis_mask):
    valid = analysis_mask > 0
    sample = gray[valid] if np.any(valid) else gray.ravel()
    corrected_sample = corrected[valid] if np.any(valid) else corrected.ravel()

    dynamic_range = float(np.percentile(corrected_sample, 95) - np.percentile(corrected_sample, 5))
    sharpness = float(cv2.Laplacian(corrected, cv2.CV_32F).var())
    clipped = float(np.mean((sample <= 5) | (sample >= 250)))

    contrast_score = np.clip(dynamic_range / 115.0, 0.0, 1.0)
    sharpness_score = np.clip(sharpness / 240.0, 0.0, 1.0)
    exposure_score = np.clip(1.0 - clipped / 0.12, 0.0, 1.0)
    quality = float(0.45 * contrast_score + 0.35 * sharpness_score + 0.20 * exposure_score)

    warnings = []
    if dynamic_range < 45:
        warnings.append("Low local contrast")
    if sharpness < 35:
        warnings.append("Image may be out of focus")
    if clipped > 0.12:
        warnings.append("Exposure clipping detected")
    return quality, tuple(warnings)


def prepare_image(image_bgr):
    """Prepare an OV9281/UVC frame for dark tubular-structure detection."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty")

    if image_bgr.ndim == 2:
        original = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        gray = image_bgr.copy()
    else:
        original = image_bgr[:, :, :3].copy()
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    analysis_mask = _analysis_region(original, gray)
    corrected = _illumination_correct(gray, analysis_mask)

    # Edge-preserving denoising removes UVC sensor speckle without erasing the
    # several-pixel-wide ridges expected from superficial vessels.
    denoised = cv2.bilateralFilter(corrected, d=7, sigmaColor=28, sigmaSpace=7)
    enhanced = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(10, 10)).apply(denoised)

    quality, warnings = _quality_metrics(gray, corrected, analysis_mask)
    return PreparedImage(
        original=original,
        gray=gray,
        corrected=corrected,
        enhanced=enhanced,
        analysis_mask=analysis_mask,
        signal_quality=quality,
        warnings=warnings,
    )
