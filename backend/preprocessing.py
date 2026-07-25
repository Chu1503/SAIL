"""Arm-restricted preprocessing for monochrome reflected-NIR images."""

from dataclasses import dataclass

import cv2
import numpy as np

from arm_segmentation import segment_arm


@dataclass(frozen=True)
class PreparedImage:
    """Images and masks shared by vessel detection and presentation."""

    original: np.ndarray
    gray: np.ndarray
    corrected: np.ndarray
    enhanced: np.ndarray
    vessel_enhanced: np.ndarray
    arm_mask: np.ndarray
    analysis_mask: np.ndarray
    signal_quality: float
    segmentation_confidence: float
    segmentation_method: str
    segmentation_model_name: str
    segmentation_model_tier: str
    segmentation_runtime: str
    segmentation_fallback_used: bool
    warnings: tuple[str, ...]


def _robust_rescale(image, mask=None, low=1.0, high=99.0):
    valid = mask > 0 if mask is not None else np.ones(image.shape, dtype=bool)
    sample = image[valid] if np.any(valid) else image.ravel()
    lo, hi = np.percentile(sample, (low, high))
    if hi <= lo + 1e-6:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image.astype(np.float32) - float(lo)) * (255.0 / float(hi - lo))
    result = np.clip(scaled, 0, 255).astype(np.uint8)
    result[~valid] = 0
    return result


def _interior_mask(arm_mask):
    """Exclude the physical arm boundary where edge filters create ridges."""
    bounded_arm = (arm_mask > 0).astype(np.uint8)
    bounded_arm[[0, -1], :] = 0
    bounded_arm[:, [0, -1]] = 0
    distance = cv2.distanceTransform(
        bounded_arm,
        cv2.DIST_L2,
        5,
    )
    inset = max(8.0, min(arm_mask.shape) * 0.030)
    interior = (distance >= inset).astype(np.uint8) * 255

    # Preserve a useful analysis region for unusually narrow model masks.
    if np.mean(interior > 0) < 0.60 * max(np.mean(arm_mask > 0), 1e-6):
        inset *= 0.55
        interior = (distance >= inset).astype(np.uint8) * 255
    return interior


def _illumination_correct(gray, arm_mask):
    """Normalize vignetting without allowing black background to bias the field."""
    mask_float = (arm_mask > 0).astype(np.float32)
    sigma = max(18.0, min(gray.shape) / 12.0)
    numerator = cv2.GaussianBlur(
        gray.astype(np.float32) * mask_float,
        (0, 0),
        sigmaX=sigma,
    )
    denominator = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=sigma)
    background = numerator / np.maximum(denominator, 1e-3)

    valid = arm_mask > 0
    reference = float(np.median(background[valid])) if np.any(valid) else 128.0
    corrected = (gray.astype(np.float32) + 2.0) / (background + 2.0) * reference
    return _robust_rescale(corrected, arm_mask, low=0.5, high=99.5)


def _multiscale_blackhat(image, support_mask, analysis_mask):
    """Enhance dark lines over the expected range of apparent vessel widths."""
    working = image.copy()
    support = support_mask > 0
    fill = int(np.median(working[support])) if np.any(support) else 128
    working[~support] = fill

    responses = []
    minimum = min(image.shape)
    for fraction in (0.012, 0.020, 0.032, 0.048):
        size = max(7, round(minimum * fraction))
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        responses.append(cv2.morphologyEx(working, cv2.MORPH_BLACKHAT, kernel))

    response = np.maximum.reduce(responses).astype(np.float32)
    return _robust_rescale(response, analysis_mask, low=5.0, high=99.5)


def _quality_metrics(gray, corrected, analysis_mask):
    valid = analysis_mask > 0
    sample = gray[valid] if np.any(valid) else gray.ravel()
    corrected_sample = corrected[valid] if np.any(valid) else corrected.ravel()

    dynamic_range = float(
        np.percentile(corrected_sample, 95) - np.percentile(corrected_sample, 5)
    )
    laplacian = cv2.Laplacian(corrected, cv2.CV_32F)
    sharpness = float(np.var(laplacian[valid])) if np.any(valid) else float(laplacian.var())
    clipped = float(np.mean((sample <= 5) | (sample >= 250)))

    quality = float(
        0.45 * np.clip(dynamic_range / 105.0, 0.0, 1.0)
        + 0.35 * np.clip(sharpness / 180.0, 0.0, 1.0)
        + 0.20 * np.clip(1.0 - clipped / 0.12, 0.0, 1.0)
    )

    warnings = []
    if dynamic_range < 35:
        warnings.append("Low local contrast")
    if sharpness < 25:
        warnings.append("Image may be out of focus")
    if clipped > 0.12:
        warnings.append("Exposure clipping detected")
    return quality, tuple(warnings)


def prepare_image(image_bgr):
    """Isolate the arm, correct the frame, and expose dark tubular structures."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty")

    if image_bgr.ndim == 2:
        raw_original = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    else:
        raw_original = image_bgr[:, :, :3].copy()
    raw_gray = cv2.cvtColor(raw_original, cv2.COLOR_BGR2GRAY)

    segmentation = segment_arm(raw_original)
    arm_mask = segmentation.mask
    analysis_mask = _interior_mask(arm_mask)

    original = np.zeros_like(raw_original)
    original[arm_mask > 0] = raw_original[arm_mask > 0]
    gray = np.zeros_like(raw_gray)
    gray[arm_mask > 0] = raw_gray[arm_mask > 0]

    corrected = _illumination_correct(raw_gray, arm_mask)

    # NLM suppresses OV9281 sensor grain; restrained multiscale unsharp masking
    # restores vessel edges without reintroducing the slowly varying light field.
    filter_input = corrected.copy()
    arm_pixels = arm_mask > 0
    fill = int(np.median(corrected[arm_pixels])) if np.any(arm_pixels) else 128
    filter_input[~arm_pixels] = fill
    denoised = cv2.fastNlMeansDenoising(
        filter_input,
        None,
        h=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    soft = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.25)
    sharpened = cv2.addWeighted(denoised, 1.45, soft, -0.45, 0)
    enhanced = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(10, 10),
    ).apply(sharpened)
    enhanced[arm_mask == 0] = 0

    vessel_enhanced = _multiscale_blackhat(
        enhanced,
        arm_mask,
        analysis_mask,
    )

    quality, warnings = _quality_metrics(gray, corrected, analysis_mask)
    return PreparedImage(
        original=original,
        gray=gray,
        corrected=corrected,
        enhanced=enhanced,
        vessel_enhanced=vessel_enhanced,
        arm_mask=arm_mask,
        analysis_mask=analysis_mask,
        signal_quality=quality,
        segmentation_confidence=segmentation.confidence,
        segmentation_method=segmentation.method,
        segmentation_model_name=segmentation.model_name,
        segmentation_model_tier=segmentation.model_tier,
        segmentation_runtime=segmentation.runtime,
        segmentation_fallback_used=segmentation.fallback_used,
        warnings=warnings,
    )
