# NIR enhancement chain: takes a raw BGR frame and returns (original_gray, detection_base, display_base).
import cv2
import numpy as np
from skimage.filters import frangi
from skimage import exposure


def auto_crop_to_roi(gray, thresh_ratio=0.08, pad=15):
    thresh_val = max(10, int(255 * thresh_ratio))
    mask = gray > thresh_val
    coords = np.argwhere(mask)
    if coords.size == 0:
        return gray, (0, 0, gray.shape[1], gray.shape[0])
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(gray.shape[0], y1 + pad)
    x1 = min(gray.shape[1], x1 + pad)
    return gray[y0:y1, x0:x1], (int(x0), int(y0), int(x1), int(y1))


def denoise(gray):
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def apply_clahe(gray, clip_limit=3.0, tile_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def vessel_enhance(gray):
    norm = gray.astype(np.float64) / 255.0
    inverted = 1.0 - norm
    vessels = frangi(inverted, sigmas=range(1, 6), black_ridges=False)
    return exposure.rescale_intensity(vessels, out_range=(0, 255)).astype(np.uint8)


def unsharp_mask(gray, amount=1.0, sigma=2):
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cropped, _bbox = auto_crop_to_roi(gray)
    denoised = denoise(cropped)
    clahe_img = apply_clahe(denoised)
    vessels = vessel_enhance(clahe_img)
    blended = cv2.addWeighted(clahe_img, 0.6, vessels, 0.4, 0)
    final = unsharp_mask(blended, amount=1.0, sigma=2)
    return cropped, clahe_img, final
