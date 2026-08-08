"""Isolate the forearm in an 850nm NIR capture and blank everything else.

Runs Meta's SAM2 locally (GPU-accelerated) prompted at a handful of
candidate points across the frame, then keeps whichever candidate mask
looks most like an isolated forearm (elongated, spans a large fraction
of the frame, not a tiny or near-full-frame blob).

Usage:
    python mask_arm.py --checkpoint <path/to/sam2.1_hiera_base_plus.pt> \
        --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
        input.jpg [input2.jpg ...] --out-dir out/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _shape_metrics(mask):
    ys, xs = np.where(mask)
    if xs.size < 8:
        return 0.0, 0.0, 0.0
    points = np.column_stack((xs, ys)).astype(np.float32)
    eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
    elongation = float(np.sqrt((eigenvalues[-1] + 1e-6) / (eigenvalues[0] + 1e-6)))
    height, width = mask.shape
    span = max(
        float((xs.max() - xs.min() + 1) / width),
        float((ys.max() - ys.min() + 1) / height),
    )
    return float(mask.mean()), elongation, span


def _candidate_score(mask, sam_score):
    """Prefer elongated, frame-spanning masks over specks or near-full-frame blobs."""
    area, elongation, span = _shape_metrics(mask)
    if area < 0.03 or area > 0.75 or span < 0.20:
        return -1e6
    return 0.40 * float(sam_score) + 1.00 * min(elongation / 4.0, 1.0) + 0.80 * min(span, 1.0)


def best_arm_mask(predictor, image_rgb):
    height, width = image_rgb.shape[:2]
    predictor.set_image(image_rgb)

    best_mask, best_score = None, -1e6
    for normalized_x in (0.50, 0.35, 0.65):
        point = np.array([[normalized_x * width, 0.50 * height]])
        masks, scores, _ = predictor.predict(
            point_coords=point,
            point_labels=np.array([1]),
            multimask_output=True,
        )
        for mask, score in zip(masks, scores):
            mask_bool = mask.astype(bool)
            candidate_score = _candidate_score(mask_bool, score)
            if candidate_score > best_score:
                best_mask, best_score = mask_bool, candidate_score

    return best_mask


def mask_arm(predictor, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = best_arm_mask(predictor, image_rgb)

    output = np.zeros_like(image_bgr)
    if mask is not None:
        output[mask] = image_bgr[mask]
    return output, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Input image path(s)")
    parser.add_argument("--checkpoint", required=True, help="Path to a sam2.1_hiera_*.pt checkpoint")
    parser.add_argument(
        "--model-cfg",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM2 config matching the checkpoint size",
    )
    parser.add_argument("--out-dir", default="out", help="Directory to write masked images to")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = build_sam2(args.model_cfg, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    autocast = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), autocast:
        for image_path in args.images:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            masked, mask = mask_arm(predictor, image_bgr)
            if mask is None:
                print(f"{image_path}: no confident arm mask found")
                continue

            out_path = out_dir / f"{Path(image_path).stem}_arm_only.png"
            cv2.imwrite(str(out_path), masked)
            print(f"{image_path}: kept {mask.mean():.1%} of the frame -> {out_path}")


if __name__ == "__main__":
    main()
