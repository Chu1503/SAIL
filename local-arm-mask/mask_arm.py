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


def _keep_significant_components(mask, min_area_fraction=0.02):
    """Keep every connected piece large enough to plausibly be arm/hand,
    instead of restricting to one -- a watch band or shadow can legitimately
    split the arm into separate pieces (e.g. hand vs forearm), and vein
    extraction should still run across all of them, not just whichever piece
    happens to contain the prompt point. Small fragments (a stray watch
    reflection, model speckle) are dropped."""
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask

    total_area = mask.size
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] / total_area >= min_area_fraction:
            keep |= labels == label
    return keep


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
            significant = _keep_significant_components(mask.astype(bool))
            candidate_score = _candidate_score(significant, score)
            if candidate_score > best_score:
                best_mask, best_score = significant, candidate_score

    return best_mask


def mask_arm(predictor, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = best_arm_mask(predictor, image_rgb)

    output = np.zeros_like(image_bgr)
    if mask is not None:
        output[mask] = image_bgr[mask]
    return output, mask


def load_predictor(checkpoint, model_cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(model_cfg, checkpoint, device=device)
    return SAM2ImagePredictor(model), device


def inference_autocast(device):
    if device == "cuda":
        return torch.autocast(device, dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


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

    predictor, device = load_predictor(args.checkpoint, args.model_cfg)
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode(), inference_autocast(device):
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
