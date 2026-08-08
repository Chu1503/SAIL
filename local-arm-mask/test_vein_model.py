"""Standalone CUBITAL sanity check -- no SAM2/PyTorch needed.

Run this in the isolated CPU-only TensorFlow venv to validate the model and
preprocessing before wiring it into the full GPU server.

Usage:
    python test_vein_model.py <image_path> [--model models/unet.keras] \
        [--mask path/to/arm_only.png] [--out-dir out]

Omit --mask to see CUBITAL's raw, unrestricted output across the whole frame
(useful for judging whether the model itself avoids background clutter).
Pass --mask <one of the arm_only.png files from mask_arm.py> to see the
result after the SAM2 arm-mask restriction, same as the real pipeline.
"""

import argparse
import os
import sys

import cv2
import numpy as np

from vein_extraction import extract_veins, load_vein_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from overlay import make_graph_mask, make_overlay  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model", default="models/unet.keras")
    parser.add_argument("--mask", default=None)
    parser.add_argument("--out-dir", default="out")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read {args.image}")

    if args.mask:
        mask_image = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            raise SystemExit(f"Could not read {args.mask}")
        arm_mask = mask_image > 0
    else:
        arm_mask = np.ones(image.shape[:2], dtype=bool)

    print("Loading CUBITAL model (CPU)...")
    model = load_vein_model(args.model)

    print("Running inference...")
    result = extract_veins(model, image, arm_mask)

    print(f"Connected vessels: {result.connected_vessels}")
    print(f"Segments:          {result.segment_count}")
    print(f"Coverage:          {result.coverage:.5f}")
    print(f"Confidence:        {result.confidence:.3f}")

    overlay = make_overlay(image, result.skeleton, result.junctions)
    graph = make_graph_mask(result.skeleton, result.endpoints, result.junctions)
    response_vis = (np.clip(result.response, 0, 1) * 255).astype(np.uint8)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_overlay.png"), overlay)
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_graph.png"), graph)
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_raw_response.png"), response_vis)

    print(f"Saved {stem}_overlay.png, {stem}_graph.png, {stem}_raw_response.png to {args.out_dir}/")


if __name__ == "__main__":
    main()
