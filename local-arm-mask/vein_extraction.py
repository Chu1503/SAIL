"""Vein segmentation via CUBITAL's pretrained U-Net (940nm NIR, forearm-specific).

Preprocessing matches CUBITAL's own inference code exactly: grayscale -> CLAHE
(clipLimit=2.0, tileGridSize=8x8) -> resize 512x512 -> normalize to [-1, 1].
See https://github.com/EdwinTSalcedo/CUBITAL/blob/master/edgeai/final_interface_vein_segmentation.py

Runs on the *original* (unmasked) capture, matching what the model saw during
training -- feeding it a background-blanked image would put an artificial
black/skin edge in front of a model that never saw one. The SAM2 arm mask is
applied to the model's output afterward, guaranteeing nothing outside the
confirmed arm region is ever reported as a vein.
"""

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import tensorflow as tf
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from vessel_detection import _prune_short_spurs, _topology_transitions  # noqa: E402

MODEL_INPUT_SIZE = 512
VEIN_CLASS_INDEX = 2
HYSTERESIS_LOW = 0.25
HYSTERESIS_HIGH = 0.50


@dataclass(frozen=True)
class VeinResult:
    skeleton: np.ndarray
    endpoints: np.ndarray
    junctions: np.ndarray
    response: np.ndarray
    connected_vessels: int
    segment_count: int
    coverage: float
    confidence: float


def configure_gpu_memory_growth():
    """Avoid TensorFlow pre-allocating all VRAM, which would starve PyTorch/SAM2
    sharing the same (likely small) GPU."""
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def load_vein_model(path):
    return tf.keras.models.load_model(path)


def _cubital_preprocess(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    resized = cv2.resize(
        enhanced, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA
    )
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    return normalized[np.newaxis, :, :, np.newaxis]


def vein_probability_map(model, image_bgr):
    """Run CUBITAL and return the vein-class softmax channel at full resolution."""
    height, width = image_bgr.shape[:2]
    model_input = _cubital_preprocess(image_bgr)
    prediction = model.predict(model_input, verbose=0)
    vein_channel = np.squeeze(prediction)[:, :, VEIN_CLASS_INDEX]
    return cv2.resize(vein_channel, (width, height), interpolation=cv2.INTER_LINEAR)


def extract_veins(model, image_bgr, arm_mask):
    """Return a VeinResult restricted to arm_mask (a bool/0-255 array)."""
    response = vein_probability_map(model, image_bgr)
    valid = arm_mask.astype(bool)
    response = response * valid

    empty = np.zeros(response.shape, dtype=bool)
    values = response[valid & (response > 0)]
    if values.size < 32:
        return VeinResult(empty, empty, empty, response, 0, 0, 0.0, 0.0)

    candidates = apply_hysteresis_threshold(response, HYSTERESIS_LOW, HYSTERESIS_HIGH)
    candidates &= valid
    candidates = cv2.morphologyEx(
        candidates.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)

    diagonal = float(np.hypot(*response.shape))
    minimum_area = max(7, round(diagonal * 0.003))
    minimum_length = max(10, round(diagonal * 0.006))
    candidates = remove_small_objects(candidates, min_size=minimum_area)
    candidates = remove_small_holes(candidates, area_threshold=max(12, minimum_area // 2))

    skeleton = skeletonize(candidates)
    skeleton = _prune_short_spurs(skeleton, max(5, round(diagonal * 0.004)))

    graph_labels_count, graph_labels = cv2.connectedComponents(
        skeleton.astype(np.uint8), connectivity=8
    )
    surviving = np.zeros_like(skeleton)
    for label in range(1, graph_labels_count):
        component = graph_labels == label
        if int(component.sum()) >= minimum_length:
            surviving |= component
    skeleton = surviving

    topology = _topology_transitions(skeleton)
    endpoints = skeleton & (topology == 1)
    junctions = skeleton & (topology >= 3)

    connected_vessels = max(
        0, cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)[0] - 1
    )
    segment_seed = skeleton & ~cv2.dilate(
        junctions.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    segment_count = max(
        0, cv2.connectedComponents(segment_seed.astype(np.uint8), connectivity=8)[0] - 1
    )

    valid_count = max(1, int(valid.sum()))
    coverage = float(skeleton.sum() / valid_count)
    confidence = float(response[skeleton].mean()) if np.any(skeleton) else 0.0

    return VeinResult(
        skeleton=skeleton,
        endpoints=endpoints,
        junctions=junctions,
        response=response,
        connected_vessels=connected_vessels,
        segment_count=segment_count,
        coverage=coverage,
        confidence=confidence,
    )
