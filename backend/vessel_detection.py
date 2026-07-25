"""Training-free multiscale vessel segmentation and centerline graph extraction."""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold, frangi, meijering, sato
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize


@dataclass(frozen=True)
class VesselGraph:
    response: np.ndarray
    region_mask: np.ndarray
    skeleton: np.ndarray
    endpoints: np.ndarray
    junctions: np.ndarray
    connected_vessels: int
    segment_count: int
    coverage: float
    confidence: float


def _normalize_response(response, valid):
    sample = response[valid]
    positive = sample[sample > 0]
    if positive.size < 32:
        return np.zeros(response.shape, dtype=np.float32)

    lo, hi = np.percentile(positive, (45.0, 99.7))
    if hi <= lo + 1e-10:
        return np.zeros(response.shape, dtype=np.float32)
    return np.clip((response - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _topology_transitions(skeleton):
    """Count 0→1 transitions around each pixel (crossing-number topology)."""
    padded = np.pad(skeleton.astype(bool), 1)
    neighbors = [
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
        padded[:-2, :-2],
    ]
    transitions = np.zeros(skeleton.shape, dtype=np.uint8)
    for index, current in enumerate(neighbors):
        following = neighbors[(index + 1) % len(neighbors)]
        transitions += (~current & following).astype(np.uint8)
    return transitions


def _prune_short_spurs(skeleton, minimum_length):
    """Remove short endpoint-to-junction twigs while preserving main paths."""
    work = skeleton.astype(bool).copy()
    height, width = work.shape

    for _ in range(12):
        topology = _topology_transitions(work)
        endpoint_coords = np.argwhere(work & (topology == 1))
        removals = []

        for start_y, start_x in endpoint_coords:
            if not work[start_y, start_x]:
                continue

            path = [(int(start_y), int(start_x))]
            previous = None
            current = path[0]

            while len(path) <= minimum_length:
                y, x = current
                neighbors = []
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if (ny, nx) == current or not work[ny, nx]:
                            continue
                        if previous is not None and (ny, nx) == previous:
                            continue
                        neighbors.append((ny, nx))

                if len(neighbors) != 1:
                    break
                previous, current = current, neighbors[0]
                path.append(current)
                if topology[current] != 2:
                    break

            terminal_topology = int(topology[current])
            if terminal_topology >= 3 and len(path) <= minimum_length:
                removals.extend(path[:-1])

        if not removals:
            break
        for y, x in removals:
            work[y, x] = False

    return work


def _filter_vessel_shapes(mask, response, minimum_length, maximum_width):
    labels_count, labels, _, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    kept = np.zeros(mask.shape, dtype=bool)

    for label in range(1, labels_count):
        component = labels == label
        centerline = skeletonize(component)
        length = int(centerline.sum())
        if length < minimum_length:
            continue

        topology = _topology_transitions(centerline)
        endpoint_count = (
            cv2.connectedComponents(
                (centerline & (topology == 1)).astype(np.uint8),
                connectivity=8,
            )[0]
            - 1
        )
        junction_count = (
            cv2.connectedComponents(
                (centerline & (topology >= 3)).astype(np.uint8),
                connectivity=8,
            )[0]
            - 1
        )
        if endpoint_count == 0:
            continue
        if junction_count > max(4, round(length / 55)):
            continue

        ys, xs = np.where(centerline)
        if xs.size < 3:
            continue

        points = np.column_stack((xs, ys)).astype(np.float32)
        centered = points - points.mean(axis=0)
        eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(centered.T)))[::-1]
        elongation = float(np.sqrt((eigenvalues[0] + 1e-6) / (eigenvalues[1] + 1e-6)))

        distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        median_width = float(2.0 * np.median(distance[centerline]))
        mean_response = float(response[centerline].mean())

        # Broad blobs and near-circular highlights are not vessel-like. A
        # branching component may have modest global elongation, so strong
        # response can compensate for that geometric test.
        if median_width > maximum_width:
            continue
        if elongation < 1.10 and mean_response < 0.34:
            continue
        kept |= component

    return kept


def _cluster_centers(pixel_mask):
    count, labels, _, centroids = cv2.connectedComponentsWithStats(
        pixel_mask.astype(np.uint8), connectivity=8
    )
    points = []
    for label in range(1, count):
        x, y = centroids[label]
        points.append((round(float(x)), round(float(y))))
    return points


def detect_vessel_graph(enhanced, corrected, blackhat, analysis_mask):
    valid = analysis_mask > 0
    image = enhanced.astype(np.float32) / 255.0

    # OV9281/NIR veins should be dark ridges. The two Hessian measures respond
    # differently at bifurcations, so their fusion is more stable than either
    # one alone without requiring learned weights.
    scales = (1.1, 1.6, 2.3, 3.3, 4.7, 6.6, 9.0, 12.0)
    frangi_response = frangi(
        image,
        sigmas=scales,
        alpha=0.5,
        beta=0.5,
        black_ridges=True,
    )
    sato_response = sato(image, sigmas=scales, black_ridges=True)
    meijering_response = meijering(image, sigmas=scales, black_ridges=True)
    frangi_response = np.nan_to_num(frangi_response)
    sato_response = np.nan_to_num(sato_response)
    meijering_response = np.nan_to_num(meijering_response)

    frangi_norm = _normalize_response(frangi_response, valid)
    sato_norm = _normalize_response(sato_response, valid)
    meijering_norm = _normalize_response(meijering_response, valid)

    local_background = cv2.GaussianBlur(
        corrected.astype(np.float32), (0, 0), sigmaX=7.0
    )
    dark_contrast = np.maximum(local_background - corrected.astype(np.float32), 0.0)
    dark_norm = _normalize_response(dark_contrast, valid)

    sobel_x = cv2.Sobel(corrected, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(corrected, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient_norm = _normalize_response(gradient, valid)

    blackhat_norm = blackhat.astype(np.float32) / 255.0
    blackhat_norm *= valid.astype(np.float32)

    tubular = np.maximum.reduce(
        (frangi_norm, 0.82 * sato_norm, 0.88 * meijering_norm)
    )
    response = (
        0.50 * tubular
        + 0.32 * blackhat_norm
        + 0.18 * dark_norm
    )
    # Only the very strongest first-order edges are down-weighted. The learned
    # arm boundary and inset mask already remove background edges, so retaining
    # softer gradients materially improves sensitivity to faint branches.
    edge_penalty = np.clip((gradient_norm - 0.78) / 0.22, 0.0, 1.0)
    response *= 1.0 - 0.55 * edge_penalty
    response *= valid.astype(np.float32)

    values = response[valid & (response > 0)]
    if values.size < 32:
        empty = np.zeros(response.shape, dtype=bool)
        return VesselGraph(response, empty, empty, empty, empty, 0, 0, 0.0, 0.0)

    high = max(0.12, float(np.percentile(values, 76.0)))
    low = max(0.038, high * 0.30)
    candidates = apply_hysteresis_threshold(response, low, high)
    candidates &= valid

    candidates = cv2.morphologyEx(
        candidates.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)

    diagonal = float(np.hypot(*enhanced.shape))
    minimum_area = max(7, round(diagonal * 0.003))
    minimum_length = max(10, round(diagonal * 0.006))
    candidates = remove_small_objects(candidates, min_size=minimum_area)
    candidates = remove_small_holes(candidates, area_threshold=max(12, minimum_area // 2))
    maximum_width = max(18.0, min(enhanced.shape) * 0.065)
    region_mask = _filter_vessel_shapes(
        candidates,
        response,
        minimum_length,
        maximum_width,
    )

    # The multiscale black-hat map preserves low-contrast vessels that may not
    # have a textbook Hessian profile. Hysteresis grows only structures attached
    # to a strong dark-line seed, giving the requested high-recall map without
    # reintroducing anything outside the arm.
    blackhat_values = blackhat_norm[valid]
    blackhat_low = float(np.percentile(blackhat_values, 80.0))
    blackhat_high = float(np.percentile(blackhat_values, 90.0))
    long_vessels = apply_hysteresis_threshold(
        blackhat_norm,
        blackhat_low,
        blackhat_high,
    )
    long_vessels &= valid
    long_vessels = cv2.morphologyEx(
        long_vessels.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    long_vessels = remove_small_objects(
        long_vessels,
        min_size=max(24, round(diagonal * 0.015)),
    )
    region_mask |= long_vessels

    skeleton = skeletonize(region_mask)
    skeleton = _prune_short_spurs(skeleton, max(5, round(diagonal * 0.004)))

    # Rebuild the region mask from only components whose final graph survived.
    graph_labels_count, graph_labels = cv2.connectedComponents(
        skeleton.astype(np.uint8), connectivity=8
    )
    surviving = np.zeros_like(region_mask)
    for label in range(1, graph_labels_count):
        component = graph_labels == label
        if int(component.sum()) >= minimum_length:
            surviving |= component
    skeleton = surviving

    topology = _topology_transitions(skeleton)
    endpoints = skeleton & (topology == 1)
    junctions = skeleton & (topology >= 3)
    endpoint_points = _cluster_centers(endpoints)
    junction_points = _cluster_centers(junctions)

    connected_vessels = max(
        0,
        cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)[0] - 1,
    )
    segment_seed = skeleton & ~cv2.dilate(
        junctions.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    segment_count = max(
        0,
        cv2.connectedComponents(segment_seed.astype(np.uint8), connectivity=8)[0]
        - 1,
    )

    valid_count = max(1, int(valid.sum()))
    coverage = float(skeleton.sum() / valid_count)
    confidence = (
        float(response[skeleton].mean()) if np.any(skeleton) else 0.0
    )

    # Store one representative pixel per topology node for clean rendering.
    endpoints_clean = np.zeros_like(skeleton)
    junctions_clean = np.zeros_like(skeleton)
    for x, y in endpoint_points:
        endpoints_clean[y, x] = True
    for x, y in junction_points:
        junctions_clean[y, x] = True

    return VesselGraph(
        response=response,
        region_mask=region_mask,
        skeleton=skeleton,
        endpoints=endpoints_clean,
        junctions=junctions_clean,
        connected_vessels=connected_vessels,
        segment_count=segment_count,
        coverage=coverage,
        confidence=confidence,
    )
