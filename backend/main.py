# FastAPI server for training-free superficial-vessel visualization.
import base64
import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from preprocessing import prepare_image
from vessel_detection import detect_vessel_graph
from overlay import make_graph_mask, make_overlay

app = FastAPI(title="VEINZ API")
logger = logging.getLogger("veinz")

allowed_origins = [
    "http://localhost:3000",
    "https://localhost",
    "http://localhost",
    "capacitor://localhost",
    "https://sail-kohl.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _to_data_url(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result image")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _opencv_fallback_visualization(img):
    """Produce a real low-memory vessel overlay without model/skimage dependencies."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.1)
    illumination = cv2.GaussianBlur(
        denoised.astype(np.float32),
        (0, 0),
        sigmaX=max(18.0, min(gray.shape) / 12.0),
    )
    reference = float(np.median(illumination))
    corrected_float = (
        (denoised.astype(np.float32) + 2.0)
        / (illumination + 2.0)
        * reference
    )
    lo, hi = np.percentile(corrected_float, (0.5, 99.5))
    if hi <= lo + 1e-6:
        corrected = gray
    else:
        corrected = np.clip(
            (corrected_float - lo) * (255.0 / (hi - lo)),
            0,
            255,
        ).astype(np.uint8)

    processed = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(10, 10),
    ).apply(corrected)

    responses = []
    minimum = min(processed.shape)
    for fraction in (0.012, 0.022, 0.036, 0.052):
        size = max(7, round(minimum * fraction))
        if size % 2 == 0:
            size += 1
        responses.append(
            cv2.morphologyEx(
                processed,
                cv2.MORPH_BLACKHAT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            )
        )
    response = np.maximum.reduce(responses)
    positive = response[response > 0]
    if positive.size:
        threshold = max(10.0, float(np.percentile(positive, 90.0)))
        candidates = (response >= threshold).astype(np.uint8)
    else:
        candidates = np.zeros_like(gray, dtype=np.uint8)

    candidates = cv2.morphologyEx(
        candidates,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates,
        connectivity=8,
    )
    filtered = np.zeros_like(candidates)
    minimum_area = max(11, round(np.hypot(*gray.shape) * 0.010))
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        span = max(int(width), int(height))
        if int(area) >= minimum_area and span >= max(14, minimum_area):
            filtered[labels == label] = 255

    # OpenCV-only morphological skeletonization keeps this path usable even
    # when the model or scikit-image runtime is what failed on the host.
    skeleton = np.zeros_like(filtered)
    work = filtered.copy()
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max(gray.shape)):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, cross)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, cross)
        if cv2.countNonZero(work) == 0:
            break

    skeleton_bool = skeleton > 0
    neighbors = cv2.filter2D(
        skeleton_bool.astype(np.uint8),
        cv2.CV_16S,
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    neighbors -= skeleton_bool.astype(np.int16)
    endpoints = skeleton_bool & (neighbors == 1)
    junctions = skeleton_bool & (neighbors >= 3)
    overlay = make_overlay(img, skeleton_bool, junctions)
    graph = make_graph_mask(skeleton_bool, endpoints, junctions)
    connected = max(
        0,
        cv2.connectedComponents(skeleton_bool.astype(np.uint8), connectivity=8)[0]
        - 1,
    )
    return processed, overlay, graph, skeleton_bool, endpoints, junctions, connected


def _fallback_result(img, error):
    """Return a valid result for every decodable capture, even after pipeline errors."""
    logger.error(
        "Advanced image processing failed; using full-frame fallback",
        exc_info=(type(error), error, error.__traceback__),
    )
    (
        processed,
        overlay,
        graph,
        skeleton,
        endpoints,
        junctions,
        connected,
    ) = _opencv_fallback_visualization(img)
    return {
        "original": _to_data_url(img),
        "processed": _to_data_url(processed),
        "overlay": _to_data_url(overlay),
        "graph": _to_data_url(graph),
        "analysis": {
            "signalQuality": 0.0,
            "pathConfidence": 0.0,
            "vesselCoverage": round(float(skeleton.mean()), 5),
            "connectedVessels": connected,
            "segments": connected,
            "endpoints": int(endpoints.sum()),
            "junctions": int(junctions.sum()),
            "warnings": [
                "Advanced processing unavailable; OpenCV fallback overlay used"
            ],
            "armSegmentation": {
                "method": "full-frame-fallback",
                "confidence": 0.0,
            },
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode uploaded image",
        )

    try:
        prepared = prepare_image(img)
        vessels = detect_vessel_graph(
            prepared.enhanced,
            prepared.corrected,
            prepared.vessel_enhanced,
            prepared.analysis_mask,
        )
        overlay = make_overlay(
            prepared.original,
            vessels.skeleton,
            vessels.junctions,
        )
        graph = make_graph_mask(
            vessels.skeleton,
            vessels.endpoints,
            vessels.junctions,
        )
    except Exception as error:
        return _fallback_result(img, error)

    warnings = list(prepared.warnings)
    if vessels.connected_vessels == 0:
        warnings.append("No reliable vessel paths detected")
    elif vessels.confidence < 0.30:
        warnings.append("Detected paths have weak visual support")

    return {
        "original": _to_data_url(prepared.original),
        "processed": _to_data_url(prepared.enhanced),
        "overlay": _to_data_url(overlay),
        "graph": _to_data_url(graph),
        "analysis": {
            "signalQuality": round(prepared.signal_quality, 3),
            "pathConfidence": round(vessels.confidence, 3),
            "vesselCoverage": round(vessels.coverage, 5),
            "connectedVessels": vessels.connected_vessels,
            "segments": vessels.segment_count,
            "endpoints": int(vessels.endpoints.sum()),
            "junctions": int(vessels.junctions.sum()),
            "warnings": warnings,
            "armSegmentation": {
                "method": prepared.segmentation_method,
                "confidence": round(prepared.segmentation_confidence, 3),
            },
        },
    }
