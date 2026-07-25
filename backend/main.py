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


def _fallback_result(img, error):
    """Return a valid result for every decodable capture, even after pipeline errors."""
    logger.error(
        "Advanced image processing failed; using full-frame fallback",
        exc_info=(type(error), error, error.__traceback__),
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)
    graph = np.zeros_like(img)
    return {
        "original": _to_data_url(img),
        "processed": _to_data_url(processed),
        "overlay": _to_data_url(img),
        "graph": _to_data_url(graph),
        "analysis": {
            "signalQuality": 0.0,
            "pathConfidence": 0.0,
            "vesselCoverage": 0.0,
            "connectedVessels": 0,
            "segments": 0,
            "endpoints": 0,
            "junctions": 0,
            "warnings": ["Advanced processing unavailable; full-frame fallback used"],
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
