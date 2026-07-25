# FastAPI server for training-free superficial-vessel visualization.
import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from preprocessing import prepare_image
from vessel_detection import detect_vessel_graph
from overlay import make_graph_mask, make_overlay

app = FastAPI(title="VEINZ API")

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

    prepared = prepare_image(img)
    vessels = detect_vessel_graph(
        prepared.enhanced,
        prepared.corrected,
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
        },
    }
