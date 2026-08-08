"""Local FastAPI server: arm-isolation-only stand-in for the hosted /process endpoint.

Speaks the exact same response contract as backend/main.py's /process so the
deployed frontend needs zero code changes to use it. Vein extraction is not
implemented here yet -- the arm-masked image is returned in every image slot,
and the pipeline.veinExtraction field is marked as not enabled.

Run with: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import base64
import logging
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from mask_arm import inference_autocast, load_predictor, mask_arm

CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT", "/home/chu/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
)
MODEL_CFG = os.environ.get("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_b+.yaml")

app = FastAPI(title="VEINZ local arm-isolation server")
logger = logging.getLogger("veinz.local")

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

_predictor = None
_device = None


@app.on_event("startup")
def _load_model():
    global _predictor, _device
    _predictor, _device = load_predictor(CHECKPOINT, MODEL_CFG)
    logger.info("SAM2 predictor loaded on %s", _device)


def _to_data_url(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result image")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


@app.get("/health")
def health():
    return {"status": "ok", "device": _device}


@app.post("/process")
async def process(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    with torch.inference_mode(), inference_autocast(_device):
        masked, mask = mask_arm(_predictor, img)

    if mask is None:
        warnings = ["SAM2 could not find a confident arm mask; showing full frame"]
        masked = img
        coverage = 1.0
    else:
        warnings = ["Vein detection not enabled yet on the local server (arm isolation only)"]
        coverage = float(mask.mean())

    masked_url = _to_data_url(masked)

    return {
        "original": _to_data_url(img),
        "processed": masked_url,
        "overlay": masked_url,
        "graph": masked_url,
        "analysis": {
            "signalQuality": 0.0,
            "pathConfidence": 0.0,
            "vesselCoverage": round(coverage, 5),
            "connectedVessels": 0,
            "segments": 0,
            "endpoints": 0,
            "junctions": 0,
            "warnings": warnings,
            "pipeline": {
                "armIsolation": {
                    "name": "SAM2 (hiera base+)",
                    "tier": "Local GPU",
                    "runtime": f"PyTorch {_device.upper()}",
                    "status": "primary" if mask is not None else "fallback",
                },
                "veinExtraction": {
                    "name": "Not enabled yet",
                    "tier": "N/A",
                    "runtime": "N/A",
                    "status": "fallback",
                },
            },
        },
    }
