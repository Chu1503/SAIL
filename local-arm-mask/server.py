"""Local FastAPI server: SAM2 arm isolation + CUBITAL vein extraction.

Speaks the exact same response contract as backend/main.py's /process so the
deployed frontend and APK need zero code changes to use it.

Run with: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import base64
import logging
import os
import sys

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from mask_arm import inference_autocast, load_predictor, mask_arm
from vein_extraction import configure_gpu_memory_growth, extract_veins, load_vein_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from overlay import make_graph_mask, make_overlay  # noqa: E402

SAM_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT", "/home/chu/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
)
SAM_MODEL_CFG = os.environ.get("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_b+.yaml")
CUBITAL_MODEL_PATH = os.environ.get(
    "CUBITAL_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "unet.keras"),
)

app = FastAPI(title="VEINZ local server")
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
_vein_model = None


@app.on_event("startup")
def _load_models():
    global _predictor, _device, _vein_model
    configure_gpu_memory_growth()
    _predictor, _device = load_predictor(SAM_CHECKPOINT, SAM_MODEL_CFG)
    logger.info("SAM2 predictor loaded on %s", _device)
    _vein_model = load_vein_model(CUBITAL_MODEL_PATH)
    logger.info("CUBITAL vein model loaded from %s", CUBITAL_MODEL_PATH)


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

    warnings = []
    if mask is None:
        warnings.append("SAM2 could not find a confident arm mask; showing full frame")
        masked = img
        mask = np.ones(img.shape[:2], dtype=bool)

    vein = extract_veins(_vein_model, img, mask)
    if vein.connected_vessels == 0:
        warnings.append("No reliable vein paths detected")

    overlay = make_overlay(masked, vein.skeleton, vein.junctions)
    graph = make_graph_mask(vein.skeleton, vein.endpoints, vein.junctions)

    return {
        "original": _to_data_url(img),
        "processed": _to_data_url(masked),
        "overlay": _to_data_url(overlay),
        "graph": _to_data_url(graph),
        "analysis": {
            "signalQuality": round(float(vein.response[mask.astype(bool)].mean()), 3)
            if np.any(mask)
            else 0.0,
            "pathConfidence": round(vein.confidence, 3),
            "vesselCoverage": round(vein.coverage, 5),
            "connectedVessels": vein.connected_vessels,
            "segments": vein.segment_count,
            "endpoints": int(vein.endpoints.sum()),
            "junctions": int(vein.junctions.sum()),
            "warnings": warnings,
            "pipeline": {
                "armIsolation": {
                    "name": "SAM2 (hiera base+)",
                    "tier": "Local GPU",
                    "runtime": f"PyTorch {_device.upper()}",
                    "status": "primary" if np.any(mask) else "fallback",
                },
                "veinExtraction": {
                    "name": "CUBITAL U-Net (940nm NIR, forearm)",
                    "tier": "Learned segmentation",
                    "runtime": "TensorFlow (local GPU)",
                    "status": "primary" if vein.connected_vessels > 0 else "fallback",
                },
            },
        },
    }
