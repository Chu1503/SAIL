"""Local FastAPI server: SAM2 arm isolation, orchestrating CUBITAL vein
extraction over a local HTTP call to vein_server.py (a separate process in
a separate venv -- PyTorch and TensorFlow's dependencies conflict if
installed together, see requirements-sam2.txt vs requirements-cubital.txt).

Speaks the exact same response contract as backend/main.py's /process so the
deployed frontend and APK need zero code changes to use it.

Run with: uvicorn server:app --host 0.0.0.0 --port 8000
Requires vein_server.py already running on port 8001.
"""

import base64
import logging
import os
import sys

import cv2
import numpy as np
import requests
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from mask_arm import inference_autocast, load_predictor, mask_arm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from overlay import (  # noqa: E402
    draw_injection_marker,
    enhance_for_display,
    make_graph_mask,
    make_overlay,
)

SAM_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT", "/home/chu/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
)
SAM_MODEL_CFG = os.environ.get("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_b+.yaml")
VEIN_SERVICE_URL = os.environ.get("VEIN_SERVICE_URL", "http://127.0.0.1:8001")

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


@app.on_event("startup")
def _load_models():
    global _predictor, _device
    _predictor, _device = load_predictor(SAM_CHECKPOINT, SAM_MODEL_CFG)
    logger.info("SAM2 predictor loaded on %s", _device)
    try:
        resp = requests.get(f"{VEIN_SERVICE_URL}/health", timeout=5)
        resp.raise_for_status()
        logger.info("Vein extraction service reachable at %s", VEIN_SERVICE_URL)
    except Exception:
        logger.warning(
            "Vein extraction service not reachable yet at %s -- "
            "make sure vein_server.py is running (cubital-env, port 8001)",
            VEIN_SERVICE_URL,
        )


def _to_data_url(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result image")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _decode_mask(b64_png):
    raw = base64.b64decode(b64_png)
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    return arr > 0


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

    ok_mask, mask_buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    ok_img, image_buf = cv2.imencode(".png", img)
    if not ok_mask or not ok_img:
        raise HTTPException(status_code=500, detail="Could not encode image for vein service")

    try:
        response = requests.post(
            f"{VEIN_SERVICE_URL}/extract-veins",
            files={
                "file": ("image.png", image_buf.tobytes(), "image/png"),
                "mask": ("mask.png", mask_buf.tobytes(), "image/png"),
            },
            timeout=60,
        )
        response.raise_for_status()
        vein = response.json()
    except Exception as error:
        logger.exception("Vein extraction service call failed")
        raise HTTPException(
            status_code=502,
            detail=f"Vein extraction service unavailable: {error}",
        )

    skeleton = _decode_mask(vein["skeleton"])
    endpoints = _decode_mask(vein["endpoints"])
    junctions = _decode_mask(vein["junctions"])

    if vein["connectedVessels"] == 0:
        warnings.append("No reliable vein paths detected")

    overlay_base = enhance_for_display(img, mask)
    overlay = make_overlay(overlay_base, skeleton, junctions)
    graph = make_graph_mask(skeleton, endpoints, junctions)

    injection_point = vein.get("injectionPoint")
    if injection_point is not None:
        overlay = draw_injection_marker(overlay, (injection_point["x"], injection_point["y"]))
    else:
        warnings.append("No confident injection point found")

    return {
        "original": _to_data_url(img),
        "processed": _to_data_url(masked),
        "overlay": _to_data_url(overlay),
        "graph": _to_data_url(graph),
        "analysis": {
            "signalQuality": round(float(vein["signalQuality"]), 3),
            "pathConfidence": round(float(vein["confidence"]), 3),
            "vesselCoverage": round(float(vein["coverage"]), 5),
            "connectedVessels": vein["connectedVessels"],
            "segments": vein["segments"],
            "endpoints": int(endpoints.sum()),
            "junctions": int(junctions.sum()),
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
                    "runtime": "TensorFlow (local)",
                    "status": "primary" if vein["connectedVessels"] > 0 else "fallback",
                },
            },
        },
    }
