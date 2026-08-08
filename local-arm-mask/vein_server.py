"""Internal-only local service: CUBITAL vein extraction.

Runs in cubital-env (a separate venv/process from SAM2's PyTorch env -- the
two frameworks' dependencies conflict if installed together). Not exposed to
the internet; only server.py (the public-facing SAM2 orchestrator, on a
different port) calls this, over localhost.

Run with: uvicorn vein_server:app --host 127.0.0.1 --port 8001
"""

import base64
import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from injection_point import find_injection_point, load_fossa_model, predict_fossa
from vein_extraction import extract_veins, load_vein_model

MODEL_PATH = os.environ.get(
    "CUBITAL_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "unet.keras"),
)
FOSSA_MODEL_PATH = os.environ.get(
    "CUBITAL_FOSSA_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "unet_multi"),
)

app = FastAPI(title="VEINZ local vein-extraction service (internal)")
logger = logging.getLogger("veinz.vein-service")

_model = None
_fossa_infer = None


@app.on_event("startup")
def _load_model():
    global _model, _fossa_infer
    _model = load_vein_model(MODEL_PATH)
    logger.info("CUBITAL vein model loaded from %s", MODEL_PATH)
    _fossa_infer = load_fossa_model(FOSSA_MODEL_PATH)
    logger.info("CUBITAL fossa-localization model loaded from %s", FOSSA_MODEL_PATH)


def _encode_mask(mask_bool):
    ok, buf = cv2.imencode(".png", mask_bool.astype(np.uint8) * 255)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode mask")
    return base64.b64encode(buf).decode("ascii")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-veins")
async def extract_veins_endpoint(
    file: UploadFile = File(...), mask: UploadFile = File(...)
):
    image_bytes = await file.read()
    mask_bytes = await mask.read()

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    mask_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or mask_img is None:
        raise HTTPException(status_code=400, detail="Could not decode image or mask")

    arm_mask = mask_img > 0
    result = extract_veins(_model, image, arm_mask)
    signal_quality = float(result.response[arm_mask].mean()) if np.any(arm_mask) else 0.0

    fossa_x, fossa_y, fossa_angle = predict_fossa(_fossa_infer, image)
    injection_point = find_injection_point(
        result.skeleton,
        result.endpoints,
        result.junctions,
        result.response,
        fossa_point=(fossa_x, fossa_y),
    )

    return {
        "skeleton": _encode_mask(result.skeleton),
        "endpoints": _encode_mask(result.endpoints),
        "junctions": _encode_mask(result.junctions),
        "connectedVessels": result.connected_vessels,
        "segments": result.segment_count,
        "coverage": result.coverage,
        "confidence": result.confidence,
        "signalQuality": signal_quality,
        "injectionPoint": (
            {"x": injection_point[0], "y": injection_point[1]}
            if injection_point is not None
            else None
        ),
        "fossa": {"x": fossa_x, "y": fossa_y, "angle": fossa_angle},
    }
