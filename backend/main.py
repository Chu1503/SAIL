# FastAPI server: receives a captured image, runs preprocessing + model + overlay, returns result images as data URLs.
import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from preprocessing import enhance
from model import predict
from overlay import make_overlay

app = FastAPI(title="VEINZ API")

allowed_origins = [
    "http://localhost:3000",
    "https://sail-kohl.vercel.app/",
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
        return {"error": "Could not decode image"}

    original, detection_base, display_base = enhance(img)
    mask = predict(detection_base)
    result = make_overlay(display_base, mask)

    return {
        "original": _to_data_url(original),
        "overlay": _to_data_url(result),
        "mask": _to_data_url(mask),
    }
