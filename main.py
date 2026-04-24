from __future__ import annotations
from typing import Any
import os
import logging

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import torch
import torchvision  # registers torchvision::nms C++ op needed by TorchScript  # noqa: F401
from PIL import Image
import numpy as np
import cv2
import base64

app = FastAPI(title="Fruit Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

CLASS_NAMES = ["background", "apple", "banana", "orange"]
THRESHOLD = 0.6
MODEL_SOURCE = "local"
logger = logging.getLogger("fruit_detector")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def load_model() -> torch.jit.ScriptModule:
    global MODEL_SOURCE

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_uri = os.getenv("MLFLOW_MODEL_URI")
    run_id = os.getenv("MLFLOW_RUN_ID")
    artifact_path = os.getenv("MLFLOW_MODEL_ARTIFACT", "torchscript/fruits_model_light.pt")
    local_model_path = os.getenv("LOCAL_MODEL_PATH", "fruits_model.pt")

    if model_uri or run_id:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            resolved_uri = model_uri or f"runs:/{run_id}/{artifact_path}"
            downloaded_model = mlflow.artifacts.download_artifacts(artifact_uri=resolved_uri)
            MODEL_SOURCE = f"mlflow:{resolved_uri}"
            loaded = torch.jit.load(downloaded_model, map_location="cpu")
            loaded.eval()
            logger.info("Loaded model from MLflow: %s", resolved_uri)
            return loaded
        except Exception as exc:
            logger.exception("Failed loading model from MLflow; falling back to local model: %s", exc)

    MODEL_SOURCE = f"local:{local_model_path}"
    loaded = torch.jit.load(local_model_path, map_location="cpu")
    loaded.eval()
    logger.info("Loaded model from local path: %s", local_model_path)
    return loaded


model = load_model()


@app.get("/")
def root():
    return {
        "message": "Fruit Detector API is running",
        "classes": CLASS_NAMES[1:],
        "model_source": MODEL_SOURCE,
    }


def extract_detections(outputs: Any) -> dict[str, Any]:
    # TorchScript Faster R-CNN returns (losses, detections).
    if isinstance(outputs, tuple) and len(outputs) == 2:
        _, detections = outputs
        if isinstance(detections, list) and detections:
            return detections[0]
    if isinstance(outputs, list) and outputs:
        return outputs[0]
    if isinstance(outputs, dict):
        return outputs
    raise ValueError(f"Unexpected model output type: {type(outputs)}")


def draw_boxes(image_np, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        if score < THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_np,
            f"{CLASS_NAMES[label]}: {score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return image_np


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    x = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0

    with torch.no_grad():
        raw_outputs = model([x])
        outputs = extract_detections(raw_outputs)

    boxes = outputs.get("boxes", torch.empty((0, 4))).tolist()
    labels = outputs.get("labels", torch.empty((0,), dtype=torch.int64)).tolist()
    scores = outputs.get("scores", torch.empty((0,))).tolist()

    detections = [
        {
            "box": box,
            "label": CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label),
            "score": round(score, 4),
        }
        for box, label, score in zip(boxes, labels, scores)
        if score >= THRESHOLD
    ]

    response: dict[str, Any] = {"detections": detections}

    if detections:
        annotated = draw_boxes(image_np.copy(), boxes, labels, scores)
        _, buf = cv2.imencode(".jpg", annotated)
        response["image"] = base64.b64encode(buf.tobytes()).decode()

    return response
