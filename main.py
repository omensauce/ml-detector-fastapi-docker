from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import base64
from io import BytesIO

from model import build_model

app = FastAPI()

model = build_model()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

CLASS_NAMES = ["background", "apple", "banana", "orange"]

transform = transforms.ToTensor()

@app.get("/")
def root():
    return {"message": "ML API is running"}


def draw_boxes(image_np, boxes, labels, scores, threshold=0.5):
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        label_name = CLASS_NAMES[label]

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{label_name}: {score:.2f}"
        cv2.putText(
            image_np,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return image_np


def encode_image(image_np):
    _, buffer = cv2.imencode(".jpg", image_np)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    
    image_np = np.array(image).copy()
    #OpenCV uses BGR, everything else (inc. Pillow and browsers) uses RGB
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    x = transform(image)

    with torch.no_grad():
        outputs = model([x])[0]

    boxes = outputs["boxes"].tolist()
    labels = outputs["labels"].tolist()
    scores = outputs["scores"].tolist()

    threshold = 0.6

    results = []
    has_detections = False

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            has_detections = True
            results.append({
                "box": box,
                "label": CLASS_NAMES[label],
                "score": float(score)
            })

    response = {"detections": results}

    if has_detections:
        drawn = draw_boxes(image_np, boxes, labels, scores, threshold)
        encoded_image = encode_image(drawn)
        response["image"] = encoded_image

    return response