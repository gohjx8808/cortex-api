import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

router = APIRouter()


class DetectionResponse(BaseModel):
    objects: list[dict]  # [{ "class": "dog", "confidence": 0.9, "box": [x,y,w,h] }]


model = YOLO("yolo11n.pt")


@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read file contents
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)

    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    return results[0].to_json()
