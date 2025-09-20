import json

import cv2
import numpy as np
from deep_translator import GoogleTranslator
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

router = APIRouter()


class DetectionResponse(BaseModel):
    objects: list[dict]  # [{ "class": "dog", "confidence": 0.9, "box": [x,y,w,h] }]


model = YOLO("yolo11n.pt")


@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    translator = GoogleTranslator(source="en", target="zh-TW")

    # Read file contents
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)

    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model(img)
    parsedResults = json.loads(results[0].to_json())

    # Translate object names
    for result in parsedResults:
        result["name"] = translator.translate(result["name"])
    return parsedResults
