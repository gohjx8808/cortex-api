import json
from enum import Enum

import cv2
import numpy as np
from deep_translator import GoogleTranslator
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO


class Language(str, Enum):
    en = "en"
    zh = "zh-CN"


router = APIRouter()


class DetectionResponse(BaseModel):
    objects: list[dict]  # [{ "class": "dog", "confidence": 0.9, "box": [x,y,w,h] }]


model = YOLO("yolo11n.pt")


ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@router.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    lang: Language = Form(..., description="Language code (e.g. en, zh-CN)"),
) -> list[dict]:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400, detail="Only JPG and PNG images are allowed"
        )

    translator = GoogleTranslator(source="en", target=lang)

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
