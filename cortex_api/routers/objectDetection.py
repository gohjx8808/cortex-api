from fastapi import APIRouter
from pydantic import BaseModel
from ultralytics import YOLO

router = APIRouter()


class DetectionResponse(BaseModel):
    objects: list[dict]  # [{ "class": "dog", "confidence": 0.9, "box": [x,y,w,h] }]


model = YOLO("yolo11n.pt")


@router.get("/detect")
def detect_objects():
    results = model.train(data="coco8.yaml", epochs=3)
    print(results)
    return {"message": "Object detection endpoint"}
