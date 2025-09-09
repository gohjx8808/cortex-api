from fastapi import FastAPI
from cortex_api.routers import objectDetection


app = FastAPI()


app.include_router(
    objectDetection.router, prefix="/object-detection", tags=["Object Detection"]
)


@app.get("/")
def read_root():
    return {"Hello": "World"}
