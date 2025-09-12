from cortex_api.routers import objectDetection
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:19006"] for Expo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    objectDetection.router, prefix="/object-detection", tags=["Object Detection"]
)


@app.get("/")
def read_root():
    return {"Hello": "World"}
