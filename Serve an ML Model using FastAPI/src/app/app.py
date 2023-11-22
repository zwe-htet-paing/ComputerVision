from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas.schemas import ImageRequest, PredictionResponse
from models.image_classifier import run_classifier

app = FastAPI(title="Image Classifier API")

origins = [
    "http://localhost:8080",
    "localhost:8080",
    "*",
    "http://127.0.0.1:8088/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def hello():
    return {"msg": "Hello World!!!"}

@app.post("/predict", response_model=PredictionResponse, status_code=200)
async def predict(request: ImageRequest):
    prediction = run_classifier(request.img_url)
    
    if not prediction:
        raise HTTPException(
            status_code=404, detail="Image could not be downloaded"
        )
    
    return PredictionResponse(
        status_code=200,
        prediction_label=prediction[0],
        probability=prediction[1]
    )
    