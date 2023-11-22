from pydantic import BaseModel


class PredictionResponse(BaseModel):
    status_code: int
    prediction_label: str
    probability: float

class ImageRequest(BaseModel):
    img_url: str
    