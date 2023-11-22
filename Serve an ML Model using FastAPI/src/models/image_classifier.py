from models.pred import *
from utils.utilities import *
from typing import Any

def run_classifier(image: str) -> Any:
    img = load_image(image)
    if img is None:
        return None
    input_batch = image_preprocess(img)
    top_labels = predict(input_batch, model=None)
    return top_labels[0]