import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT, IMAGE_WIDTH = 160, 160
PIN_MEMORY = True
LOAD_MODEL = False
DATA_DIRS = {
    "train": {"image": "dataset/train_images/", "mask": "dataset/train_masks/"},
    "val": {"image": "dataset/val_images/", "mask": "dataset/val_masks/"},
}


# Transformation setup
train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(
        mean=[0.0, 0.0, 0.0], 
        std=[1.0, 1.0, 1.0], 
        max_pixel_value=255.0
        ),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(
        mean=[0.0, 0.0, 0.0], 
        std=[1.0, 1.0, 1.0], 
        max_pixel_value=255.0
        ),
    ToTensorV2(),
])