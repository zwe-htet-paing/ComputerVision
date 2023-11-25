import torch

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from utils import seed_everything
seed_everything()  # If you want deterministic behavior

# Dataset and device configuration
DATASET = 'dataset/PASCAL_VOC'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of worker
NUM_WORKERS = 4

# Training hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100

# Image and grid cell sizes
IMAGE_SIZE = 416
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

# Number of classes
NUM_CLASSES = 20

# Optimization and evaluation parameters
WEIGHT_DECAY = 1e-4
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45

# Data loading configuration
PIN_MEMORY = True

# Model loading and saving
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"

# Directories for images and labels
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

# Anchor boxes for each feature map scaled between 0 and 1 
# 3 feature maps at 3 different scales based on YOLOv3 paper 
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

# Class labels for PASCAL dataset
PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", 
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", 
    "sheep", "sofa", "train", "tvmonitor"
]

# Class labels for COCO dataset
COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# Scale factor for image augmentation
scale = 1.1

# Training data transformations
train_transforms = A.Compose(
    [
        # Resize to the longest side while maintaining the aspect ratio
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        # Pad if needed to achieve the target height and width
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        # Random crop to the target size
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        # Adjust color brightness, contrast, saturation, and hue
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        # Randomly apply rotation or affine transformations
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        # Horizontal flip with a probability of 0.5
        A.HorizontalFlip(p=0.5),
        # Apply blur with a probability of 0.1
        A.Blur(p=0.1),
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) with a probability of 0.1
        A.CLAHE(p=0.1),
        # Apply posterization with a probability of 0.1
        A.Posterize(p=0.1),
        # Convert the image to grayscale with a probability of 0.1
        A.ToGray(p=0.1),
        # Randomly shuffle color channels with a low probability
        A.ChannelShuffle(p=0.05),
        # Normalize pixel values and convert to PyTorch tensor
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

# Test data transformations
test_transforms = A.Compose(
    [
        # Resize to the specified size while maintaining the aspect ratio
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        # Pad if needed to achieve the target height and width
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        # Normalize pixel values and convert to PyTorch tensor
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
