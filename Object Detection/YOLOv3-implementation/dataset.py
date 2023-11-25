import config
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
from utils import (
    convert_cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        grid_sizes=[13, 26, 52],
        num_classes=20,
        transform=None,
    ):
        # Read the csv file with image names and labels 
        self.annotations = pd.read_csv(csv_file)
        # Image and label directories
        self.img_dir = img_dir
        self.label_dir = label_dir
        # Image size
        self.image_size = image_size
        # Transformations
        self.transform = transform
        # Grid sizes for each scale
        self.grid_sizes = grid_sizes
        # Anchor boxes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # Combine anchors for all scales
        # Number of anchor boxes
        self.num_anchors = self.anchors.shape[0]
        # Number of anchor boxes per scale 
        self.num_anchors_per_scale = self.num_anchors // 3
        # Number of classes 
        self.num_classes = num_classes
        # Ignore IoU threshold 
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.annotations)

    def __getitem__(self, index):
        # Getting the label path
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        
        # We are applying roll to move class label to the last column 
        # 5 columns: x, y, width, height, class_label 
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        
        # Getting the image path
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Albumentations augmentations 
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale 
        # target : [probabilities, x, y, width, height, class_label] 
        targets = [torch.zeros((self.num_anchors // 3, grid_sizes, grid_sizes, 6)) for grid_sizes in self.grid_sizes]
        
        # Identify anchor box and cell for each bounding box
        for box in bboxes:
            # Calculate iou of bounding box with anchor boxes 
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            
            # Selecting the best anchor box 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            
            # At each scale, assigning the bounding box to the  
            # best matching anchor box 
            has_anchor = [False] * 3  # Each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                
                # Identifying the grid size for the scale
                S = self.grid_sizes[scale_idx]
                
                # Identifying the cell to which the bounding box belongs 
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                
                # Check if the anchor box is already assigned
                if not anchor_taken and not has_anchor[scale_idx]:
                    ## Set the probability to 1 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    
                    # Calculating the center of the bounding box relative to 
                    # the cell
                    x_cell, y_cell = S * x - j, S * y - i  # Both values are between [0,1]
                    
                    # Calculating the width and height of the bounding box  
                    # relative to the cell 
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )
                    
                    # Idnetify the box coordinates 
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    
                    # Assigning the box coordinates to the target
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    
                    # Assigning the class label to the target
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    
                    # Set the anchor box as assigned for the scale
                    has_anchor[scale_idx] = True

                # If the anchor box is already assigned, check if the  
                # IoU is greater than the threshold 
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # If anchor is not taken and has high IoU, 
                    # Set the probability to -1 to ignore the prediction
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        # Return the image and the target 
        return image, tuple(targets)

def get_loaders(train_csv_path, test_csv_path):
    # Initialize YOLODataset instances for training, testing, and training evaluation
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transforms,
        grid_sizes=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        grid_sizes=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=config.test_transforms,
        grid_sizes=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )

    # Create DataLoader instances for training, testing, and training evaluation
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader

def test_dataset():
    # Test the YOLODataset with configured anchors and transformations
    anchors = config.ANCHORS
    transform = config.test_transforms

    dataset = YOLODataset(
        "dataset/PASCAL_VOC/train.csv",
        "dataset/PASCAL_VOC/images/",
        "dataset/PASCAL_VOC/labels/",
        grid_sizes=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    grid_sizes = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(grid_sizes).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    iterloader = iter(loader)
    
    for x, y in iterloader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += convert_cells_to_bboxes(
                y[i], is_preds=False, grid_sizes=y[i].shape[2], anchors=anchor
            )[0]
        # Apply non-maximum suppression and plot the image with bounding boxes
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        # print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test_dataset()
