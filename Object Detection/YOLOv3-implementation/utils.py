import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch

from collections import Counter
from tqdm import tqdm

def iou_width_height(boxes1, boxes2):
    """
    Calculate the Intersection over Union (IoU) of bounding boxes using their width and height.

    Parameters:
        boxes1 (tensor): Tensor containing width and height of the first bounding boxes.
        boxes2 (tensor): Tensor containing width and height of the second bounding boxes.

    Returns:
        tensor: Intersection over Union of the corresponding boxes.
    """
    # Calculate the intersection area
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )

    # Calculate the union area
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )

    # Calculate IoU by dividing intersection by union (with handling for potential division by zero)
    iou = intersection / union.where(union > 0, torch.tensor(1.0))

    return iou


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculate Intersection over Union (IoU) given predicted and target bounding boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): Format of the boxes, either "midpoint" or "corners".

    Returns:
        tensor: Intersection over union for all examples.
    """

    # Extract coordinates based on the box format
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # Calculate the intersection coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Calculate the intersection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculate the areas of the individual boxes
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # Calculate IoU by dividing intersection by union (with handling for potential division by zero)
    iou = intersection / (box1_area + box2_area - intersection + 1e-6)

    return iou


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Perform Non-Maximum Suppression (NMS) on a list of bounding boxes.

    Parameters:
        bboxes (list): List of bounding boxes, each specified as [class_prediction, prob_score, x1, y1, x2, y2].
        iou_threshold (float): IoU threshold where predicted bounding boxes are considered correct.
        threshold (float): Threshold to remove predicted bounding boxes (independent of IoU).
        box_format (str): "midpoint" (x,y,w,h) or "corners" (x1,y1,x2,y2) used to specify the format of bounding boxes.

    Returns:
        list: Bounding boxes after performing NMS given a specific IoU threshold.
    """

    # Ensure the input is a list
    assert type(bboxes) == list

    # Filter out bounding boxes with confidence score below the specified threshold
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort bounding boxes by confidence score in descending order
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # List to store bounding boxes after NMS
    bboxes_after_nms = []

    while bboxes:
        # Choose the box with the highest confidence score
        chosen_box = bboxes.pop(0)

        # Filter out boxes with the same class or high IoU with the chosen box
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        # Add the chosen box to the final list
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    Calculate mean average precision (mAP) across all classes.

    Parameters:
        pred_boxes (list): List of predicted bounding boxes, each specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2].
        true_boxes (list): List of true bounding boxes, similar to pred_boxes.
        iou_threshold (float): IoU threshold where predicted bounding boxes are considered correct.
        box_format (str): "midpoint" or "corners" used to specify the format of bounding boxes.
        num_classes (int): Number of classes.

    Returns:
        float: mAP value across all classes given a specific IoU threshold.
    """

    # List storing all average precisions for respective classes
    average_precisions = []

    # Used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Separate predictions and ground truths for the current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # Find the amount of bounding boxes for each training example
        # if img 0 has 3 boxes, img 1 has 5 boxes then amount_bboxes={0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # Convert the amount_bboxes dictionary to torch tensors
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sort predictions by box probabilities (index 2)
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If no ground truths exist for this class, skip to the next class
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Extract ground truths that have the same training index as the detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]), # only [x1, y1, x2, y2]
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # Only consider a ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # True positive, and mark this bounding box as seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # If IoU is lower, consider the detection as a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # Use torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def convert_cells_to_bboxes(predictions, anchors, grid_sizes, is_preds=True):
    """
    Scales the predictions coming from the model to be relative to the entire image.
    
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    grid_sizes: the number of cells the image is divided into on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors * S * S, 1+5)
                      with class index, object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    
    # Extract bounding box predictions
    box_predictions = predictions[..., 1:5]
    
    if is_preds:
        # Reshape anchors for compatibility
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        
        # Apply sigmoid to bounding box coordinates
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        
        # Transform bounding box dimensions using exponential function
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        
        # Apply sigmoid to objectness score
        scores = torch.sigmoid(predictions[..., 0:1])
        
        # Find the index of the class with the maximum probability
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        # If ground truth, use scores and class directly
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]
        
    S = grid_sizes

    # Generate indices for each cell in the grid
    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    
    # Calculate bounding box coordinates relative to the entire image
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    
    # Concatenate class index, object score, and bounding box coordinates
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    
    # Convert to list for further use
    return converted_bboxes.tolist()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Save model and optimizer state to a checkpoint file.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be saved.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to be saved.
        filename (str): The filename for the checkpoint file.
    """
    print("[INFO] Saving checkpoint")
    
    # Create a dictionary to store the model and optimizer state
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    
    # Save the checkpoint to the specified file
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Load model and optimizer state from a checkpoint file.

    Parameters:
        checkpoint_file (str): The filename of the checkpoint file to be loaded.
        model (torch.nn.Module): The PyTorch model to load the state into.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to load the state into.
        lr (float): The learning rate to set for the optimizer.
    """
    print("[INFO] Loading checkpoint")
    
    # Load the checkpoint from the specified file
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    
    # Load the model and optimizer state
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update the learning rate in the optimizer
    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
        
def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.

    Parameters:
        seed (int): The seed value for random number generators.
    """
    # Set PYTHONHASHSEED to control hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set seed for Python built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    
    # Set seed for PyTorch on GPU (if available)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior of CuDNN
    torch.backends.cudnn.deterministic = True
    
    # Disable CuDNN benchmark mode for reproducibility
    torch.backends.cudnn.benchmark = False


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image."""
    # Choose colormap based on the number of classes
    cmap = plt.get_cmap("tab20b")
    
    # Get class labels based on the dataset
    class_labels = config.COCO_LABELS if config.DATASET == 'COCO' else config.PASCAL_CLASSES
    
    # Generate colors for each class
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    
    # Convert image to NumPy array
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(im)

    # Box[0] is x midpoint, box[2] is width
    # Box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch for each box
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        
        # Extract class prediction and bounding box coordinates
        class_pred = box[0]
        box = box[2:]
        
        # Calculate upper left coordinates of the bounding box
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        
        # Add the patch to the Axes
        ax.add_patch(rect)
        
        # Add class label as text
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # Show the plot
    plt.show()


def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    """
    Plot examples with bounding boxes predicted by the model.

    Parameters:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for loading images.
        thresh (float): Confidence threshold for predictions.
        iou_thresh (float): IoU threshold for non-maximum suppression.
        anchors (list): List of anchor boxes for each scale.

    Note:
        This function assumes that the model is in evaluation mode.
    """
    model.eval()

    # Get a batch of images and move to GPU
    x, y = next(iter(loader))
    x = x.to("cuda")

    with torch.no_grad():
        # Forward pass through the model
        out = model(x)

        # Initialize a list to store bounding boxes for each image
        bboxes = [[] for _ in range(x.shape[0])]

        # Process predictions for each scale
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]

            # Convert model predictions to bounding boxes
            boxes_scale_i = convert_cells_to_bboxes(out[i], anchor, grid_sizes=S, is_preds=True)

            # Aggregate bounding boxes for each image
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # Switch back to training mode
        model.train()

    # Process each image in the batch
    for i in range(batch_size):
        # Apply non-maximum suppression to get filtered bounding boxes
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )

        # Plot the image with predicted bounding boxes
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)


def get_evaluation_bboxes(loader, model, iou_threshold, anchors, threshold, box_format="midpoint", device="cuda"):
    """
    Get predicted and true bounding boxes for evaluation.

    Args:
        loader: DataLoader for the evaluation dataset.
        model: YOLO model for making predictions.
        iou_threshold: Intersection over Union (IoU) threshold for non-maximum suppression.
        anchors: List of anchor boxes for each scale.
        threshold: Confidence threshold for considering predicted bounding boxes.
        box_format: Format for representing bounding boxes ("midpoint" or "corners").
        device: Device for running the model.

    Returns:
        Tuple of lists containing all predicted and true bounding boxes.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []

    # Iterate through the evaluation dataset
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]

        # Iterate through each scale's predictions
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = convert_cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # Get true bounding boxes
        true_bboxes = convert_cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        # Process each instance in the batch
        for idx in range(batch_size):
            # Apply non-maximum suppression to predicted boxes
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            # Append predicted and true boxes to the respective lists
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    # Restore the model to training mode
    model.train()
    return all_pred_boxes, all_true_boxes


def check_class_accuracy(model, loader, threshold):
    """
    Check classification accuracy for object detection.

    Args:
        model: YOLO model for making predictions.
        loader: DataLoader for the dataset.
        threshold: Confidence threshold for considering predicted bounding boxes.

    Prints:
        Class accuracy, No object accuracy, and Object accuracy.
    """
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1  # Indicator for object presence
            noobj = y[i][..., 0] == 0  # Indicator for no object

            # Calculate classification accuracy for objects
            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            # Calculate object presence accuracy
            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)

            # Calculate no object presence accuracy
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    # Print and display the results
    print(f"Class accuracy is: {(correct_class / (tot_class_preds + 1e-16)) * 100:.2f}%")
    print(f"No object accuracy is: {(correct_noobj / (tot_noobj + 1e-16)) * 100:.2f}%")
    print(f"Object accuracy is: {(correct_obj / (tot_obj + 1e-16)) * 100:.2f}%")
    model.train()


def get_mean_std(loader):
    """
    Calculate the mean and standard deviation of the dataset.

    Args:
        loader: DataLoader for the dataset.

    Returns:
        Tuple containing the mean and standard deviation of the dataset.
    """
    # Initialize variables
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    # Iterate through the dataset
    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    # Calculate mean and standard deviation
    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
