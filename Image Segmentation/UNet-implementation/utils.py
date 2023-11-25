import torch
import torchvision
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Save model checkpoint.

    Args:
        state (dict): Model state to be saved.
        filename (str): Name of the file to save the checkpoint.
    """
    print("[INFO] Saving chekpoint...")
    torch.save(state, filename)
     
def load_checkpoint(checkpoint, model):
    """
    Load model checkpoint.

    Args:
        checkpoint (dict): Loaded checkpoint containing model state.
        model (torch.nn.Module): Model to load the state into.
    """
    print("[INFO] Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
     

def display_batch(loader, num_batch=2):
    # Get a batch from the loader
    for batch_idx, (images, masks) in enumerate(loader):
        if batch_idx >= num_batch:
            break

        # Convert tensors to numpy arrays
        images = [TF.to_pil_image(image) for image in images]
        masks = [TF.to_pil_image(mask) for mask in masks]

        # Display the images and masks side by side
        fig, axes = plt.subplots(2, len(images), figsize=(10, 5))

        for i, (image, mask) in enumerate(zip(images, masks)):
            axes[0, i].imshow(image)
            axes[0, i].axis('off')
            axes[0, i].set_title(f"{i + 1}")

            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f"{i + 1}")

        plt.show()

def save_predictions_as_images(
    loader, model, folder="saved_images/", device="cuda"
):
    """
    Save model predictions and ground truth images.

    Args:
        loader (torch.utils.data.DataLoader): Data loader for validation data.
        model (torch.nn.Module): Model for predictions.
        folder (str): Folder to save the images (default: "saved_images/").
        device (str): Device to use for inference (default: "cuda").
    """
    model.eval()
    for idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device=device)
        with torch.no_grad():
            predictions = torch.sigmoid(model(inputs))
            predictions = (predictions > 0.5).float()
        torchvision.utils.save_image(
            predictions, f"{folder}/prediction_{idx}.png"
        )
        torchvision.utils.save_image(targets.unsqueeze(1), f"{folder}/target_{idx}.png")

    model.train()
            
