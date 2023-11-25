import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import config
from model import UNet
from dataset import get_data_loaders
from utils import load_checkpoint, save_checkpoint, save_predictions_as_images, display_batch


def train_one_epoch(epoch, loader, model, optimizer, loss_fn, scaler):
    """Training function for one epoch."""
    loop = tqdm(loader, desc=f"[INFO] Epoch {epoch+1}")

    for batch_idx, (data, target) in enumerate(loop):
        data, targets = data.to(device=config.DEVICE), target.float().unsqueeze(1).to(device=config.DEVICE)

        # Forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
        #     print(data.shape, predictions.shape, targets.shape)
        #     print("Input data:", data)
        #     print("Predictions:", predictions)
        #     print("Targets:", targets)
        #     print("Loss:", loss.item())
        #     break
        # break
            
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())
        
def evaluate_model(loader, model):
    """Evaluate model accuracy and dice score."""
    num_correct = 0
    num_pixels = 0
    dice_score = 0
        
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(config.DEVICE)
            target = target.to(config.DEVICE).unsqueeze(1)
            
            # Make predictions using the model and apply threshold
            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()
            
            # Update evaluation metrics
            num_correct += (predictions == target).sum()
            num_pixels += torch.numel(predictions)
            
            # Compute dice score
            dice_score += (2 * (predictions * target).sum()) / (
                (predictions + target).sum() + 1e-8
            )

    print(f"[INFO] Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}%")
    print(f"[INFO] Dice score: {dice_score/len(loader)}")
    model.train()

def main():
    # Model, loss function, optimizer setup
    model = UNet(in_channels=3, out_channels=1).to(config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Train and validation data loaders setup
    train_loader, val_loader = get_data_loaders(
        config.DATA_DIRS["train"]["image"], config.DATA_DIRS["train"]["mask"],
        config.DATA_DIRS["val"]["image"], config.DATA_DIRS["val"]["mask"],
        config.BATCH_SIZE, config.train_transform, config.val_transform,
        config.NUM_WORKERS, config.PIN_MEMORY
    )
    
    display_batch(train_loader, num_batch=2)

    # Load pre-trained model if specified
    if config.LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

        # Evaluate initial model accuracy
        evaluate_model(val_loader, model)

    # Gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        # Train model for one epoch
        train_one_epoch(epoch, train_loader, model, optimizer, loss_fn, scaler)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(), 
            "optimizer": optimizer.state_dict()
            }
        save_checkpoint(checkpoint)

        # Check accuracy
        evaluate_model(val_loader, model)

        # Save predictions as images
        save_predictions_as_images(val_loader, model, folder="saved_images/", device=config.DEVICE)

if __name__ == "__main__":
    main()
