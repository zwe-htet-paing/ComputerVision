import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initializes the CustomDataset.

        Args:
            image_dir (str): Directory containing input images.
            mask_dir (str): Directory containing corresponding mask images.
            transform (callable, optional): Optional transform to be applied to the images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
    
        
    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding mask based on the given index.

        Args:
            index (int): Index of the image in the dataset.

        Returns:
            tuple: A tuple containing the image and its mask.
        """
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, image_filename.replace(".jpg", "_mask.gif"))
        
        # Load image and mask
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # Convert mask values to binary (0 or 1)
        mask[mask == 255.0] = 1.0
        
        # Apply transformations if available
        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]
            
        return image, mask
    
def get_data_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    """
    Get train and validation data loaders.

    Args:
        train_dir (str): Directory for training images.
        train_mask_dir (str): Directory for training masks.
        val_dir (str): Directory for validation images.
        val_mask_dir (str): Directory for validation masks.
        batch_size (int): Batch size for data loaders.
        train_transform (albumentations): Transformations for training data.
        val_transform (albumentations): Transformations for validation data.
        num_workers (int): Number of workers for data loaders.
        pin_memory (bool): Whether to use pin memory for data loaders.

    Returns:
        tuple: Train and validation data loaders.
    """
    train_dataset = CustomDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_dataset = CustomDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader