from torch.utils.data import DataLoader
from CustomDataset import CVPDataset
from torchvision import transforms
import os
from pathlib import Path

# Paths to Tiny ImageNet-10 dataset directories
base_dir = r'..\tiny-imagenet-10'  # Base directory for Tiny ImageNet-10


# Define transforms
transforms_dict = {
    0: transforms.Compose([transforms.ToTensor()]),
    1: transforms.Compose([transforms.GaussianBlur(kernel_size=3, sigma=1), transforms.ToTensor()]),
    2: transforms.Compose([transforms.GaussianBlur(kernel_size=3, sigma=2), transforms.ToTensor()]),
    3: transforms.Compose([transforms.GaussianBlur(kernel_size=3, sigma=3), transforms.ToTensor()]),
    4: transforms.Compose([transforms.GaussianBlur(kernel_size=3, sigma=4), transforms.ToTensor()])
}

def get_dataloader(operation="train", tx="none", age=1.0, batch_size=64, shuffle=True):
    """
    Returns a DataLoader for the specified dataset, transformation, and age.

    Args:
        operation (str): "train", "test", or "validation" to select dataset.
        tx (str): "acuity" for visual acuity, "CS" for contrast sensitivity, or "none" for no transform.
        age (float): Age in months for the transform.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: PyTorch DataLoader with the specified configuration.
    """
    
    # Common settings for the dataloaders
    batch_size = 64
    shuffle = True
    num_workers = 4  # Adjust based on CPU cores
    pin_memory = True

    if tx == "acuity":
        # Visual acuity transformation
        dataset = CVPDataset(base_dir, age=age, transforms_dict=transforms_dict,operation=operation)
    elif tx == "CS":
        # Contrast sensitivity transformation
        dataset = CVPDataset(base_dir, age=age, is_contrast_sensitivity=True,operation=operation)
    elif tx == "none":
        # No transformations
        dataset = CVPDataset(base_dir, transforms_dict=transforms_dict, is_noTransform=True,operation=operation)
    else:
        raise ValueError("Invalid transform. Choose from 'acuity', 'CS', or 'none'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=num_workers, pin_memory=pin_memory)

# Example usage
if __name__ == "__main__":
    # Parameters
    operation = "train"  # Options: "train", "test", "validation"
    tx = "none"  # Options: "acuity", "CS", "none"
    age = 1.0  # Age in months
    batch_size = 64  # Batch size
    shuffle = True  # Shuffle dataset

    dataloader = get_dataloader(operation=operation, tx=tx, age=age, batch_size=batch_size, shuffle=shuffle)

    # Iterate through the DataLoader
    for i, data in enumerate(dataloader):
        print(f"Batch {i + 1}: Shape {data[0].shape}")
        
        if i == 2:  # Stop after 3 batches for demonstration
            break
