import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pathlib
import cv2 as cv
from pathlib import Path
from PIL import Image
from timeit import default_timer as timer

# importing the get_contrast_sensitivity_transform from contrast_sensitivity.py
from contrast_sensitivity import get_contrast_sensitivity_transform

## -----------------------------------------------------------------------------
class CVPDataset(Dataset):
  def __init__(self, root_dir,
                age: float = 1.0,
                transforms_dict = None,
                is_contrast_sensitivity : bool = False,
                is_noTransform : bool = False,
                operation = "train"):
    """
    Subclasses torch.utils.data.Dataset class, and returns dataset with visual acuity transform

    Arguments:
      root_dir (string): Train or Test directory containing images.
      age: Age in months
      transform_dict : A dictionary of transforms for each age group
      is_contrast_sensitivity : Whether to apply contrast sensitivity transform or not, if False, then visual acuity transform is applied

    Returns:
      transformed image
    """
    self.root_dir = root_dir
    self.age = age
    self.is_noTransform = is_noTransform
    self.is_contrast_sensitivity = is_contrast_sensitivity
    self.transforms_dict = transforms_dict
    self.visual_acuity_transform = self.get_visual_acuity_transform()
    self.operation = operation
    
    # Get all image paths and corresponding labels
    self.paths = []
    self.class_idx = []
    
    current_dir = Path(__file__).resolve().parent

    if self.operation == "train":
        # Read the CSV files
        csv_path = current_dir / 'train_images_with_classes_10.csv'
        train_df = pd.read_csv(csv_path)

        # Extract columns into separate arrays
        imageNames = train_df['Image Name'].values
        classNames = train_df['Class Name'].values
        classIdx = train_df['Class Index'].values

        # Ensure base_dir_array is a NumPy array of strings
        base_dir_array = np.array([self.root_dir + '/train/'] * len(imageNames), dtype=np.str_)

        # Correctly create the class directory part for each class, ensuring it's a NumPy array of strings
        class_dir_array = np.array(classNames + '/images/', dtype=np.str_)

        imageNames = np.array(imageNames, dtype=np.str_)

        # Create paths using NumPy's vectorized operations
        trainPaths = np.char.add(np.char.add(base_dir_array, class_dir_array),imageNames)
        self.paths = trainPaths
        self.class_idx = classIdx
    else:
        # Read the CSV files
        csv_path = current_dir / 'val_images_with_classes_10.csv'
        val_df = pd.read_csv(csv_path)

        # Extract columns into separate arrays
        imageNames = val_df['Image Name'].values
        classNames = val_df['Class Name'].values
        classIdx = val_df['Class Index'].values

        # Ensure base_dir_array is a NumPy array of strings
        base_dir_array = np.array([self.root_dir + '/val/'] * len(imageNames), dtype=np.str_)

        # Correctly create the class directory part for each class, ensuring it's a NumPy array of strings
        class_dir_array = np.array(classNames + '/images/', dtype=np.str_)

        imageNames = np.array(imageNames, dtype=np.str_)

        # Create paths using NumPy's vectorized operations
        valPaths = np.char.add(np.char.add(base_dir_array, class_dir_array),imageNames)
        
        self.paths = valPaths
        self.class_idx = classIdx
                
    print(f"{len(self.paths)} images found")

  def get_visual_acuity_transform(self):
    """Applies visual acuity transform"""

    if not self.is_contrast_sensitivity:
      # sigma factor for applying visual acuity transform
      sigma = 0
      if self.is_noTransform:
          sigma = 0
          return self.transforms_dict[sigma]
      else:
          if self.age > 6.0:
            sigma = 0
          elif (self.age >4.5) & (self.age <=6.0):
            sigma = 1
          elif (self.age >2.5) & (self.age <=4.5):
            sigma = 2
          elif (self.age >1.5) & (self.age <=2.5):
            sigma = 3
          elif (self.age >0.0) & (self.age <=1.5):
            sigma = 4
          else:
              return transforms.ToTensor()
          
          return self.transforms_dict[sigma]

  def get_contrast_sensitivity_transform_tensor(self, image_path):
    """Applies contrast sensitivity transform and converts the image to tensor"""

    image_np = get_contrast_sensitivity_transform(self.age, image_path)
    # Convert NumPy array to PyTorch tensor
    image_tensor = torch.from_numpy(image_np).float()

    # Rearrange dimensions to (C, H, W) format if needed
    # Assuming image_np has dimensions (H, W, C)
    image_tensor = image_tensor.permute(2, 0, 1)

    return image_tensor


  def __len__(self):
    """Mandatory overwrite: Returns the number of images in our root directory"""
    return len(self.paths)

  def __getitem__(self, idx):
    """Mandatory overwrite: Applies the necessary transform"""
    image_path = self.paths[idx]

    if not self.is_contrast_sensitivity:
      image = Image.open(image_path)
      
      # Ensure image is RGB
      if image.mode != 'RGB':
          image = image.convert('RGB')
          
      if self.age != 0:
        image = self.visual_acuity_transform(image)
    else:
      image = self.get_contrast_sensitivity_transform_tensor(self.paths[idx])
    
    if self.operation in ["train", "val"]:
        return image, self.class_idx[idx]
    else:
        return image
## -----------------------------------------------------------------------------

root_dir = r'..\tiny-imagenet-200' # path to the image dataset
transforms_dict = {
    0: transforms.Compose([transforms.ToTensor()]) ,
    1: transforms.Compose([transforms.GaussianBlur(kernel_size= 15, sigma = 1), transforms.ToTensor()]),
    2: transforms.Compose([transforms.GaussianBlur(kernel_size= 15, sigma = 2), transforms.ToTensor()]),
    3: transforms.Compose([transforms.GaussianBlur(kernel_size= 15, sigma = 3), transforms.ToTensor()]),
    4: transforms.Compose([transforms.GaussianBlur(kernel_size= 15, sigma = 4), transforms.ToTensor()])
    }

## Plotting --------------------------------------------------------------------
if __name__ == "__main__": # execute below code only if this file is run

    """ Manipulate the transform and age in months here: ------->"""
    age = 1.0 # maipulate age from here
    start_time = timer()
    # visual acuity
    transformed_CVP_dataset = CVPDataset(root_dir, age=age, transforms_dict = transforms_dict)
    # Contrast sensitivity
    #transformed_CVP_dataset = CVPDataset(root_dir, age = age, is_contrast_sensitivity= True)
    # No transforms
    #transformed_CVP_dataset = CVPDataset(root_dir, transforms_dict = transforms_dict, is_noTransform = True)
    end_time = timer()
    
    runtime = end_time - start_time
    print(f"runtime is {runtime*1000} ms")

    fig = plt.figure()
    for i, sample in enumerate(transformed_CVP_dataset):
        
        # Extract image and label if in training mode
        if transformed_CVP_dataset.operation == "train":
            image, label = sample
        else:
            image = sample
            
        # print(i, sample.shape)
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('sample')
        ax.axis('off')
        plt.imshow(sample.permute(1,2,0))
          
        if i == 0:
            plt.show()
            break
