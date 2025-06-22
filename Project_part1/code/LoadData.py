import torch
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
                is_noTransform : bool = False):
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
    self.paths = list(Path(self.root_dir).glob('*/*.jpg'))

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
    transform = transforms.ToTensor()
    image_tensor = transform(image_np)
    return image_tensor


  def __len__(self):
    """Mandatory overwrite: Returns the number of images in our root directory"""
    return len(self.paths)

  def __getitem__(self, idx):
    """Mandatory overwrite: Applies the necessary transform"""

    if not self.is_contrast_sensitivity:
      image = Image.open(self.paths[idx])
      if self.age != 0:
        image = self.visual_acuity_transform(image)
    else:
      image = self.get_contrast_sensitivity_transform_tensor(self.paths[idx])

    return image
## -----------------------------------------------------------------------------

root_dir = r'..\data\train' # path to the image dataset
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
    transformed_CVP_dataset = CVPDataset(root_dir, age=age, transforms_dict = transforms_dict, operation='train')
    # Contrast sensitivity
    #transformed_CVP_dataset = CVPDataset(root_dir, age = age, is_contrast_sensitivity= True)
    # No transforms
    #transformed_CVP_dataset = CVPDataset(root_dir, transforms_dict = transforms_dict, is_noTransform = True)
    end_time = timer()
    
    runtime = end_time - start_time
    print(f"runtime is {runtime*1000} ms")

    fig = plt.figure()
    for i, sample in enumerate(transformed_CVP_dataset):
    
      # print(i, sample.shape)
      ax = plt.subplot(1, 4, i + 1)
      plt.tight_layout()
      ax.set_title('Sample #{}'.format(i+1))
      ax.axis('off')
      plt.imshow(sample.permute(1,2,0))
    
      if i == 0:
          plt.show()
          break
