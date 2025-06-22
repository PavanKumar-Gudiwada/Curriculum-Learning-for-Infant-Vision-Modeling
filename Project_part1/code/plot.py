import torch
import matplotlib.pyplot as plt
from LoadData import CVPDataset, transforms_dict

root_dir = r'..\data\train' # path to the image dataset

def get_subplot(dataset: torch.utils.data.Dataset, 
                age: float, 
                col: int,
                title: str):
    for i, sample in enumerate(dataset):
        # # Determine the row and column indices
        # ax = plt.subplot(5, 6, col * 5 + i + 1)

        ax = axes[i,col]         # Access the subplot with transposed indices
        
        ax.axis('off')
        
        # Adjust the sample format if necessary
        img = sample.permute(1, 2, 0)  # Assuming image format is [C, H, W]
        ax.imshow(img)
        
        if i == 0:
          ax.set_title(title + f"\nAge: {age} months")

        # Break if the column is filled
        if i == 4:
            break

fig = plt.figure()
  
visual_ac_1= CVPDataset(root_dir, age=1, transforms_dict = transforms_dict)
visual_ac_3= CVPDataset(root_dir, age=3, transforms_dict = transforms_dict)
visual_ac_7= CVPDataset(root_dir, age=7, transforms_dict = transforms_dict)
contrast_1 = CVPDataset(root_dir, age = 1, is_contrast_sensitivity= True)
contrast_7 = CVPDataset(root_dir, age = 7, is_contrast_sensitivity= True)
contrast_13 = CVPDataset(root_dir, age = 13, is_contrast_sensitivity= True)

rows, cols = 5, 6
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

# dataset, age, column number, title
datasets_and_columns = [
    (visual_ac_1, 1.0, 0, "Visual Acuity"),
    (visual_ac_3, 3.0, 1, "Visual Acuity"),
    (visual_ac_7, 7.0, 2, "Visual Acuity"),
    (contrast_1, 1.0, 3, "Contrast Sensitivity"),
    (contrast_7, 7.0, 4, "Contrast Sensitivity"),
    (contrast_13, 13.0, 5, "Contrast Sensitivity")
]

# Fill each column with images from the corresponding dataset
for dataset, age, col, title in datasets_and_columns: 
    get_subplot(dataset, age, col, title)
    
plt.show()
