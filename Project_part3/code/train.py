import torch
from torch.utils.data import DataLoader
from engine import engine
import numpy as np
from LoadData import get_dataloader

if __name__ == '__main__':
    
    '''this code illustrates acuity curriculum, 
    for any other curriculum code can be modified from here.
    the possible operations: 'train' and 'val'
    the possible transformations tx: 'acuity' , 'CS', 'none'
    the epochs mentioned in the engine call are incremental
    '''

    # Dictionary to store the dataloaders
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Create a dataloader
    train_dataloader_acuity_1 = get_dataloader(operation="train", 
                                                  tx="acuity", age=1.0, 
                                                  batch_size=64, shuffle=True)
    
    train_dataloader_acuity_5 = get_dataloader(operation="train", 
                                                  tx="acuity", age=5.0, 
                                                  batch_size=64, shuffle=True)

    train_dataloader_acuity_13 = get_dataloader(operation="train", 
                                                  tx="acuity", age=13.0, 
                                                  batch_size=64, shuffle=True)                                              
    
    val_dataloader_acuity_1 = get_dataloader(operation="val", 
                                                  tx="acuity", age=1.0, 
                                                  batch_size=64, shuffle=True)
    
    val_dataloader_acuity_5 = get_dataloader(operation="val", 
                                                  tx="acuity", age=5.0, 
                                                  batch_size=64, shuffle=True)

    val_dataloader_acuity_13 = get_dataloader(operation="val", 
                                                  tx="acuity", age=13.0, 
                                                  batch_size=64, shuffle=True)

    print(f"train dataloaders are ready")
    print(f"val dataloaders are ready")

    # Train the model
    Evaluation_dict = engine(
        train_dataloader=train_dataloader_acuity_1,
        test_dataloader=val_dataloader_acuity_1,
        epochs=10, 
        class_names=class_names,
        checkpoint_path = "acuity_curriculum_model.pth"
    )

    Evaluation_dict = engine(
        train_dataloader=train_dataloader_acuity_5,
        test_dataloader=val_dataloader_acuity_5,
        epochs=25, 
        class_names=class_names,
        checkpoint_path = "acuity_curriculum_model.pth"
    )

    Evaluation_dict = engine(
        train_dataloader=train_dataloader_acuity_13,
        test_dataloader=val_dataloader_acuity_13,
        epochs=65, 
        class_names=class_names,
        checkpoint_path = "acuity_curriculum_model.pth"
    )
