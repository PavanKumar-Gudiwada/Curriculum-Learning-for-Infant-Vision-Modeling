import gc
import os
import torch
from torch import nn
from torchvision.models import efficientnet_b2
from torch.cuda.amp import GradScaler, autocast
from timeit import default_timer as timer
from tqdm.auto import tqdm

# Accuracy function
def accuracy_fn(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def validate_model(model, test_dataloader, loss_fn, device):
    model.eval()
    val_loss = 0
    val_accuracy = 0

    with torch.inference_mode():
        for batch in test_dataloader:
            test_batch, label_batch = batch
            test_batch, label_batch = test_batch.to(device), label_batch.to(device)

            y_logits = model(test_batch)
            loss = loss_fn(y_logits, label_batch)
            val_loss += loss.item()

            y_pred = torch.argmax(y_logits, dim=1)
            val_accuracy += accuracy_fn(y_pred, label_batch)

    val_loss /= len(test_dataloader)
    val_accuracy /= len(test_dataloader)
    
    return val_loss, val_accuracy

def engine(class_names, train_dataloader, test_dataloader, epochs=5, checkpoint_path="model_checkpoint.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = efficientnet_b2()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1408, out_features=len(class_names), bias=True)
    )
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    # For mixed precision training
    scaler = GradScaler()
    
    Evaluation_dict = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        Evaluation_dict = checkpoint['evaluation_dict']
    else:
        start_epoch = 0

    start_time = timer()

    for epoch in tqdm(range(start_epoch,epochs)):
        epochStart = timer()
        # Train Step
        model.train()
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_accuracy = 0

        for batch in train_dataloader:
            train_batch, label_batch = batch
            train_batch, label_batch = train_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):  # Mixed precision training
                y_logits = model(train_batch)
                loss = loss_fn(y_logits, label_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            y_pred = torch.argmax(y_logits, dim=1)
            train_acc += accuracy_fn(y_pred, label_batch)

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        Evaluation_dict["train_loss"].append(train_loss)
        Evaluation_dict["train_acc"].append(train_acc)
        
        val_loss, val_accuracy = validate_model(model, test_dataloader, loss_fn, device)    

        Evaluation_dict["val_loss"].append(val_loss)
        Evaluation_dict["val_acc"].append(val_accuracy)
        
        epochEnd = timer()
        
        print(f"Epoch {epoch + 1}/{epochs} in {epochEnd-epochStart}s")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")        
        
        # Save model once in every 10epochs or all epochs done
        save_now = ((epoch + 1) % 10 == 0 or epoch == epochs - 1)
        
        if save_now == True:
            
            saveStrt = timer()
            
            # Save the model after every epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'evaluation_dict': Evaluation_dict
            }, checkpoint_path)
            
            saveEnd = timer()
            
            print(f'saved in {saveEnd-saveStrt}s')
        
        gc.collect()
        torch.cuda.empty_cache()
        
    end_time = timer()

    print(f"Total training time: {end_time - start_time:.2f} seconds")

    return Evaluation_dict
