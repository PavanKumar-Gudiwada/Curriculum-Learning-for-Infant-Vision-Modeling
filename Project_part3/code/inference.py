import os
import torch
import random
from pathlib import Path
from torch import nn
from LoadData import get_dataloader, transforms_dict
from CustomDataset import CVPDataset
from torchvision.models import efficientnet_b2
from torch.utils.data import DataLoader, Subset, ConcatDataset
from RDM import compute_rdm, compare_rdms, plot_all_rdms

if __name__ == '__main__':

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model paths
    # This gets the absolute path to the current script
    current_dir = Path(__file__).resolve().parent

    # Go up to the project root and then into networks
    networks_dir = current_dir.parent / "networks"

    model_paths = {
        "no_transform": networks_dir / "no_transform_curriculum_model.pth",
        "acuity": networks_dir / "acuity_curriculum_model.pth",
        "CS": networks_dir / "CS_curriculum_model.pth",
        "shuffle": networks_dir / "Shuffle_curriculum_model.pth"
    }

    # Dictionary to store loaded models
    models = {}
    activations = {}

    # Create a hook function
    def get_hook(activations_dict, name):
        def hook(model, input, output):
            acts = output.detach().cpu()

            # If it's a batch, iterate over each sample
            if acts.ndim == 4:  # [B, C, H, W]
                for i in range(acts.size(0)):
                    activations_dict[name].append(acts[i].flatten())
            else:  # single sample [C, H, W]
                activations_dict[name].append(acts.flatten())

        return hook

    for name, path in model_paths.items():
        if path.exists():
            print(f"Loading model: {name} from {path}")
            checkpoint = torch.load(str(path))

            model = efficientnet_b2()
            model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1408, out_features=10, bias=True)
            )
            model = model.to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models[name] = model

            # Create a dictionary to hold activations for this model
            activations[name] = {'conv_stem': [], 'blocks_4': [], 'conv_head': []}

            # Register hooks
            model.features[0].register_forward_hook(get_hook(activations[name], 'conv_stem'))  # conv_stem
            model.features[4].register_forward_hook(get_hook(activations[name], 'blocks_4'))   # 5th block
            model.features[-1].register_forward_hook(get_hook(activations[name], 'conv_head')) # final conv
        else:
            print(f"Model not found at: {path}")
            exit()

    # Input data
    # Set seed for reproducibility (optional)
    random.seed(42)

    # Constants
    base_dir = r'..\project_part3\tiny-imagenet-10'
    num_no_transform = 50
    num_acuity = 25
    num_cs = 25
    batch_size = 32  # or however many you want per batch

    print("Current working directory:", os.getcwd())
    # Load full datasets
    no_transform_dataset = CVPDataset(base_dir, is_noTransform=True, transforms_dict=transforms_dict, operation="train")
    acuity_dataset = CVPDataset(base_dir, age=1.0, transforms_dict=transforms_dict, operation="train")
    cs_dataset = CVPDataset(base_dir, age=1.0, is_contrast_sensitivity=True, transforms_dict=transforms_dict, operation="train")

    # Randomly sample subsets
    indices_no_transform = random.sample(range(len(no_transform_dataset)), num_no_transform)
    indices_acuity = random.sample(range(len(acuity_dataset)), num_acuity)
    indices_cs = random.sample(range(len(cs_dataset)), num_cs)

    subset_no_transform = Subset(no_transform_dataset, indices_no_transform)
    subset_acuity = Subset(acuity_dataset, indices_acuity)
    subset_cs = Subset(cs_dataset, indices_cs)

    # Combine all subsets
    combined_dataset = ConcatDataset([subset_no_transform, subset_acuity, subset_cs])

    # Final dataloader
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Example: get first batch
    for batch in combined_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        print("Batch size:", images.size(), "Labels:", labels)
        
        #pass the input data through all models    
        for model in models.values():
            _ = model(images)

    #compute RDMs for each model and layer
    rdms = {"no_transform": {}, "acuity": {}, "CS": {}, "shuffle": {}}
    for model_name, act_dict in activations.items():
        for layer_name, act_list in act_dict.items():
            if act_list:
                rdm = compute_rdm(act_list)
                rdms[model_name][layer_name] = rdm
                print(f"RDM for {model_name} at {layer_name}: {rdm.shape}")

    plot_all_rdms(rdms)

    # Models to compare with no_transform
    targets = ["acuity", "CS", "shuffle"]
    layers = ["conv_stem", "blocks_4", "conv_head"]

    print("\n=== RSA: no_transform vs others ===")
    for target_model in targets:
        for layer in layers:
            rdm_ref = rdms["no_transform"].get(layer)
            rdm_cmp = rdms[target_model].get(layer)

            if rdm_ref is not None and rdm_cmp is not None:
                rho, _ = compare_rdms(rdm_ref, rdm_cmp)
                print(f"RSA (no_transform vs {target_model}) at {layer}: {rho:.3f}")
            else:
                print(f"Missing RDMs for layer {layer} in {target_model}")
