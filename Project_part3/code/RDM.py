import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def compute_rdm(activation_list, metric='correlation'):
    # Stack into a [N, D] matrix
    acts = torch.stack(activation_list).numpy()
    # Compute pairwise dissimilarity
    dists = pdist(acts, metric=metric)
    rdm = squareform(dists)  # shape: [N, N]
    return rdm

def compare_rdms(rdm1, rdm2):
    # Flatten the upper triangle (excluding diagonal)
    triu_idx = torch.triu_indices(*torch.tensor(rdm1.shape), offset=1)
    rdm1_flat = rdm1[triu_idx[0], triu_idx[1]]
    rdm2_flat = rdm2[triu_idx[0], triu_idx[1]]

    # Compute Spearman correlation (rank-based)
    rho, pval = spearmanr(rdm1_flat, rdm2_flat)
    return rho, pval

def plot_all_rdms(rdms):
    models = ["no_transform", "acuity", "CS", "shuffle"]
    layers = ["conv_stem", "blocks_4", "conv_head"]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 10))
    fig.suptitle("Representational Dissimilarity Matrices (RDMs)", fontsize=16)

    for row_idx, model in enumerate(models):
        for col_idx, layer in enumerate(layers):
            ax = axes[row_idx, col_idx]
            rdm = rdms.get(model, {}).get(layer, None)

            if rdm is not None:
                im = ax.imshow(rdm, cmap='viridis', vmin=rdm.min(), vmax=rdm.max())
                ax.set_title(f"{model} - {layer}", fontsize=10)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, "Missing", ha='center', va='center')
                ax.axis('off')

     # Add a single colorbar to the side
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
