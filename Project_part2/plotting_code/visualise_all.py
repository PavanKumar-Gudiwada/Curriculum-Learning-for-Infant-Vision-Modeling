import torch
import matplotlib.pyplot as plt




###-------------------------------------------------------------
# Load the checkpoint
checkpoint_path = "../networks/acuity_curriculum_model.pth"
checkpoint = torch.load(checkpoint_path)

# Extract start_epoch and evaluation dictionary
start_epoch = checkpoint['epoch'] + 1
evaluation_dict_acuity = checkpoint['evaluation_dict']
###-------------------------------------------------------------
# Load the checkpoint
checkpoint_path = "../networks/CS_curriculum_model.pth"
checkpoint = torch.load(checkpoint_path)

# Extract start_epoch and evaluation dictionary
start_epoch = checkpoint['epoch'] + 1
evaluation_dict_CS = checkpoint['evaluation_dict']
###-------------------------------------------------------------
# Load the checkpoint
checkpoint_path = "../networks/no_transform_curriculum_model.pth"
checkpoint = torch.load(checkpoint_path)

# Extract start_epoch and evaluation dictionary
start_epoch = checkpoint['epoch'] + 1
evaluation_dict_notf = checkpoint['evaluation_dict']
###-------------------------------------------------------------
# Load the checkpoint
checkpoint_path = "../networks/Shuffle_curriculum_model.pth"
checkpoint = torch.load(checkpoint_path)

# Extract start_epoch and evaluation dictionary
start_epoch = checkpoint['epoch'] + 1
evaluation_dict_shuffle = checkpoint['evaluation_dict']
###-------------------------------------------------------------

# list_of_dicts = [evaluation_dict_notf, evaluation_dict_CS, evaluation_dict_acuity, evaluation_dict_shuffle]




figs, axes = plt.subplots(nrows=2, ncols= 4, figsize= (20,10))
legend_fontsize = 12

####---------------------------------------------------------------
# Plot training and validation loss for No Transform
ax = axes[0,0]
ax.plot(range(len(evaluation_dict_notf['train_loss'])), evaluation_dict_notf['train_loss'], label='Train Loss', color='blue')
ax.plot(range(len(evaluation_dict_notf['val_loss'])), evaluation_dict_notf['val_loss'], label='Validation Loss', color='orange')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss Over Epochs - No Transformations')
ax.legend(fontsize=legend_fontsize)
ax.grid(True)

# Create a figure for accuracy for No Transform
ax = axes[1,0]
ax.plot(range(len(evaluation_dict_notf['train_acc'])), evaluation_dict_notf['train_acc'], label='Train Accuracy', color='green')
ax.plot(range(len(evaluation_dict_notf['val_acc'])), evaluation_dict_notf['val_acc'], label='Validation Accuracy', color='red')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Over Epochs - No Transformations')
ax.legend(fontsize=legend_fontsize)
ax.grid(True)

####---------------------------------------------------------------
# Helper function to add vertical lines dynamically
def add_vlines(ax, lines, ymin, ymax):
    for line in lines:
        ax.vlines(x=line[0], ymin=ymin, ymax=ymax, colors=line[1], linestyles="dashed", label=line[2])

# Define vline configurations
vline_config_acuity = [(10, "r", "10 epochs: 1 mo"), (25, "g", "15 epochs: 5 mo"), (65, "b", "40 epochs: 13 mo")]
vline_config_CS = [(10, "r", "10 epochs: 3 mo"), (25, "g", "15 epochs: 7 mo"), (65, "b", "40 epochs: 13 mo")]
vline_config_shuffle = [(5, "r", "CS- 5 epochs: 3 mo"), (15, "g", "Acuity- 10 epochs: 3 mo"),
                        (35, "b", "CS- 20 epochs: 13 mo"), (65, "brown", "Acuity- 30 epochs: 13 mo")]

####---------------------------------------------------------------
# Plot training and validation loss for Acuity
ax = axes[0, 1]
ax.plot(range(len(evaluation_dict_acuity['train_loss'])), evaluation_dict_acuity['train_loss'], label='Train Loss', color='blue')
ax.plot(range(len(evaluation_dict_acuity['val_loss'])), evaluation_dict_acuity['val_loss'], label='Validation Loss', color='orange')
ymin, ymax = ax.get_ylim()
add_vlines(ax, vline_config_acuity, ymin, ymax)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss Over Epochs - Acuity Curriculum')
ax.legend(fontsize=legend_fontsize)
ax.grid(True)

# Plot accuracy for Acuity
ax = axes[1, 1]
ax.plot(range(len(evaluation_dict_acuity['train_acc'])), evaluation_dict_acuity['train_acc'], label='Train Accuracy', color='green')
ax.plot(range(len(evaluation_dict_acuity['val_acc'])), evaluation_dict_acuity['val_acc'], label='Validation Accuracy', color='red')
ymin, ymax = ax.get_ylim()
add_vlines(ax, vline_config_acuity, ymin, ymax)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Over Epochs - Acuity Curriculum')
ax.legend(fontsize=legend_fontsize)
ax.grid(True)

####---------------------------------------------------------------
# Plot training and validation loss for CS
ax = axes[0, 2]
ax.plot(range(len(evaluation_dict_CS['train_loss'])), evaluation_dict_CS['train_loss'], label='Train Loss', color='blue')
ax.plot(range(len(evaluation_dict_CS['val_loss'])), evaluation_dict_CS['val_loss'], label='Validation Loss', color='orange')
ymin, ymax = ax.get_ylim()
add_vlines(ax, vline_config_CS, ymin, ymax)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss Over Epochs - CS Curriculum')
ax.legend(fontsize=legend_fontsize)
ax.grid(True)

# Plot accuracy for CS
ax = axes[1, 2]
ax.plot(range(len(evaluation_dict_CS['train_acc'])), evaluation_dict_CS['train_acc'], label='Train Accuracy', color='green')
ax.plot(range(len(evaluation_dict_CS['val_acc'])), evaluation_dict_CS['val_acc'], label='Validation Accuracy', color='red')
ymin, ymax = ax.get_ylim()
add_vlines(ax, vline_config_CS, ymin, ymax)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Over Epochs - CS Curriculum')
ax.legend(fontsize=legend_fontsize)
ax.grid(True)

####---------------------------------------------------------------
# Plot training and validation loss for Shuffle Curriculum
ax = axes[0, 3]
ax.plot(range(len(evaluation_dict_shuffle['train_loss'])), evaluation_dict_shuffle['train_loss'], label='Train Loss', color='blue')
ax.plot(range(len(evaluation_dict_shuffle['val_loss'])), evaluation_dict_shuffle['val_loss'], label='Validation Loss', color='orange')
ymin, ymax = ax.get_ylim()
add_vlines(ax, vline_config_shuffle, ymin, ymax)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Loss Over Epochs - Shuffle Curriculum')
ax.legend(fontsize=legend_fontsize)
ax.grid(True)

# Plot accuracy for Shuffle Curriculum
ax = axes[1, 3]
ax.plot(range(len(evaluation_dict_shuffle['train_acc'])), evaluation_dict_shuffle['train_acc'], label='Train Accuracy', color='green')
ax.plot(range(len(evaluation_dict_shuffle['val_acc'])), evaluation_dict_shuffle['val_acc'], label='Validation Accuracy', color='red')
ymin, ymax = ax.get_ylim()
add_vlines(ax, vline_config_shuffle, ymin, ymax)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Over Epochs - Shuffle Curriculum')
ax.legend(fontsize=legend_fontsize)
ax.grid(True)

# Show the plots
plt.tight_layout()
plt.show()