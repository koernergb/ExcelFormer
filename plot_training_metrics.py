import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--log_file", type=str, required=True)
parser.add_argument("--sample_size", type=int, default=None)
args = parser.parse_args()

size_str = f"{args.sample_size:,}" if args.sample_size else "Full"

# Initialize lists to store metrics
epochs = []
losses = []
val_scores = []

# Read and parse the log file
with open(args.log_file, 'r') as f:
    for line in f:
        # Extract training loss
        if 'Average Loss:' in line:
            loss = float(line.split('Average Loss:')[1].strip())
            losses.append(loss)
            epochs.append(len(losses))  # Use length of losses as epoch number
        
        # Extract only validation score
        if 'Validation score:' in line:
            val_score = float(line.split('Validation score:')[1].strip())
            val_scores.append(val_score)

# Create figure with two subplots
plt.style.use('seaborn')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Training Loss
ax1.plot(epochs, losses, 'b-', label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title(f'ExcelFormer Training Loss over Time - {size_str} Samples')
ax1.legend()
ax1.grid(True)

# Plot 2: Validation Score
epochs_scores = list(range(1, len(val_scores) + 1))
ax2.plot(epochs_scores, val_scores, 'g-', label='Validation Score')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score')
ax2.set_title(f'ExcelFormer Validation Score over Time - {size_str} Samples')
ax2.legend()
ax2.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig(f'training_metrics_excelformer_{size_str.replace(",", "")}.png')
plt.close() 