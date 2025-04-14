import re
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store metrics
epochs = []
losses = []
val_scores = []
test_scores = []

# Read and parse the log file
with open('/home/umflint.edu/koernerg/excelformer/ExcelFormer/training_log_20250317_112802.txt', 'r') as f:
    for line in f:
        # Extract training loss
        if 'Average Loss:' in line:
            loss = float(line.split('Average Loss:')[1].strip())
            losses.append(loss)
            epochs.append(len(losses))  # Use length of losses as epoch number
        
        # Extract validation and test scores
        if 'Validation score:' in line:
            val_score = float(line.split('Validation score:')[1].strip())
            val_scores.append(val_score)
        if 'Test score:' in line:
            test_score = float(line.split('Test score:')[1].strip())
            test_scores.append(test_score)

# Create figure with two subplots
plt.style.use('seaborn')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Training Loss
ax1.plot(epochs, losses, 'b-', label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('ExcelFormer MI Training Loss over Time')
ax1.legend()
ax1.grid(True)

# Plot 2: Validation and Test Scores
epochs_scores = list(range(1, len(val_scores) + 1))
ax2.plot(epochs_scores, val_scores, 'g-', label='Validation Score')
ax2.plot(epochs_scores, test_scores, 'r-', label='Test Score')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score')
ax2.set_title('ExcelFormer MI Validation and Test Scores over Time')
ax2.legend()
ax2.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close() 