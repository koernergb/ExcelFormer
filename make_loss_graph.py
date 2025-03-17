import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Initialize storage for metrics
    epochs = []
    train_losses = []
    val_scores = []
    test_scores = []
    
    current_epoch = None
    current_loss = None
    
    for line in lines:
        line = line.strip()
        
        # Parse epoch info
        if line.startswith('Epoch '):
            try:
                current_epoch = int(line.split()[1].zfill(3))
            except:
                continue
                
        # Parse average loss
        elif 'Average Loss:' in line:
            current_loss = float(line.split(': ')[1])
            
        # Parse validation and test scores
        elif 'Validation score:' in line:
            val_score = float(line.split(': ')[1])
            
        elif 'Test score:' in line:
            test_score = float(line.split(': ')[1])
            
            # Store complete epoch data
            if current_epoch is not None and current_loss is not None:
                epochs.append(current_epoch)
                train_losses.append(current_loss)
                val_scores.append(val_score)
                test_scores.append(test_score)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_scores': val_scores,
        'test_scores': test_scores
    }

def main():
    log_path = '/home/umflint.edu/koernerg/excelformer/ExcelFormer/training_log_20250305_163704.txt'
    metrics = parse_training_log(log_path)
    
    # Print summary
    print(f"Total epochs: {len(metrics['epochs'])}")
    print(f"\nFinal metrics:")
    print(f"Train loss: {metrics['train_losses'][-1]:.4f}")
    print(f"Val score: {metrics['val_scores'][-1]:.4f}")
    print(f"Test score: {metrics['test_scores'][-1]:.4f}")
    
    # Print best validation score
    best_val_idx = metrics['val_scores'].index(max(metrics['val_scores']))
    print(f"\nBest validation score:")
    print(f"Epoch {metrics['epochs'][best_val_idx]}")
    print(f"Val score: {metrics['val_scores'][best_val_idx]:.4f}")
    print(f"Test score: {metrics['test_scores'][best_val_idx]:.4f}")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot training loss
    ax1.plot(metrics['epochs'], metrics['train_losses'], 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Time')
    ax1.grid(True)
    ax1.legend()

    # Plot validation and test scores
    ax2.plot(metrics['epochs'], metrics['val_scores'], 'g-', label='Validation Score')
    ax2.plot(metrics['epochs'], metrics['test_scores'], 'r-', label='Test Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation and Test Scores over Time')
    ax2.grid(True)
    ax2.legend()

    # Highlight best validation epoch
    ax2.axvline(x=metrics['epochs'][best_val_idx], color='k', linestyle='--', alpha=0.3)
    ax2.text(metrics['epochs'][best_val_idx], 0.5, f'Best Val\nEpoch {metrics["epochs"][best_val_idx]}', 
             rotation=90, verticalalignment='center')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == '__main__':
    main()