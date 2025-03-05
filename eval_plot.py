import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy.special
import sklearn.metrics

from lib.data import Dataset, Transformations, build_dataset, prepare_tensors
from lib.metrics import calculate_metrics
from lib.util import TaskType
from bin.excel_former import ExcelFormer

def main():
    # Load the saved model state
    model_path = 'result/ExcelFormer/default/mixup(hidden_mix)/android_security/42/500/pytorch_model.pt'
    checkpoint = torch.load(model_path)
    model_state = checkpoint['model_state_dict']
    
    # Load dataset EXACTLY as in training
    dataset = build_dataset(
        Path('DATA/android_security'),
        Transformations(normalization='quantile'),  # Start simple, no cat encoding yet
        cache=False
    )

    print(f"Initial dataset shape: {dataset.X_num['train'].shape}")
    print(f"First few feature values:\n{dataset.X_num['train'][:5, :5]}")

    # Initialize model with the same number of features as the trained model
    n_features = next(iter(model_state.values())).shape[0]
    print(f"Number of features in saved model: {n_features}")

    model = ExcelFormer(
        d_numerical=n_features,
        categories=None,
        d_out=2,
        token_bias=True,
        n_layers=3,
        d_token=256,
        n_heads=32,
        attention_dropout=0.3,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        prenormalization=True,
        kv_compression=None,
        kv_compression_sharing=None
    )
    
    # Load the saved state
    model.load_state_dict(model_state)
    model.eval()

    # Prepare tensors
    X_num, X_cat, Y = prepare_tensors(dataset, 'cuda' if torch.cuda.is_available() else 'cpu')

    # Get predictions
    with torch.no_grad():
        outputs = model(X_num['test'], None)
        y_pred = outputs.cpu().numpy()

    # Get true labels and calculate probabilities
    y_true = Y['test'].cpu().numpy()
    y_pred_proba = scipy.special.softmax(y_pred, axis=1)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred_proba)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred_proba)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()

if __name__ == '__main__':
    main()