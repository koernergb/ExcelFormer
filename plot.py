import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy.special
import sklearn.metrics
from sklearn.feature_selection import mutual_info_classif
import os

from lib.data import Dataset, Transformations, build_dataset, prepare_tensors 
from lib.metrics import calculate_metrics
from lib.util import TaskType
from bin.excel_former import ExcelFormer

def main():
    # Load the saved model state
    model_path = 'result/ExcelFormer/default/mixup(hidden_mix)/android_security/42/500/pytorch_model.pt'
    checkpoint = torch.load(model_path)
    model_state = checkpoint['model_state_dict']
    
    # First load and prepare dataset WITH categorical encoding
    dataset = build_dataset(
        Path('DATA/android_security'),
        Transformations(
            normalization='quantile',
            cat_encoding='counter',
            seed=42
        ),
        cache=False
    )

    # Select features
    feature_indices = np.array([ 8, 3, 13, 48, 7, 0, 45, 1, 40, 15, 39, 36, 37, 43, 6, 11, 38, 35, 4, 46, 47, 30, 32, 17, 5, 2, 12])
    dataset.X_num = {k: v[:, feature_indices] for k, v in dataset.X_num.items()}

    # Set device and prepare model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ExcelFormer(
        d_numerical=27,
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
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()

    # Prepare data
    X_num, X_cat, Y = prepare_tensors(dataset, device)

    # Process test set in batches
    batch_size = 1024
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for i in range(0, len(X_num['test']), batch_size):
            batch = X_num['test'][i:i + batch_size]
            outputs = model(batch, None)
            all_preds.append(outputs.cpu().numpy())
            all_true.append(Y['test'][i:i + batch_size].cpu().numpy())
    
    # Combine predictions
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    
    # Calculate probabilities and ROC
    y_pred_proba = scipy.special.softmax(y_pred, axis=1)[:, 1]
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