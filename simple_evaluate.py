print("=== SIMPLIFIED EXCELFORMER EVALUATION ===")

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

from bin import ExcelFormer
from simple_excelformer_training import load_simple_data, create_dataset_splits, apply_catboost_encoding

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--sample_size", type=str, choices=['10000', '100000', 'full'], default='full')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get model configuration from checkpoint
    n_features = checkpoint['n_features']
    sample_size = checkpoint['sample_size']
    catenc = checkpoint['catenc']
    
    print(f"Model info: {n_features} features, sample_size={sample_size}, catenc={catenc}")
    
    # Load data (same as training)
    df_clean, train_indices, val_indices, test_indices, metadata = load_simple_data(sample_size)
    splits = create_dataset_splits(df_clean, train_indices, val_indices, test_indices)
    splits = apply_catboost_encoding(splits, catenc)
    
    # Convert to tensors
    data_tensors = {}
    for name, data in splits.items():
        data_tensors[name] = {
            'X': torch.tensor(data['X_num'], dtype=torch.float32).to(device),
            'y': torch.tensor(data['y'], dtype=torch.long).to(device)
        }
    
    # Recreate model
    model = ExcelFormer(
        d_numerical=n_features,
        d_out=2,
        categories=None,
        prenormalization=True,
        token_bias=True,
        n_layers=3,
        n_heads=32,
        d_token=256,
        attention_dropout=0.3,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        kv_compression=None,
        kv_compression_sharing=None,
        init_scale=0.01,
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    
    # Evaluate on validation and test sets
    results = {}
    
    with torch.no_grad():
        for name in ['val', 'test']:
            X = data_tensors[name]['X']
            y_true = data_tensors[name]['y'].cpu().numpy()
            
            # Get predictions
            logits = model(X, None)
            y_pred_proba = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            # Calculate metrics
            auc = roc_auc_score(y_true, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            
            results[name] = {
                'auc': auc,
                'fpr': fpr,
                'tpr': tpr,
                'y_true': y_true,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"{name.capitalize()} AUC: {auc:.4f}")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    for name, data in results.items():
        color = 'blue' if name == 'val' else 'red'
        label = f'{name.capitalize()} ROC (AUC = {data["auc"]:.3f})'
        plt.plot(data['fpr'], data['tpr'], color=color, lw=2, label=label)
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    size_str = f"{sample_size:,}" if sample_size != 'full' else "Full"
    plt.title(f'ExcelFormer ROC Curve - {size_str} Samples')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save plot
    plot_filename = f'roc_curve_excelformer_simple_{sample_size}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nROC curve saved as: {plot_filename}")
    
    # Print training info from checkpoint
    print(f"\nTraining info:")
    print(f"  Best validation AUC: {checkpoint['best_val_auc']:.4f}")
    print(f"  Final test AUC: {checkpoint['final_test_auc']:.4f}")
    print(f"  Training epoch: {checkpoint['epoch']}")
    print(f"  Sample size: {checkpoint['sample_size']}")
    print(f"  CatBoost encoding: {checkpoint['catenc']}")

if __name__ == "__main__":
    main() 