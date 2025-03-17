import sys
import os
import math
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import CatBoostEncoder
from tqdm import tqdm
from pathlib import Path

from bin import ExcelFormer
from lib import Transformations, build_dataset, prepare_tensors, make_optimizer, DATA

def get_args():
    parser = argparse.ArgumentParser()
    # With 25 features (no unnamed: 0 or pkgname)
    parser.add_argument("--model_path", type=str, default='./result/ExcelFormer/default/mixup(none)/android_security/42/pytorch_model.pt')
    # Original with wrong features
    # parser.add_argument("--model_path", type=str, default='./result/ExcelFormer/default/mixup(hidden_mix)/android_security/42/500/pytorch_model.pt')
    parser.add_argument("--dataset", type=str, default='android_security')
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--catenc", action='store_true', default=True)
    return parser.parse_args()

def main():
    args = get_args()
    
    # Create timestamped filename for results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'evaluation_results_{timestamp}.txt'

    # Create a function to write to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # Redirect stdout to our logger
    sys.stdout = Logger(output_file)

    print(f"Starting evaluation at {timestamp}")
    print(f"Saving results to: {output_file}")
    print(f"Loading model from: {args.model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    print("\nSaved Metrics from Training:")
    print(f"Best validation score: {checkpoint['best_score']:.4f}")
    print(f"Best test score: {checkpoint['best_test_score']:.4f}")
    print(f"Final test score: {checkpoint['final_test_score']:.4f}")
    
    # Load dataset
    transformation = Transformations(
        normalization=args.normalization if args.normalization != '__none__' else None
    )
    dataset = build_dataset(DATA / args.dataset, transformation, cache=False)
    
    print("\n=== Initial Data Analysis ===")
    print("Number of samples:", len(dataset.y['test']))
    print("Number of features:", dataset.n_num_features + dataset.n_cat_features)
    print("Class distribution:", np.bincount(dataset.y['test'].astype(int)))

    print("\n=== Feature Processing Details ===")
    print("Number of numerical features:", dataset.n_num_features)
    print("Number of categorical features:", dataset.n_cat_features)
    
    print("\n=== Initial Feature Shapes ===")
    print(f"Initial numerical features: {dataset.X_num['test'].shape}")
    if dataset.X_cat is not None:
        print(f"Initial categorical features: {dataset.X_cat['test'].shape}")
    
    # Convert to float32
    if dataset.X_num['test'].dtype == np.float64:
        dataset.X_num['test'] = dataset.X_num['test'].astype(np.float32)
        dataset.X_num['val'] = dataset.X_num['val'].astype(np.float32)
    
    # Debug categorical encoding
    print("\n=== Categorical Encoding Debug ===")
    print(f"args.catenc: {args.catenc}")
    print(f"dataset.X_cat is not None: {dataset.X_cat is not None}")
    if args.catenc and dataset.X_cat is not None:
        cardinalities = dataset.get_category_sizes('train')
        print(f"Cardinalities: {cardinalities}")
        enc = CatBoostEncoder(
            cols=list(range(len(cardinalities))), 
            return_df=False
        ).fit(dataset.X_cat['train'], dataset.y['train'])
        
        # Encode both test and validation sets
        encoded_cat_test = enc.transform(dataset.X_cat['test']).astype(np.float32)
        encoded_cat_val = enc.transform(dataset.X_cat['val']).astype(np.float32)
        
        print(f"Encoded categorical shape: {encoded_cat_test.shape}")
        print(f"Current numerical shape: {dataset.X_num['test'].shape}")
        
        # Concatenate for both sets
        dataset.X_num['test'] = np.concatenate([
            encoded_cat_test,
            dataset.X_num['test']
        ], axis=1)
        dataset.X_num['val'] = np.concatenate([
            encoded_cat_val,
            dataset.X_num['val']
        ], axis=1)
        print(f"Final concatenated shape: {dataset.X_num['test'].shape}")
    
    print(f"\nShape after categorical encoding: {dataset.X_num['test'].shape}")

    # Initialize model with exact feature count
    model = ExcelFormer(
        d_numerical=dataset.X_num['test'].shape[1],  # Use selected feature count
        d_out=2,  # Binary classification
        categories=None,
        token_bias=True,
        n_layers=3,
        n_heads=32,
        d_token=256,
        attention_dropout=0.3,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        prenormalization=True,
        kv_compression=None,
        kv_compression_sharing=None,
        init_scale=0.01,
    ).to(device)

    print(f"\nModel weight dtype: {next(model.parameters()).dtype}")

    # Convert to float32 consistently
    model = model.float()
    
    # Load model weights and set to eval mode
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n=== Feature Verification ===")
    # Get input dimension from first layer's weight shape
    input_dim = next(model.parameters()).shape[0]  # First dimension of first weight matrix
    print(f"Model input dimension: {input_dim}")
    print(f"Current feature count: {dataset.X_num['test'].shape[1]}")

    if input_dim != dataset.X_num['test'].shape[1]:
        raise ValueError(
            f"Feature count mismatch! Model expects {input_dim} features "
            f"but got {dataset.X_num['test'].shape[1]} features. "
            "Make sure you're using the same feature set used during training."
        )

    print("\nFeature names:")
    all_features = (dataset.cat_feature_names or []) + (dataset.num_feature_names or [])
    for i, feature in enumerate(all_features):
        print(f"{i+1}. {feature}")

    # Now prepare tensors and evaluate
    X_num_test = torch.tensor(dataset.X_num['test'], dtype=torch.float32).to(device)
    y_test = torch.tensor(dataset.y['test'], dtype=torch.float32).to(device)
    
    # Evaluate both validation and test sets
    print("\nEvaluating validation and test sets:")
    
    # Create datasets for both
    val_dataset = TensorDataset(
        torch.tensor(dataset.X_num['val'], dtype=torch.float32).to(device),
        torch.tensor(dataset.y['val'], dtype=torch.float32).to(device)
    )
    test_dataset = TensorDataset(X_num_test, y_test)
    
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Get predictions for both sets
    predictions_val = []
    predictions_test = []
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Processing validation set"):
            x_num, _ = batch
            pred = model(x_num, None)
            predictions_val.append(pred)
            
        for batch in tqdm(test_loader, desc="Processing test set"):
            x_num, _ = batch
            pred = model(x_num, None)
            predictions_test.append(pred)
    
    predictions_val = torch.cat(predictions_val).cpu().numpy()
    predictions_test = torch.cat(predictions_test).cpu().numpy()
    
    # Calculate probabilities and ROC curves for both sets
    y_val_proba = torch.softmax(torch.tensor(predictions_val), dim=1).numpy()[:, 1]
    y_test_proba = torch.softmax(torch.tensor(predictions_test), dim=1).numpy()[:, 1]
    
    fpr_val, tpr_val, _ = roc_curve(dataset.y['val'], y_val_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test.cpu().numpy(), y_test_proba)
    
    roc_auc_val = roc_auc_score(dataset.y['val'], y_val_proba)
    roc_auc_test = roc_auc_score(y_test.cpu().numpy(), y_test_proba)
    
    # Plot both ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'Validation ROC (AUC = {roc_auc_val:.3f})')
    plt.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Test ROC (AUC = {roc_auc_test:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ExcelFormer ROC Curves - Top 25 XGBoost Features')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curves_comparison.png')
    plt.show()
    
    # Calculate and print metrics for both sets
    metrics = dataset.calculate_metrics(
        {
            'val': predictions_val,
            'test': predictions_test
        },
        prediction_type='logits'
    )
    
    print("\nVALIDATION SET METRICS:")
    print(f"ROC AUC: {metrics['val']['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['val']['accuracy']:.4f}")
    
    print("\nTEST SET METRICS:")
    print(f"ROC AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['test']['accuracy']:.4f}")

    print("\nEvaluation complete!")
    sys.stdout.log.close()  # Close the log file

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    main()
