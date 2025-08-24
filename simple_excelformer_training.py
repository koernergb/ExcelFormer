print("=== SIMPLIFIED EXCELFORMER TRAINING ===")

import sys
import os
import math
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score, accuracy_score
from category_encoders import CatBoostEncoder

from bin import ExcelFormer
from lib import make_optimizer

# Diagnostic information
print("Python executable:", sys.executable)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Feature definitions
SELECTED_FEATURES = [
    'ContentRating', 'Genre', 'CurrentVersion', 'AndroidVersion', 
    'DeveloperCategory', 'lowest_android_version', 'highest_android_version',
    'privacy_policy_link', 'developer_website', 'days_since_last_update',
    'isSpamming', 'max_downloads_log', 'LenWhatsNew', 'PHONE',
    'OneStarRatings', 'developer_address', 'FourStarRatings', 'intent',
    'ReviewsAverage', 'STORAGE', 'LastUpdated', 'TwoStarRatings',
    'LOCATION', 'FiveStarRatings', 'ThreeStarRatings'
]

CATEGORICAL_FEATURES = [
    'ContentRating', 'highest_android_version', 'CurrentVersion',
    'lowest_android_version', 'AndroidVersion', 'DeveloperCategory', 'Genre'
]

NUMERICAL_FEATURES = [f for f in SELECTED_FEATURES if f not in CATEGORICAL_FEATURES]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='result/ExcelFormer/simple')
    parser.add_argument("--dataset", type=str, default='android_security')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--save", action='store_true', help='whether to save model')
    parser.add_argument("--catenc", action='store_true', default=True, help='use catboost encoder')
    parser.add_argument("--sample_size", type=str, choices=['10000', '100000', 'full'], default='full')
    return parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_simple_data(sample_size='full', data_dir='./standardized_data'):
    """Load data without complex dependencies"""
    print(f"Loading data for sample size: {sample_size}")
    
    # Load metadata
    metadata_path = os.path.join(data_dir, 'preprocessing_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load cleaned data
    df_clean = pd.read_pickle(metadata['cleaned_data_path'])
    
    # Load indices
    size_str = 'full' if sample_size == 'full' else str(sample_size)
    sample_info = metadata['sample_sizes'][size_str]
    
    train_indices = np.load(sample_info['train_indices'])
    val_indices = np.load(sample_info['val_indices'])
    test_indices = np.load(sample_info['test_indices'])
    
    print(f"Loaded data - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return df_clean, train_indices, val_indices, test_indices, metadata

def create_dataset_splits(df_clean, train_indices, val_indices, test_indices):
    """Create train/val/test splits from dataframe"""
    # Get features in canonical order
    canonical_order = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = df_clean[canonical_order]
    y = df_clean['status']
    
    # Create splits
    splits = {}
    for name, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
        X_split = X.loc[indices]
        y_split = y.loc[indices]
        
        # Split into categorical and numerical
        X_cat = X_split[CATEGORICAL_FEATURES].values if CATEGORICAL_FEATURES else None
        X_num = X_split[NUMERICAL_FEATURES].values.astype(np.float32) if NUMERICAL_FEATURES else None
        
        splits[name] = {
            'X_cat': X_cat,
            'X_num': X_num,
            'y': y_split.values
        }
    
    print(f"Created dataset splits:")
    for name, data in splits.items():
        cat_shape = data['X_cat'].shape if data['X_cat'] is not None else None
        num_shape = data['X_num'].shape if data['X_num'] is not None else None
        print(f"  {name}: X_cat={cat_shape}, X_num={num_shape}, y={data['y'].shape}")
    
    return splits

def apply_catboost_encoding(splits, use_catboost=True):
    """Apply CatBoost encoding to categorical features"""
    if not use_catboost or splits['train']['X_cat'] is None:
        print("Skipping CatBoost encoding")
        return splits
    
    print("Applying CatBoost encoding...")
    
    # Fit encoder on training data
    n_cat_features = splits['train']['X_cat'].shape[1]
    enc = CatBoostEncoder(
        cols=list(range(n_cat_features)), 
        return_df=False
    ).fit(splits['train']['X_cat'], splits['train']['y'])
    
    # Apply encoding and concatenate with numerical features
    new_splits = {}
    for name, data in splits.items():
        encoded_cat = enc.transform(data['X_cat']).astype(np.float32)
        
        if data['X_num'] is not None:
            # Concatenate: categorical_encoded + numerical
            X_combined = np.concatenate([encoded_cat, data['X_num']], axis=1)
        else:
            X_combined = encoded_cat
        
        new_splits[name] = {
            'X_num': X_combined,
            'X_cat': None,
            'y': data['y']
        }
        
        print(f"  {name}: encoded shape = {X_combined.shape}")
    
    return new_splits

def calculate_metrics(y_true, y_pred_logits):
    """Calculate metrics for binary classification"""
    # Convert logits to probabilities
    y_pred_proba = torch.softmax(torch.tensor(y_pred_logits), dim=1).numpy()[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, (y_pred_proba > 0.5).astype(int))
    
    return {'roc_auc': auc, 'accuracy': acc}

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    args = get_args()
    seed_everything(args.seed)
    
    # Create output directory
    output_dir = f'{args.output}/{args.dataset}/{args.seed}/{args.sample_size}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load data
    df_clean, train_indices, val_indices, test_indices, metadata = load_simple_data(args.sample_size)
    
    # Create dataset splits
    splits = create_dataset_splits(df_clean, train_indices, val_indices, test_indices)
    
    # Apply CatBoost encoding
    splits = apply_catboost_encoding(splits, args.catenc)
    
    # Convert to tensors
    data_tensors = {}
    for name, data in splits.items():
        data_tensors[name] = {
            'X': torch.tensor(data['X_num'], dtype=torch.float32).to(device),
            'y': torch.tensor(data['y'], dtype=torch.long).to(device)
        }
    
    n_features = data_tensors['train']['X'].shape[1]
    print(f"Total features for model: {n_features}")
    
    # Create data loaders
    batch_size = 128
    dataloaders = {}
    for name, tensors in data_tensors.items():
        dataset = TensorDataset(tensors['X'], tensors['y'])
        shuffle = (name == 'train')
        dataloaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    print(f"Created data loaders with batch size: {batch_size}")
    
    # Build model
    model = ExcelFormer(
        d_numerical=n_features,
        d_out=2,  # Binary classification
        categories=None,  # No categorical features after encoding
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
    
    print(f"Created ExcelFormer model with {n_features} input features")
    
    # Optimizer and scheduler
    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    
    optimizer = make_optimizer(
        'adamw',
        [
            {'params': parameters_with_wd},
            {'params': parameters_without_wd, 'weight_decay': 0.0},
        ],
        lr=1e-3,
        weight_decay=1e-5,
    )
    
    n_epochs = 500
    warm_up = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs - warm_up, eta_min=0)
    
    # Training setup
    loss_fn = F.cross_entropy
    loss_meter = AverageMeter()
    
    best_val_auc = -np.inf
    best_test_auc = -np.inf
    final_test_auc = -np.inf
    no_improvement = 0
    
    train_losses = []
    val_aucs = []
    test_aucs = []
    
    print(f"Starting training for {n_epochs} epochs...")
    
    for epoch in range(1, n_epochs + 1):
        # Warmup learning rate
        if epoch <= warm_up:
            lr = 1e-3 * epoch / warm_up
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
        
        # Training
        model.train()
        loss_meter.reset()
        
        for batch_idx, (X, y) in enumerate(dataloaders['train']):
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(X, None)  # No categorical features
            loss = loss_fn(logits, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_meter.update(loss.item(), len(y))
        
        train_losses.append(loss_meter.avg)
        
        # Evaluation
        model.eval()
        predictions = {}
        
        with torch.no_grad():
            for name, loader in dataloaders.items():
                if name == 'train':
                    continue  # Skip train evaluation for speed
                    
                all_logits = []
                all_labels = []
                
                for X, y in loader:
                    logits = model(X, None)
                    all_logits.append(logits.cpu().numpy())
                    all_labels.append(y.cpu().numpy())
                
                predictions[name] = {
                    'logits': np.vstack(all_logits),
                    'labels': np.concatenate(all_labels)
                }
        
        # Calculate metrics
        val_metrics = calculate_metrics(predictions['val']['labels'], predictions['val']['logits'])
        test_metrics = calculate_metrics(predictions['test']['labels'], predictions['test']['logits'])
        
        val_auc = val_metrics['roc_auc']
        test_auc = test_metrics['roc_auc']
        
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)
        
        # Print progress
        if epoch % 10 == 0 or epoch <= 10:
            print(f'Epoch {epoch:3d} | Loss: {loss_meter.avg:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Check for improvement
        if val_auc > (best_val_auc + args.min_delta):
            best_val_auc = val_auc
            final_test_auc = test_auc
            no_improvement = 0
            
            if args.save:
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_auc': best_val_auc,
                    'final_test_auc': final_test_auc,
                    'n_features': n_features,
                    'sample_size': args.sample_size,
                    'catenc': args.catenc,
                    'metadata': metadata
                }
                torch.save(checkpoint, f'{output_dir}/best_model.pt')
                print(f' <<< BEST VAL AUC: {val_auc:.4f} - Saved model')
        else:
            no_improvement += 1
        
        if test_auc > best_test_auc:
            best_test_auc = test_auc
        
        # Early stopping
        if no_improvement >= args.early_stop:
            print(f'Early stopping at epoch {epoch} (no improvement for {args.early_stop} epochs)')
            break
    
    # Save final results
    results = {
        'model': 'ExcelFormer',
        'sample_size': args.sample_size,
        'catenc': args.catenc,
        'n_features': n_features,
        'total_epochs': epoch,
        'best_val_auc': best_val_auc,
        'final_test_auc': final_test_auc,
        'best_test_auc': best_test_auc,
        'train_losses': train_losses,
        'val_aucs': val_aucs,
        'test_aucs': test_aucs,
        'args': vars(args)
    }
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Sample size: {args.sample_size}")
    print(f"Features: {n_features}")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Final test AUC: {final_test_auc:.4f}")
    print(f"Best test AUC: {best_test_auc:.4f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 