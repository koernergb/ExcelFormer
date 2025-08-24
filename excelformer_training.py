print("Entered training script...")
print("=== MODIFIED EXCELFORMER SCRIPT WITH STANDARDIZED PREPROCESSING ===")

import sys
import os
import math
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import CatBoostEncoder

from bin import ExcelFormer
from lib import Transformations, prepare_tensors, make_optimizer, DATA

# Import the master preprocessing functions
import sys
sys.path.append('.')
from master_preprocessing import load_standardized_data, verify_data_consistency

# Diagnostic information
print("Python executable:", sys.executable)
print("Python path:", sys.path)
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'Not set'))
print("CUDA_HOME:", os.environ.get('CUDA_HOME', 'Not set'))

# Torch and CUDA details
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

# Check for CUDA devices
import subprocess
try:
    nvidia_smi_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
    print("nvidia-smi output:\n", nvidia_smi_output)
except Exception as e:
    print("Error running nvidia-smi:", e)

print("Passed imports...")
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))

DATASETS = [
    'analcatdata_supreme', 'isolet', 'cpu_act', 'visualizing_soil', 'yprop_4_1', 'gesture', 'churn', 'sulfur', 'bank-marketing', 'Brazilian_houses'
    'eye', 'MagicTelescope', 'Ailerons', 'pol', 'polv2', 'credit', 'california', 'house_sales', 'house', 'diamonds', 'helena', 'jannis', 'higgs-small',
    'road-safety', 'medical_charges', 'SGEMM_GPU_kernel_performance', 'covtype', 'nyc-taxi-green-dec-2016', 'android_security'
]

def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='result/ExcelFormer/standardized')
    parser.add_argument("--dataset", type=str, default='android_security')
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.5, help='hyper-parameter of Beta Distribution in mixup, we choose 0.5 for all datasets in default config')
    parser.add_argument("--mix_type", type=str, default='none', choices=['niave_mix', 'feat_mix', 'hidden_mix', 'none'], help='mixup type, set to "niave_mix" for naive mixup, set to "none" if no mixup')
    parser.add_argument("--save", action='store_true', help='whether to save model')
    parser.add_argument("--catenc", action='store_true', help='whether to use catboost encoder for categorical features')
    parser.add_argument("--resume", type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument("--sample_size", type=str, choices=['10000', '100000', 'full'], default='full',
                        help="Subset size for training/validation/test (matches XGBoost splits)")
    args = parser.parse_args()

    # Convert sample_size to appropriate type
    if args.sample_size != 'full':
        args.sample_size = int(args.sample_size)

    args.output = f'{args.output}/mixup({args.mix_type})/{args.dataset}/{args.seed}'
    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
    if args.sample_size is not None:
        args.output += f"/{args.sample_size}"
        if not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
    
    # some basic model configuration
    cfg = {
        "model": {
            "prenormalization": True,
            'kv_compression': None,
            'kv_compression_sharing': None,
            'token_bias': True
        },
        "training": {
            "max_epoch": 500,
            "optimizer": "adamw",
        }
    }
    
    return args, cfg

def record_exp(args, final_score, best_score, **kwargs):
    # 'best': the best test score during running
    # 'final': the final test score acquired by validation set
    results = {'config': args, 'final': final_score, 'best': best_score, **kwargs}
    with open(f"{args['output']}/results.json", 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_standardized_dataset_from_dataframe(df_clean, train_indices, val_indices, test_indices, 
                                               categorical_features, numerical_features, 
                                               canonical_feature_order, args):
    """
    Create a dataset object from standardized dataframe and indices.
    This replaces the original Dataset.from_dir() method.
    """
    
    # Extract features and target using canonical order
    X = df_clean[canonical_feature_order]
    y = df_clean['status']
    
    # Create train/val/test sets using standardized indices
    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]
    X_val = X.loc[val_indices]
    y_val = y.loc[val_indices]
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    
    print(f"[STANDARDIZED] Train/Val/Test shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Verify feature order matches canonical order
    print(f"[STANDARDIZED] Feature order verification:")
    print(f"  Expected: {canonical_feature_order[:3]}...")
    print(f"  Actual:   {X_train.columns.tolist()[:3]}...")
    assert X_train.columns.tolist() == canonical_feature_order, "Feature order mismatch!"
    print("✅ Feature order verified!")
    
    # Check class balance
    print(f"[STANDARDIZED] Class balance in train set:")
    print(f"  Class 0 (benign): {(y_train == 0).sum()}")
    print(f"  Class 1 (malware): {(y_train == 1).sum()}")
    print(f"  Ratio: {(y_train == 1).mean():.3f}")
    
    # Split features into categorical and numerical (maintaining canonical order)
    X_num = {
        'train': X_train.select_dtypes(include=['int64', 'float64']).values.astype(np.float32),
        'val': X_val.select_dtypes(include=['int64', 'float64']).values.astype(np.float32),
        'test': X_test.select_dtypes(include=['int64', 'float64']).values.astype(np.float32)
    }
    X_cat = {
        'train': X_train.select_dtypes(include=['object']).values,
        'val': X_val.select_dtypes(include=['object']).values,
        'test': X_test.select_dtypes(include=['object']).values
    }
    
    y_dict = {
        'train': y_train.values,
        'val': y_val.values,
        'test': y_test.values
    }
    
    print(f"[STANDARDIZED] Feature split:")
    print(f"  Numerical features: {numerical_features}")
    print(f"  Categorical features: {categorical_features}")
    print("  X_num shapes:")
    for k, v in X_num.items():
        print(f"    {k}: {v.shape}")
    print("  X_cat shapes:")
    for k, v in X_cat.items():
        print(f"    {k}: {v.shape}")
    
    # Create a dataset-like object that matches the original interface
    class StandardizedDataset:
        def __init__(self):
            self.X_num = X_num
            self.X_cat = X_cat
            self.y = y_dict
            self.y_info = {}
            self.task_type = 'BINCLASS'  # TaskType.BINCLASS equivalent
            self.n_classes = 2
            self.num_feature_names = numerical_features
            self.cat_feature_names = categorical_features
            
        @property
        def is_binclass(self):
            return True
            
        @property
        def is_multiclass(self):
            return False
            
        @property
        def is_regression(self):
            return False
            
        @property
        def n_num_features(self):
            return 0 if self.X_num is None else self.X_num['train'].shape[1]
            
        @property
        def n_cat_features(self):
            return 0 if self.X_cat is None else self.X_cat['train'].shape[1]
            
        @property
        def n_features(self):
            return self.n_num_features + self.n_cat_features
            
        def get_category_sizes(self, part):
            if self.X_cat is None:
                return []
            X = self.X_cat[part]
            XT = X.T.tolist()
            return [len(set(x)) for x in XT]
            
        def calculate_metrics(self, predictions, prediction_type):
            # Import here to avoid circular dependencies
            from lib.metrics import calculate_metrics as calculate_metrics_
            from lib.util import TaskType
            
            metrics = {
                x: calculate_metrics_(
                    self.y[x], predictions[x], TaskType.BINCLASS, prediction_type, self.y_info
                )
                for x in predictions
            }
            # Add score metric
            for part_metrics in metrics.values():
                part_metrics['score'] = part_metrics['accuracy']  # For binary classification
            return metrics
    
    return StandardizedDataset()

"""args"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args, cfg = get_training_args()
seed_everything(args.seed)

print("\n" + "="*60)
print("VERIFYING STANDARDIZED DATA CONSISTENCY")
print("="*60)

# Verify data consistency first
if not verify_data_consistency('./standardized_data'):
    print("❌ Data consistency check failed! Run master preprocessing script first.")
    sys.exit(1)

print("\n" + "="*60)
print("LOADING STANDARDIZED DATA")
print("="*60)

# Load standardized data and indices
df_clean, train_indices, val_indices, test_indices, metadata = load_standardized_data(
    sample_size=args.sample_size,
    data_dir='./standardized_data'
)

# Get feature definitions from standardized preprocessing
selected_features = metadata['selected_features']
categorical_features = metadata['categorical_features']
numerical_features = metadata['numerical_features']
canonical_feature_order = metadata['canonical_feature_order']

print(f"Using standardized features:")
print(f"  Total features: {len(selected_features)}")
print(f"  Categorical: {len(categorical_features)}")
print(f"  Numerical: {len(numerical_features)}")
print(f"  Canonical order: {canonical_feature_order}")
print(f"  Sample size: {args.sample_size}")
print(f"  Data checksum: {metadata['data_checksum']}")

print("\n" + "="*60)
print("CREATING DATASET FROM STANDARDIZED DATA")
print("="*60)

# Create dataset using standardized data (replaces build_dataset call)
dataset = create_standardized_dataset_from_dataframe(
    df_clean, train_indices, val_indices, test_indices,
    categorical_features, numerical_features, canonical_feature_order, args
)

# Ensure float32 data type
if dataset.X_num['train'].dtype == np.float64:
    dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}

print("\n" + "="*60)
print("APPLYING CATEGORICAL ENCODING")
print("="*60)

# Apply categorical encoding (CatBoost for ExcelFormer)
if args.catenc and dataset.X_cat is not None:
    print("Using CatBoost encoding for categorical features...")
    cardinalities = dataset.get_category_sizes('train')
    enc = CatBoostEncoder(
        cols=list(range(len(cardinalities))), 
        return_df=False
    ).fit(dataset.X_cat['train'], dataset.y['train'])
    
    # CRITICAL: Maintain canonical order - categorical features first!
    X_num_processed = {}
    for k in ['train', 'val', 'test']:
        encoded_cat = enc.transform(dataset.X_cat[k]).astype(np.float32)
        # Concatenate: encoded_categorical + numerical (matches canonical order)
        X_num_processed[k] = np.concatenate([encoded_cat, dataset.X_num[k]], axis=1)
    print(f"Shape after CatBoost encoding: {X_num_processed['train'].shape}")
    print(f"  Categorical encoded: {encoded_cat.shape[1]} features")
    print(f"  Numerical: {dataset.X_num['train'].shape[1]} features")
else:
    print("Using raw categorical features (no CatBoost encoding)")
    X_num_processed = dataset.X_num

print("\n" + "="*60)
print("PREPARING TENSORS")
print("="*60)

# Prepare tensors for training
X_num, X_cat, ys = prepare_tensors(
    # Create a temporary dataset-like object for prepare_tensors
    type('DatasetObj', (), {
        'X_num': X_num_processed,
        'X_cat': None if args.catenc else dataset.X_cat,
        'y': dataset.y,
        'n_classes': dataset.n_classes,
        'is_binclass': dataset.is_binclass,
        'is_multiclass': dataset.is_multiclass,
        'is_regression': dataset.is_regression,
        'calculate_metrics': dataset.calculate_metrics,
        'n_features': X_num_processed['train'].shape[1],
        'num_feature_names': dataset.num_feature_names,
        'cat_feature_names': dataset.cat_feature_names,
        'get_category_sizes': dataset.get_category_sizes,
    })(),
    device=device
)

print(f"Final tensor shapes:")
print(f"  X_num['train']: {X_num['train'].shape}")
print(f"  X_num['val']: {X_num['val'].shape}")
print(f"  X_num['test']: {X_num['test'].shape}")
print(f"  ys['train']: {ys['train'].shape}")

if args.catenc:
    X_cat = None

# After processing, get final feature count
n_num_features = X_num['train'].shape[1]
print(f"Total features for model: {n_num_features}")

# Define output dimension
if hasattr(dataset, 'is_binclass') and dataset.is_binclass:
    d_out = 2
elif hasattr(dataset, 'is_multiclass') and dataset.is_multiclass:
    d_out = dataset.n_classes
else:
    d_out = 1

print(f"Model output dimension: {d_out}")

print("\n" + "="*60)
print("SETTING UP TRAINING")
print("="*60)

# Set batch size based on dataset features
batch_size_dict = {
    'churn': 128, 'eye': 128, 'gesture': 128, 'california': 256, 'house': 256,
    'higgs-small': 512, 'helena': 512, 'jannis': 512, 'covtype': 1024
}

if args.dataset in batch_size_dict:
    batch_size = batch_size_dict[args.dataset]
    val_batch_size = 512
else:
    # batch size settings for datasets
    if n_num_features <= 32:
        batch_size = 512
        val_batch_size = 8192
    elif n_num_features <= 100:
        batch_size = 128
        val_batch_size = 512
    elif n_num_features <= 1000:
        batch_size = 32
        val_batch_size = 64
    else:
        batch_size = 16
        val_batch_size = 16

print(f"Batch sizes: train={batch_size}, val={val_batch_size}")

# Update training config
cfg['training'].update({
    "batch_size": batch_size, 
    "eval_batch_size": val_batch_size, 
    "patience": args.early_stop
})

# Create data loaders
data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

print(f"Data loaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

print("\n" + "="*60)
print("BUILDING MODEL")
print("="*60)

# Prepare model parameters
cardinalities = dataset.get_category_sizes('train')
n_categories = len(cardinalities)
if args.catenc:
    n_categories = 0  # all categorical features are converted to numerical ones
cardinalities = None if n_categories == 0 else cardinalities

# Model configuration
kwargs = {
    'd_numerical': n_num_features,
    'd_out': d_out,
    'categories': cardinalities,
    **cfg['model']
}

default_model_configs = {
    'ffn_dropout': 0., 'attention_dropout': 0.3, 'residual_dropout': 0.0,
    'n_layers': 3, 'n_heads': 32, 'd_token': 256,
    'init_scale': 0.01,
}

default_training_configs = {
    'lr': 1e-3,
    'weight_decay': 1e-5,
}

kwargs.update(default_model_configs)
cfg['training'].update(default_training_configs)

print(f"Model configuration:")
print(f"  d_numerical: {kwargs['d_numerical']}")
print(f"  d_out: {kwargs['d_out']}")
print(f"  categories: {kwargs['categories']}")
print(f"  Using CatBoost: {args.catenc}")

# Build model
model = ExcelFormer(**kwargs).to(device)

# Optimizer
def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
optimizer = make_optimizer(
    cfg['training']['optimizer'],
    (
        [
            {'params': parameters_with_wd},
            {'params': parameters_without_wd, 'weight_decay': 0.0},
        ]
    ),
    cfg['training']['lr'],
    cfg['training']['weight_decay'],
)

# Parallelization
if torch.cuda.device_count() > 1:
    print('Using nn.DataParallel')
    model = nn.DataParallel(model)

# Loss function
loss_fn = (
    F.binary_cross_entropy_with_logits
    if dataset.is_binclass
    else F.cross_entropy
    if dataset.is_multiclass
    else F.mse_loss
)

print(f"Loss function: {loss_fn}")

print("\n" + "="*60)
print("TRAINING UTILITIES")
print("="*60)

def apply_model(x_num, x_cat=None, mixup=False):
    if mixup:
        return model(x_num, x_cat, mixup=True, beta=args.beta, mtype=args.mix_type)
    return model(x_num, x_cat)

@torch.inference_mode()
def evaluate(parts):
    model.eval()
    predictions = {}
    for part in parts:
        assert part in ['train', 'val', 'test']
        infer_time = 0.
        predictions[part] = []
        for batch in dataloaders[part]:
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            start = time.time()
            predictions[part].append(apply_model(x_num, x_cat))
            infer_time += time.time() - start
        predictions[part] = torch.cat(predictions[part]).cpu().numpy()
        if part == 'test':
            print(f'Test inference time: {infer_time:.2f}s')
    prediction_type = None if dataset.is_regression else 'logits'
    return dataset.calculate_metrics(predictions, prediction_type)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

# Training setup
metric = 'roc_auc' if dataset.is_binclass else 'score'
init_score = evaluate(['test'])['test'][metric]
print(f'Test score before training: {init_score:.4f}')

losses, val_metric, test_metric = [], [], []
n_epochs = 500

# Warmup and lr scheduler
warm_up = 10
scheduler = CosineAnnealingLR(
    optimizer=optimizer, 
    T_max=n_epochs - warm_up,
    eta_min=0
)
max_lr = cfg['training']['lr']
report_frequency = len(train_loader)
loss_holder = AverageMeter()

# Initialize variables
running_time = 0.0
start_epoch = 1
best_score = -np.inf
final_test_score = -np.inf
best_test_score = -np.inf
no_improvement = 0
losses = []
val_metric = []
test_metric = []

# Load checkpoint if resuming
if args.resume is not None and os.path.exists(args.resume):
    print(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume)
    
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict'))
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', -np.inf)
        best_test_score = checkpoint.get('best_test_score', -np.inf)
        final_test_score = checkpoint.get('final_test_score', -np.inf)
        no_improvement = checkpoint.get('no_improvement', 0)
        losses = checkpoint.get('losses', losses)
        val_metric = checkpoint.get('val_metric', val_metric)
        test_metric = checkpoint.get('test_metric', test_metric)
        running_time = checkpoint.get('running_time', 0.0)
        print(f"Resuming from epoch {start_epoch}")

print(f"Starting training from epoch {start_epoch}")
for epoch in range(start_epoch, n_epochs + 1):
    model.train()
    epoch_start = time.time()
    
    # Warm up learning rate
    if warm_up > 0 and epoch <= warm_up:
        lr = max_lr * epoch / warm_up
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()
        
    for iteration, batch in enumerate(train_loader):
        x_num, x_cat, y = (
            (batch[0], None, batch[1])
            if len(batch) == 2
            else batch
        )

        start = time.time()
        optimizer.zero_grad()
        
        if args.mix_type == 'none':  # no mixup
            preds = apply_model(x_num, x_cat)
            if dataset.is_binclass:
                loss = loss_fn(preds[:, 1], y.float())
            else:
                loss = loss_fn(preds, y)
        else:
            # Mixup training (not commonly used, but keeping for compatibility)
            preds, feat_masks, shuffled_ids = apply_model(x_num, x_cat, mixup=True)
            # Note: mixup implementation would need modification for standardized preprocessing
            # For now, defaulting to no mixup
            if dataset.is_binclass:
                loss = loss_fn(preds[:, 1], y.float())
            else:
                loss = loss_fn(preds, y)
                
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_time += time.time() - start
        loss_holder.update(loss.item(), len(ys))
        
        if iteration % report_frequency == 0:
            print(f'Epoch {epoch:03d} | Batch {iteration} | Loss: {loss_holder.val:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # Print epoch summary
    epoch_time = time.time() - epoch_start
    print(f'\nEpoch {epoch:03d} Summary:')
    print(f'Average Loss: {loss_holder.avg:.4f}')
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    print(f'Time: {epoch_time:.2f}s')
    
    # Evaluate and print metrics
    scores = evaluate(['val', 'test'])
    val_score, test_score = scores['val'][metric], scores['test'][metric]
    print(f'Validation score: {val_score:.4f}')
    print(f'Test score: {test_score:.4f}')
    print('-' * 50)

    losses.append(loss_holder.avg)
    loss_holder.reset()
    val_metric.append(val_score)
    test_metric.append(test_score)
    
    if val_score > (best_score + args.min_delta):
        best_score = val_score
        final_test_score = test_score
        print(f' <<< BEST VALIDATION EPOCH: {val_score:.4f}')
        no_improvement = 0
        
        if args.save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
                'best_test_score': best_test_score,
                'final_test_score': final_test_score,
                'no_improvement': no_improvement,
                'losses': losses,
                'val_metric': val_metric,
                'test_metric': test_metric,
                'running_time': running_time,
                'n_features': n_num_features,
                'preprocessing_metadata': metadata,
                'sample_size': args.sample_size,
                'catenc': args.catenc
            }
            model_path = f"{args.output}/pytorch_model_standardized.pt"
            torch.save(checkpoint, model_path)
            print(f"Saved best model to: {model_path}")
    else:
        no_improvement += 1
        
    if test_score > best_test_score:
        best_test_score = test_score

    if no_improvement == args.early_stop:
        print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop} epochs)")
        break

print("\n" + "="*60)
print("TRAINING COMPLETE - RECORDING RESULTS")
print("="*60)

# Record experiment results
record_exp(
    vars(args), final_test_score, best_test_score,
    losses=str(losses), val_score=str(val_metric), test_score=str(test_metric),
    cfg=cfg, time=running_time,
    preprocessing='standardized',
    categorical_encoding='CatBoost' if args.catenc else 'Raw',
    feature_order=canonical_feature_order,
    data_checksum=metadata['data_checksum'],
    sample_size=args.sample_size
)

print(f"Final Results:")
print(f"  Best validation score: {best_score:.4f}")
print(f"  Final test score: {final_test_score:.4f}")
print(f"  Best test score: {best_test_score:.4f}")
print(f"  Total training time: {running_time:.2f}s")

print("\nFeatures used for ExcelFormer training:")
print(f"  Categorical: {dataset.cat_feature_names}")
print(f"  Numerical: {dataset.num_feature_names}")
print(f"  Canonical order: {canonical_feature_order}")

print(f"\nTraining dataset size: {args.sample_size}")
print(f"Preprocessing: Standardized")
print(f"Categorical encoding: {'CatBoost' if args.catenc else 'Raw'}")
print(f"Data checksum: {metadata['data_checksum']}")

print("\n[DEBUG][TRAIN] Final feature information:")
if args.catenc and dataset.X_cat is not None:
    print(f"Cat features (CatBoost encoded): {len(dataset.cat_feature_names)} -> {encoded_cat.shape[1]} features")
    print(f"Num features: {len(dataset.num_feature_names)} features")
    print(f"Total model input: {n_num_features} features")
    print(f"Feature order: [CatBoost_encoded] + [Numerical]")
else:
    print(f"Cat features (raw): {len(dataset.cat_feature_names)} features")
    print(f"Num features: {len(dataset.num_feature_names)} features")
    print(f"Total model input: {n_num_features} features")
    print(f"Feature order: [Categorical] + [Numerical]")

print(f"\n[DEBUG][TRAIN] Final tensor shapes:")
print(f"  X_num['train']: {X_num['train'].shape}")
print(f"  X_num['val']: {X_num['val'].shape}")
print(f"  X_num['test']: {X_num['test'].shape}")

# Save detailed results with preprocessing info
detailed_results = {
    'model': 'ExcelFormer',
    'preprocessing': 'standardized',
    'sample_size': args.sample_size,
    'categorical_encoding': 'CatBoost' if args.catenc else 'Raw',
    'feature_order': canonical_feature_order,
    'data_checksum': metadata['data_checksum'],
    'train_size': len(train_indices),
    'val_size': len(val_indices),
    'test_size': len(test_indices),
    'final_val_score': best_score,
    'final_test_score': final_test_score,
    'best_test_score': best_test_score,
    'total_training_time': running_time,
    'total_epochs': epoch,
    'early_stopped': no_improvement >= args.early_stop,
    'n_features': n_num_features,
    'categorical_features': dataset.cat_feature_names,
    'numerical_features': dataset.num_feature_names,
    'model_config': kwargs,
    'training_config': cfg['training'],
    'args': vars(args)
}

timestamp = time.strftime('%Y%m%d_%H%M%S')
results_filename = f'excelformer_results_standardized_{args.sample_size}_{timestamp}.json'
with open(results_filename, 'w') as f:
    json.dump(detailed_results, f, indent=2)
print(f"\nSaved detailed results to: {results_filename}")

print("\n" + "="*60)
print("EXCELFORMER TRAINING WITH STANDARDIZED PREPROCESSING COMPLETE!")
print("="*60)