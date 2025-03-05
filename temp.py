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
    
    # Print the shapes of key parameters to debug
    print("Model state keys and shapes:")
    for key, value in model_state.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
            
    # First load and prepare dataset
    dataset = build_dataset(
        Path('DATA/android_security'),
        Transformations(normalization='standard'),
        cache=False
    )
    
    print(f"Dataset properties:")
    print(f"n_num_features: {dataset.n_num_features}")
    print(f"n_cat_features: {dataset.n_cat_features}")
    print(f"output_dim: {dataset.nn_output_dim}")
    
    # Initialize model with dimensions matching the saved state
    model = ExcelFormer(
        d_numerical=27,  # Match the saved model dimension
        categories=[],   # No categorical features
        d_out=2,        # Binary classification
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
    
    # Load state dict
    model.load_state_dict(model_state)
    model.eval()
    
    # Rest of the code...

if __name__ == '__main__':
    main()