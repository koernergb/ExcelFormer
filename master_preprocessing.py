import pandas as pd
import numpy as np
import os
import json
import hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Dict, Union, Optional, Tuple

def create_standardized_data_and_indices(
    csv_path: str,
    output_dir: str = './standardized_data',
    sample_sizes: List[Union[int, str]] = [10000, 100000, 'full'],
    random_seed: int = 42
) -> Dict:
    """
    Master preprocessing function that creates standardized data and indices for both models.
    
    Args:
        csv_path: Path to the corrected_permacts.csv file
        output_dir: Directory to save outputs
        sample_sizes: List of sample sizes to create indices for
        random_seed: Random seed for reproducibility
    
    Returns:
        Dict with metadata about created files
    """
    
    # Feature definitions (identical to both training scripts)
    selected_features = [
        'ContentRating', 'Genre', 'CurrentVersion', 'AndroidVersion', 
        'DeveloperCategory', 'lowest_android_version', 'highest_android_version',
        'privacy_policy_link', 'developer_website', 'days_since_last_update',
        'isSpamming', 'max_downloads_log', 'LenWhatsNew', 'PHONE',
        'OneStarRatings', 'developer_address', 'FourStarRatings', 'intent',
        'ReviewsAverage', 'STORAGE', 'LastUpdated', 'TwoStarRatings',
        'LOCATION', 'FiveStarRatings', 'ThreeStarRatings'
    ]
    
    categorical_features = [
        'ContentRating', 'highest_android_version', 'CurrentVersion',
        'lowest_android_version', 'AndroidVersion', 'DeveloperCategory', 'Genre'
    ]
    
    numerical_features = [f for f in selected_features if f not in categorical_features]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== MASTER PREPROCESSING SCRIPT ===")
    print(f"Loading data from: {csv_path}")
    
    # Step 1: Load and clean data (standardized for both models)
    df = pd.read_csv(csv_path)
    print(f"Initial DataFrame shape: {df.shape}")
    
    # Drop NaNs
    df = df.dropna()
    print(f"Shape after dropping NaNs: {df.shape}")
    
    # Drop unnecessary columns (standardized)
    columns_to_drop = []
    if 'Unnamed: 0' in df.columns:
        columns_to_drop.append('Unnamed: 0')
    if 'pkgname' in df.columns:
        columns_to_drop.append('pkgname')
    
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
        print(f"Dropped columns: {columns_to_drop}")
        print(f"Shape after dropping unnecessary columns: {df.shape}")
    
    # Step 2: Feature selection and ordering (CRITICAL FOR CONSISTENCY)
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in DataFrame: {missing_features}")
    
    # Keep only selected features + target, in the exact order specified
    df_clean = df[selected_features + ['status']].copy()
    print(f"Shape after feature selection: {df_clean.shape}")
    
    # Step 3: Define canonical feature ordering (CRITICAL)
    # Both models will use this exact order after their respective encodings
    canonical_feature_order = categorical_features + numerical_features
    print(f"\nCanonical feature order (categorical + numerical):")
    for i, feat in enumerate(canonical_feature_order, 1):
        feat_type = "CAT" if feat in categorical_features else "NUM"
        print(f"  {i:2d}. {feat} ({feat_type})")
    
    # Reorder features in dataframe to match canonical order
    df_clean = df_clean[canonical_feature_order + ['status']]
    
    # Step 4: Save cleaned data
    cleaned_data_path = os.path.join(output_dir, 'cleaned_data.pkl')
    df_clean.to_pickle(cleaned_data_path)
    print(f"\nSaved cleaned data to: {cleaned_data_path}")
    
    # Step 5: Create train/val/test splits for full dataset (with stratification)
    X_full = df_clean.drop('status', axis=1)
    y_full = df_clean['status']
    
    # Full dataset split (70/15/15) with stratification
    X_train_full, X_temp, y_train_full, y_temp = train_test_split(
        X_full, y_full, test_size=0.3, random_state=random_seed, stratify=y_full
    )
    X_val_full, X_test_full, y_val_full, y_test_full = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_seed, stratify=y_temp
    )
    
    # Get indices for full dataset
    full_train_indices = X_train_full.index.values
    full_val_indices = X_val_full.index.values
    full_test_indices = X_test_full.index.values
    
    print(f"\nFull dataset split:")
    print(f"  Train: {len(full_train_indices)} samples")
    print(f"  Val:   {len(full_val_indices)} samples") 
    print(f"  Test:  {len(full_test_indices)} samples")
    
    # Step 6: Create sample indices for each specified sample size
    np.random.seed(random_seed)
    created_files = {}
    
    for size in sample_sizes:
        print(f"\nCreating indices for sample size: {size}")
        
        if size == 'full':
            # Use full dataset indices
            train_indices = full_train_indices
            val_indices = full_val_indices
            test_indices = full_test_indices
            size_str = 'full'
        else:
            # Create stratified sample from full dataset
            # Sample from entire cleaned dataset, then split
            total_sample_indices = np.random.choice(
                df_clean.index, size=size, replace=False
            )
            
            X_sample = df_clean.loc[total_sample_indices].drop('status', axis=1)
            y_sample = df_clean.loc[total_sample_indices]['status']
            
            # Split the sample (70/15/15) with stratification
            X_train_sample, X_temp_sample, y_train_sample, y_temp_sample = train_test_split(
                X_sample, y_sample, test_size=0.3, random_state=random_seed, stratify=y_sample
            )
            X_val_sample, X_test_sample, y_val_sample, y_test_sample = train_test_split(
                X_temp_sample, y_temp_sample, test_size=0.5, random_state=random_seed, stratify=y_temp_sample
            )
            
            train_indices = X_train_sample.index.values
            val_indices = X_val_sample.index.values
            test_indices = X_test_sample.index.values
            size_str = str(size)
        
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val:   {len(val_indices)} samples")
        print(f"  Test:  {len(test_indices)} samples")
        
        # Save indices
        train_file = os.path.join(output_dir, f'train_indices_{size_str}.npy')
        val_file = os.path.join(output_dir, f'val_indices_{size_str}.npy')
        test_file = os.path.join(output_dir, f'test_indices_{size_str}.npy')
        
        np.save(train_file, train_indices)
        np.save(val_file, val_indices)
        np.save(test_file, test_indices)
        
        created_files[size_str] = {
            'train_indices': train_file,
            'val_indices': val_file, 
            'test_indices': test_file,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices)
        }
        
        print(f"  Saved indices to: train_indices_{size_str}.npy, val_indices_{size_str}.npy, test_indices_{size_str}.npy")
    
    # Step 7: Create metadata file with FAST checksum
    # Use shape and sample values for checksum instead of full dataframe string
    checksum_data = f"{df_clean.shape}_{df_clean.iloc[0].to_string()}_{df_clean.iloc[-1].to_string()}_{df_clean.dtypes.to_string()}"
    fast_checksum = hashlib.md5(checksum_data.encode()).hexdigest()
    
    metadata = {
        'csv_path': csv_path,
        'cleaned_data_path': cleaned_data_path,
        'random_seed': random_seed,
        'total_samples': len(df_clean),
        'selected_features': selected_features,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'canonical_feature_order': canonical_feature_order,
        'sample_sizes': created_files,
        'preprocessing_steps': [
            '1. Load CSV and drop NaNs',
            '2. Drop unnecessary columns (Unnamed: 0, pkgname)',
            '3. Select and reorder features in canonical order',
            '4. Create stratified train/val/test splits',
            '5. Save cleaned data and indices'
        ],
        'data_checksum': fast_checksum
    }
    
    metadata_path = os.path.join(output_dir, 'preprocessing_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved metadata to: {metadata_path}")
    print(f"Data checksum: {metadata['data_checksum']}")
    
    return metadata


def load_standardized_data(
    sample_size: Union[int, str],
    data_dir: str = './standardized_data'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load preprocessed data and indices for a specific sample size.
    
    Args:
        sample_size: Sample size to load ('full', 10000, 100000, etc.)
        data_dir: Directory containing standardized data
    
    Returns:
        Tuple of (cleaned_dataframe, train_indices, val_indices, test_indices, metadata)
    """
    
    # Load metadata
    metadata_path = os.path.join(data_dir, 'preprocessing_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load cleaned data
    cleaned_data_path = metadata['cleaned_data_path']
    df_clean = pd.read_pickle(cleaned_data_path)
    
    # Load indices
    size_str = 'full' if sample_size == 'full' else str(sample_size)
    
    if size_str not in metadata['sample_sizes']:
        raise ValueError(f"Sample size {size_str} not found. Available: {list(metadata['sample_sizes'].keys())}")
    
    train_indices = np.load(metadata['sample_sizes'][size_str]['train_indices'])
    val_indices = np.load(metadata['sample_sizes'][size_str]['val_indices'])
    test_indices = np.load(metadata['sample_sizes'][size_str]['test_indices'])
    
    print(f"Loaded standardized data for sample size: {size_str}")
    print(f"  Data shape: {df_clean.shape}")
    print(f"  Train indices: {len(train_indices)}")
    print(f"  Val indices: {len(val_indices)}")
    print(f"  Test indices: {len(test_indices)}")
    print(f"  Feature order: {metadata['canonical_feature_order']}")
    
    return df_clean, train_indices, val_indices, test_indices, metadata


def verify_data_consistency(data_dir: str = './standardized_data') -> bool:
    """
    Verify that the preprocessed data is consistent and hasn't been corrupted.
    
    Args:
        data_dir: Directory containing standardized data
    
    Returns:
        True if data is consistent, False otherwise
    """
    
    try:
        # Load metadata
        metadata_path = os.path.join(data_dir, 'preprocessing_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load and verify cleaned data
        df_clean = pd.read_pickle(metadata['cleaned_data_path'])
        
        # Verify checksum using FAST method
        checksum_data = f"{df_clean.shape}_{df_clean.iloc[0].to_string()}_{df_clean.iloc[-1].to_string()}_{df_clean.dtypes.to_string()}"
        current_checksum = hashlib.md5(checksum_data.encode()).hexdigest()
        original_checksum = metadata['data_checksum']
        
        if current_checksum != original_checksum:
            print(f"❌ Data checksum mismatch!")
            print(f"  Original: {original_checksum}")
            print(f"  Current:  {current_checksum}")
            return False
        
        # Verify feature order
        expected_order = metadata['canonical_feature_order']
        actual_order = df_clean.columns[:-1].tolist()  # Exclude 'status'
        
        if actual_order != expected_order:
            print(f"❌ Feature order mismatch!")
            print(f"  Expected: {expected_order}")
            print(f"  Actual:   {actual_order}")
            return False
        
        # Verify all sample files exist
        for size_str, files in metadata['sample_sizes'].items():
            for file_type, file_path in files.items():
                if file_type.endswith('_indices') and not os.path.exists(file_path):
                    print(f"❌ Missing file: {file_path}")
                    return False
        
        print("✅ Data consistency verified!")
        print(f"  Checksum: {current_checksum}")
        print(f"  Features: {len(expected_order)} in correct order")
        print(f"  Sample sizes: {list(metadata['sample_sizes'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying data consistency: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    csv_path = "./DATA/android_security/corrected_permacts.csv"  # Update this path
    
    # Create standardized data and indices
    metadata = create_standardized_data_and_indices(
        csv_path=csv_path,
        output_dir='./standardized_data',
        sample_sizes=[10000, 100000, 'full'],
        random_seed=42
    )
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE!")
    print("="*50)
    
    # Verify consistency
    verify_data_consistency('./standardized_data')
    
    print("\nFiles created:")
    for size, files in metadata['sample_sizes'].items():
        print(f"  Sample size {size}:")
        print(f"    - train_indices_{size}.npy ({files['train_size']} samples)")
        print(f"    - val_indices_{size}.npy ({files['val_size']} samples)")
        print(f"    - test_indices_{size}.npy ({files['test_size']} samples)")
    
    print("\nNext steps:")
    print("1. Update XGBoost script to use load_standardized_data()")
    print("2. Update ExcelFormer script to use load_standardized_data()")
    print("3. Both models will now use identical samples and feature ordering!")