import sys
import os
from pathlib import Path
import numpy as np
from category_encoders import CatBoostEncoder

# Add ExcelFormer lib to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.data import Dataset, build_dataset, Transformations
from sklearn.feature_selection import mutual_info_classif

print("=== Loading and Processing Data ===")

# Use minimal transformations - we'll handle categorical encoding manually
transformations = Transformations(
    seed=42,
    normalization='quantile'
)

# Load dataset without categorical encoding
dataset = build_dataset('DATA/android_security', transformations, True)

print("\nDebug: Initial Dataset info:")
print(f"X_num shape: {dataset.X_num['train'].shape if dataset.X_num else 'None'}")
print(f"X_cat shape: {dataset.X_cat['train'].shape if dataset.X_cat else 'None'}")

# Apply CatBoost encoding exactly as in training script
if dataset.X_cat is not None:
    print("\nApplying CatBoost encoding...")
    cardinalities = dataset.get_category_sizes('train')
    enc = CatBoostEncoder(
        cols=list(range(len(cardinalities))),
        return_df=False
    ).fit(dataset.X_cat['train'], dataset.y['train'])
    
    # Transform categorical features and concatenate with numerical
    X_cat_encoded = enc.transform(dataset.X_cat['train']).astype(np.float32)
    if dataset.X_num is not None:
        dataset.X_num['train'] = np.concatenate([X_cat_encoded, dataset.X_num['train']], axis=1)
    else:
        dataset.X_num = {'train': X_cat_encoded}
    
    print(f"Shape after encoding: {dataset.X_num['train'].shape}")

# Create combined feature names in correct order (categorical first)
all_features = (dataset.cat_feature_names or []) + (dataset.num_feature_names or [])

print("\nDebug: Feature names:")
print(f"Categorical features: {dataset.cat_feature_names}")
print(f"Numerical features: {dataset.num_feature_names}")

print("\n=== Calculating Mutual Information Scores ===")
mi_scores = mutual_info_classif(dataset.X_num['train'], dataset.y['train'], random_state=42)

# Create feature-MI score pairs and sort
feature_mi_pairs = list(zip(all_features, mi_scores))
feature_mi_pairs.sort(key=lambda x: x[1], reverse=True)

# Select features with MI > 0.01
MI_THRESHOLD = 0.01
significant_features = [f for f, mi in feature_mi_pairs if mi >= MI_THRESHOLD]

print("\n=== MI Selected Features ===")
print("Features selected by MI scores (sorted by MI):")
for feature in significant_features:
    mi = next(mi for f, mi in feature_mi_pairs if f == feature)
    print(f"{feature}: {mi:.4f}")

# Load XGBoost selected features
print("\n=== Loading XGBoost Features ===")
try:
    xgb_features = np.load('top_25_xgboost_features_20250312_180210.npy', allow_pickle=True)
    print("Debug: XGBoost features type:", type(xgb_features))
    print("Debug: XGBoost features shape:", xgb_features.shape)
    print("Debug: XGBoost features content:", xgb_features)
    
    # Handle different possible formats
    if isinstance(xgb_features, np.ndarray):
        if xgb_features.dtype == np.dtype('O'):  # Object array
            if len(xgb_features) == 1:
                xgb_feature_names = xgb_features[0]
            else:
                xgb_feature_names = list(xgb_features)
        else:
            xgb_feature_names = list(xgb_features)
    else:
        xgb_feature_names = xgb_features
except Exception as e:
    print(f"Error loading XGBoost features: {str(e)}")
    print(f"XGBoost features content for debugging: {xgb_features}")
    raise

print("\nXGBoost selected features:")
for feature in xgb_feature_names:
    print(feature)

# Compare feature sets
mi_feature_set = set(significant_features)
xgb_feature_set = set(xgb_feature_names)

print("\n=== Feature Comparison ===")
print(f"Number of MI features: {len(mi_feature_set)}")
print(f"Number of XGBoost features: {len(xgb_feature_set)}")
print(f"Features are identical (ignoring order): {mi_feature_set == xgb_feature_set}")

if mi_feature_set != xgb_feature_set:
    print("\nFeatures in MI but not in XGBoost:")
    for f in mi_feature_set - xgb_feature_set:
        print(f"  {f}")
    
    print("\nFeatures in XGBoost but not in MI:")
    for f in xgb_feature_set - mi_feature_set:
        print(f"  {f}")

print("\nFeature order comparison:")
print("\nMI order:")
for i, f in enumerate(significant_features, 1):
    print(f"{i}. {f}")

print("\nXGBoost order:")
for i, f in enumerate(xgb_feature_names, 1):
    print(f"{i}. {f}") 