import numpy as np

# Load and print the top 25 features _20250312_180210.npy
top_features = np.load('top_25_xgboost_features_20250312_180210.npy', allow_pickle=True)
print("Top 25 XGBoost features:")
for i, feature in enumerate(top_features, 1):
    print(f"{i}. {feature}")
