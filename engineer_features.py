# pkg_features.py
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os
from tqdm import tqdm
import time
import fasttext
import fasttext.util
from sklearn.cluster import KMeans
from lib import Dataset, Transformations, build_dataset
from sklearn.feature_selection import mutual_info_classif

class PackageFeatureGenerator:
    def __init__(self, cache_dir='cache/pkg_features/'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.n_embedding_dims = 100  # FastText default
        self.n_clusters = 100
        self.random_state = 42
        
        # Fitted transformers
        self.fasttext_model = None
        self.kmeans = None
        
    def _generate_embeddings(self, pkg_names):
        """Generate embeddings using FastText"""
        print("\n=== Generating Package Name Embeddings ===")
        start_time = time.time()
        
        # Download FastText model if not present
        if not os.path.exists('cc.en.300.bin'):
            print("Downloading FastText model...")
            fasttext.util.download_model('en', if_exists='ignore')
        
        # Load model - changed from 100 to 300 dimensions
        print("Loading FastText model...")
        self.fasttext_model = fasttext.load_model('cc.en.300.bin')
        
        # Generate embeddings
        print(f"Generating embeddings for {len(pkg_names)} packages...")
        embeddings = np.zeros((len(pkg_names), 300))  # Changed dimension to match model
        for i, name in enumerate(tqdm(pkg_names)):
            embeddings[i] = self.fasttext_model.get_word_vector(name.replace('.', ' '))
        
        print(f"Time taken: {time.time() - start_time:.2f}s")
        return embeddings
    
    def generate_features(self, df, force_recompute=False):
        """Main feature generation method"""
        cache_file = self.cache_dir / 'pkg_features.pkl'
        
        if not force_recompute and cache_file.exists():
            print("Loading cached features...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("\n=== Starting Feature Generation ===")
        total_start = time.time()
        
        # 1. Generate embeddings
        embeddings = self._generate_embeddings(df['pkgname'])
        
        # 2. Generate clusters
        print("\n=== Clustering Package Names ===")
        start_time = time.time()
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            verbose=1
        )
        clusters = self.kmeans.fit_predict(embeddings)
        
        # Create feature DataFrame
        feature_df = pd.DataFrame(
            embeddings,
            columns=[f'pkg_embedding_{i}' for i in range(self.n_embedding_dims)]
        )
        
        # Add cluster features
        feature_df['pkg_cluster'] = clusters
        feature_df['position_in_cluster'] = df.groupby('pkg_cluster').cumcount()
        feature_df['cluster_size'] = df['pkg_cluster'].map(df['pkg_cluster'].value_counts())
        feature_df['relative_pos'] = feature_df['position_in_cluster'] / feature_df['cluster_size']
        
        print(f"\nTotal time: {time.time() - total_start:.2f}s")
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(feature_df, f)
        
        return feature_df

def process_package_name(pkg_name):
    """Better package name processing"""
    # Split on dots first
    parts = pkg_name.split('.')
    
    # Skip common prefixes that don't add meaning
    skip_prefixes = {'com', 'org', 'net', 'io', 'air'}
    meaningful_parts = [p for p in parts if p not in skip_prefixes]
    
    # Split CamelCase and compound words
    tokens = []
    for part in meaningful_parts:
        # Split on numbers
        part = ''.join([' '+c if c.isdigit() and i>0 else c 
                       for i,c in enumerate(part)])
        
        # Split CamelCase
        part = ''.join([' '+c if c.isupper() and i>0 else c 
                       for i,c in enumerate(part)])
        
        # Add individual tokens
        tokens.extend(part.lower().split())
    
    return tokens

def main():
    # Load original dataset and engineered features
    print("Loading data...")
    df = pd.read_csv('DATA/android_security/corrected_permacts.csv')
    y = df['status']  # Get labels directly
    
    # Generate embeddings with dot-split tokenization
    generator = PackageFeatureGenerator()
    print("\nProcessing package names...")
    embeddings_list = []
    
    # Initialize FastText model first
    if not os.path.exists('cc.en.300.bin'):
        print("Downloading FastText model...")
        fasttext.util.download_model('en', if_exists='ignore')
    
    print("Loading FastText model...")
    generator.fasttext_model = fasttext.load_model('cc.en.300.bin')
    
    for pkg_name in tqdm(df['pkgname']):
        # Get meaningful tokens
        tokens = process_package_name(pkg_name)
        
        # Get embedding for each token
        token_embeddings = np.array([
            generator.fasttext_model.get_word_vector(token)
            for token in tokens if token  # Skip empty tokens
        ])
        
        if len(token_embeddings) > 0:
            pkg_embedding = token_embeddings.mean(axis=0)
        else:
            pkg_embedding = np.zeros(300)  # Fallback for empty cases
            
        embeddings_list.append(pkg_embedding)
    
    embeddings = np.array(embeddings_list)
    
    # Calculate MI scores
    print("\nCalculating MI scores...")
    start_time = time.time()
    mi_scores = mutual_info_classif(embeddings, y)
    print(f"MI calculation took: {time.time() - start_time:.2f}s")
    
    # Compare with original numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(['status', 'Unnamed: 0'], axis=1)
    numeric_features = numeric_features.fillna(0)  # Handle NaN values
    orig_mi_scores = mutual_info_classif(numeric_features.values, y)
    
    # Calculate average MI for embeddings vs original features
    avg_embedding_mi = np.mean(mi_scores)
    
    # Print comparison
    print("\nMutual Information Comparison:")
    print(f"Package Name Embeddings (avg): {avg_embedding_mi:.4f}")
    print("\nOriginal Features:")
    for feature, mi in sorted(zip(numeric_features.columns, orig_mi_scores), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {mi:.4f}")

if __name__ == "__main__":
    main()