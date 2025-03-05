import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import scipy.stats as stats

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('./DATA/android_security/corrected_permacts.csv')  # Replace with your actual file path

# 1. Direct sorting check
print("\n=== Testing if index represents sorting by package name ===")
# Create a dataframe with just index and package name
test_df = df[['Unnamed: 0', 'pkgname']].copy()

# Sort by package name and see if the index follows the same pattern
sorted_by_name = test_df.sort_values('pkgname')

# Calculate correlation between sorted index and original index
correlation = np.corrcoef(
    sorted_by_name['Unnamed: 0'].values, 
    sorted_by_name.index.values
)[0, 1]
print(f"Correlation between package name order and index: {correlation:.4f}")

# Visual check - plot the first 1000 entries
plt.figure(figsize=(10, 6))
plt.scatter(range(1000), sorted_by_name['Unnamed: 0'].iloc[:1000], alpha=0.5)
plt.title("Index values after sorting by package name")
plt.xlabel("Position after sorting by package name")
plt.ylabel("Original index value")
plt.grid(True)
plt.savefig("index_vs_pkgname_sort.png")
plt.close()

# 2. Lexicographical check
print("\n=== Testing if dataset is lexicographically sorted by package name ===")
# Check if the dataset is already sorted by package name
is_sorted_by_name = df['pkgname'].is_monotonic_increasing
print(f"Is dataset sorted by package name: {is_sorted_by_name}")

# Check if dataset is approximately sorted by name 
# (allowing for some variations in sorting logic)
name_rank = df['pkgname'].rank()
index_rank = df.index.rank()
rank_correlation = name_rank.corr(index_rank)
print(f"Rank correlation between index and package name: {rank_correlation:.4f}")

# Spearman rank correlation
spearman_corr, spearman_p = stats.spearmanr(df.index, df['pkgname'].rank())
print(f"Spearman rank correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")

# 3. Component-wise sorting check
print("\n=== Testing if sorted by package name components ===")
# Split package names by components and check sorting
df['pkg_company'] = df['pkgname'].apply(lambda x: x.split('.')[1] if len(x.split('.')) > 1 else '')
df['pkg_product'] = df['pkgname'].apply(lambda x: x.split('.')[2] if len(x.split('.')) > 2 else '')

# Check if sorted first by company
is_sorted_by_company = all(df['pkg_company'].iloc[i] <= df['pkg_company'].iloc[i+1] 
                          for i in range(len(df)-1))
print(f"Is sorted by company component: {is_sorted_by_company}")

# Within each company, check if sorted by product
company_product_sorted = True
companies_checked = 0
companies_sorted = 0

# Get list of companies with multiple apps
companies_with_multiple = df['pkg_company'].value_counts()[df['pkg_company'].value_counts() > 1].index.tolist()

# Sample up to 100 companies to check
sample_companies = companies_with_multiple[:100] if len(companies_with_multiple) > 100 else companies_with_multiple

for company in sample_companies:
    company_group = df[df['pkg_company'] == company]
    if len(company_group) > 1:
        companies_checked += 1
        is_product_sorted = company_group['pkg_product'].is_monotonic_increasing
        if is_product_sorted:
            companies_sorted += 1
        
print(f"Companies checked: {companies_checked}")
print(f"Companies with products in sorted order: {companies_sorted}")
print(f"Percentage of companies with sorted products: {companies_sorted/companies_checked*100:.2f}%")

# 4. Check mutual information directly (without CatBoost encoding)
print("\n=== Checking Mutual Information with raw package names ===")

# Determine if classification or regression task
# Adjust this based on your actual target variable
has_target = 'status' in df.columns or 'target' in df.columns
target_col = 'status' if 'status' in df.columns else 'target' if 'target' in df.columns else None

if has_target:
    target = df[target_col]
    is_classification = target.dtype == bool or len(target.unique()) < 10
    
    # Convert package names to category codes
    pkg_codes = df['pkgname'].astype('category').cat.codes.values.reshape(-1, 1)
    
    # Calculate MI directly
    if is_classification:
        raw_mi = mutual_info_classif(
            pkg_codes, 
            target,
            discrete_features=True
        )[0]
        print(f"MI with raw package name (classification): {raw_mi:.4f}")
    else:
        raw_mi = mutual_info_regression(
            pkg_codes, 
            target,
            discrete_features=True
        )[0]
        print(f"MI with raw package name (regression): {raw_mi:.4f}")
else:
    print("Target variable not found. Skipping MI calculation.")

# 5. Plot frequency distribution of first characters to see if ordering might be alphabetical
print("\n=== Checking alphabetical distribution ===")
# Get first character of each package name's company component
df['first_char'] = df['pkg_company'].str[0:1] if 'pkg_company' in df.columns else df['pkgname'].str[0:1]

# Plot distribution
plt.figure(figsize=(12, 6))
df['first_char'].value_counts().sort_index().plot(kind='bar')
plt.title("Distribution of First Characters in Package Names")
plt.xlabel("First Character")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("pkg_first_char_distribution.png")
plt.close()

# 6. Check sequential patterns of package names
print("\n=== Checking sequential patterns ===")
# Look at small consecutive slices of data to see if there's a pattern
slice_size = 50
num_slices = 3
for i in range(num_slices):
    start_idx = i * 1000
    end_idx = start_idx + slice_size
    slice_df = df.iloc[start_idx:end_idx]
    print(f"\nSlice {i+1} (indices {start_idx}-{end_idx}):")
    for j, row in slice_df.iterrows():
        print(f"{j}. {row['pkgname']}")

print("\nAnalysis complete!")