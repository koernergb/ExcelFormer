import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('./DATA/android_security/corrected_permacts.csv')

# 1. Basic statistics about index vs. features
print("========== ANALYZING INDEX PATTERNS ==========")

# Check if the data is already sorted by some column
for col in df.columns:
    is_sorted = df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing
    if is_sorted:
        print(f"Data appears to be sorted by {col}")

# 2. Look for transitions in pkgname and DevRegisteredDomain
print("\nAnalyzing package and domain patterns...")
df['pkgname_prefix'] = df['pkgname'].apply(lambda x: '.'.join(x.split('.')[:2]) if isinstance(x, str) else 'unknown')

# Find where package prefixes change
df['prefix_change'] = df['pkgname_prefix'] != df['pkgname_prefix'].shift(1)

# Print package prefix stats
print(f"\nNumber of distinct package prefix segments: {df['prefix_change'].sum()}")

# Print DevRegisteredDomain stats
print("\nDomain Registration Stats:")
print(f"Apps with registered domains: {df['DevRegisteredDomain'].sum()}")
print(f"Apps without registered domains: {len(df) - df['DevRegisteredDomain'].sum()}")
print(f"Percentage with registered domains: {(df['DevRegisteredDomain'].sum() / len(df)) * 100:.2f}%")

# 3. Check correlations using Unnamed: 0 as index
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
correlations = {}
for col in numeric_cols:
    if col != 'Unnamed: 0':  # Skip correlating with itself
        correlations[col] = df['Unnamed: 0'].corr(df[col])

print(f"\nTop 10 numeric correlations with index:")
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
for col, corr in sorted_correlations[:10]:
    print(f"{col}: {corr:.3f}")

# 4. Check if index relates to feature importance (MI scores)
# Using a RF model to predict a target variable
# Assuming the MI scores are in a column called 'MI_score' or similar
if 'MI_score' in df.columns:
    rf = RandomForestRegressor(n_estimators=50)
    rf.fit(df[['Unnamed: 0']], df['MI_score'])
    print(f"R² for index predicting MI scores: {rf.score(df[['Unnamed: 0']], df['MI_score'])}")

# 5. Look at developer grouping patterns
segments = []
current_segment = []
current_domain = None

for i, row in df.iterrows():
    if row['DevRegisteredDomain'] != current_domain:
        if current_segment:
            segments.append((current_domain, len(current_segment)))
        current_segment = [i]
        current_domain = row['DevRegisteredDomain']
    else:
        current_segment.append(i)

# Add the last segment
if current_segment:
    segments.append((current_domain, len(current_segment)))

# Analyze segment sizes
segment_sizes = [size for _, size in segments]
print(f"Number of developer segments: {len(segments)}")
print(f"Average segment size: {np.mean(segment_sizes)}")
print(f"Min/Max segment size: {np.min(segment_sizes)}/{np.max(segment_sizes)}")

# 6. Output the first change points for inspection
change_points = df[df['prefix_change']].head(20)
print("\nFirst 20 transition points between domains:")
print(change_points[['Unnamed: 0', 'pkgname', 'DevRegisteredDomain']])

# 7. Check for timestamp pattern if available
if 'LastUpdated' in df.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Unnamed: 0'].values[::1000], df['LastUpdated'].values[::1000], alpha=0.5)
    plt.title('Index position vs LastUpdated date')
    plt.xlabel('Index Position')
    plt.ylabel('Last Updated')
    plt.savefig('index_vs_date.png')
    
    # Check if sorted by date within developer
    is_date_sorted_within_dev = True
    for domain, group in df.groupby('DevRegisteredDomain'):
        if len(group) > 1 and not group['LastUpdated'].is_monotonic_increasing:
            is_date_sorted_within_dev = False
            break
    
    print(f"Records appear to be sorted by date within developer: {is_date_sorted_within_dev}")

# Save key findings to a text file
with open('index_analysis_results.txt', 'w') as f:
    f.write("Index pattern analysis results\n")
    f.write(f"Number of domain segments: {len(segments)}\n")
    f.write(f"Top domains by segment size:\n")
    for domain, size in sorted(segments, key=lambda x: x[1], reverse=True)[:10]:
        f.write(f"  {domain}: {size} entries\n")

# First calculate correlations with index to get weights
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
correlations = {}
for col in numeric_cols:
    if col != 'Unnamed: 0':  # Skip correlating with itself
        correlations[col] = df['Unnamed: 0'].corr(df[col])

# Create app quality score using correlation coefficients as weights
quality_components = {
    'LenTitle': 0.189,                    # Positive weight (longer titles better)
    'isSpamming': -0.182,                 # Negative weight (spam is bad)
    'max_downloads_log': 0.130,           # Positive weight (more downloads better)
    'days_since_last_update': -0.130      # Negative weight (older updates worse)
}

# Initialize quality score
df['app_quality_score'] = 0

# Add each component, handling missing values
for feature, weight in quality_components.items():
    if feature in df.columns:
        # Fill NaN with mean for numeric features
        feature_values = df[feature].fillna(df[feature].mean())
        df['app_quality_score'] += feature_values * weight

# Add binary features
if 'developer_website' in df.columns:
    df['app_quality_score'] += df['developer_website'].notnull().astype(int) * 0.152
if 'privacy_policy_link' in df.columns:
    df['app_quality_score'] += df['privacy_policy_link'].notnull().astype(int) * 0.100

# Normalize to 0-1 range
df['app_quality_score'] = MinMaxScaler().fit_transform(df[['app_quality_score']])

# Print statistics about the new feature
print("\nApp Quality Score Statistics:")
print(df['app_quality_score'].describe())

# Calculate correlation with index to verify
print(f"\nCorrelation with index: {df['app_quality_score'].corr(df['Unnamed: 0']):.3f}")