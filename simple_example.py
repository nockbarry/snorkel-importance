"""
Simple standalone example of using SnorkelIndicatorImportance

This script demonstrates the core functionality without requiring Snorkel or XGBoost.
"""

import numpy as np
import sys
sys.path.append('.')
from indicator_importance import SnorkelIndicatorImportance

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("Snorkel Indicator Importance - Simple Example")
print("=" * 80)

# Step 1: Create synthetic data
print("\n1. Creating synthetic weak supervision data...")
n_samples = 100
n_indicators = 20

# Simulate indicator matrix (e.g., labeling function outputs)
# Values represent: -1 (negative vote), 0 (abstain), 1 (positive vote)
indicator_matrix = np.random.choice([-1, 0, 1], size=(n_samples, n_indicators), p=[0.2, 0.5, 0.3])

# Add some correlation to make it more realistic
for i in range(n_samples):
    if np.random.rand() > 0.7:
        # Some instances have correlated indicators
        indicator_matrix[i, 0:5] = 1
    if np.random.rand() > 0.8:
        # Some instances trigger different patterns
        indicator_matrix[i, 10:15] = -1

# Simulate Snorkel-learned weights (some LFs are more important than others)
snorkel_weights = np.random.randn(n_indicators) * 0.5
# Make some weights explicitly strong
snorkel_weights[0] = 2.0  # Very important positive LF
snorkel_weights[5] = -1.8  # Important negative LF
snorkel_weights[10] = 0.1  # Weak LF

# Create indicator names
indicator_names = [
    f"LF_{i}_{['weak', 'moderate', 'strong'][i % 3]}" 
    for i in range(n_indicators)
]

print(f"   - Created {n_samples} samples with {n_indicators} labeling functions")
print(f"   - Indicator matrix shape: {indicator_matrix.shape}")
print(f"   - Weight range: [{snorkel_weights.min():.2f}, {snorkel_weights.max():.2f}]")

# Step 2: Initialize the importance calculator
print("\n2. Initializing SnorkelIndicatorImportance...")
importance_calc = SnorkelIndicatorImportance(
    indicator_matrix=indicator_matrix,
    snorkel_weights=snorkel_weights,
    indicator_names=indicator_names,
    outlier_method='iqr',
    outlier_threshold=1.5
)
print("   ✓ Importance calculator initialized")

# Step 3: Analyze a single instance
print("\n3. Analyzing Instance 0...")
print("-" * 80)
result = importance_calc.get_row_importance(row_idx=0, top_k=10)

print(f"\nTop 10 Most Important Indicators for Instance 0:")
print(f"{'Rank':<6} {'Indicator':<25} {'Score':<12} {'Value':<8}")
print("-" * 55)
for i, (name, score) in enumerate(result.top_indicators, 1):
    # Get the actual indicator value
    idx = indicator_names.index(name)
    value = indicator_matrix[0, idx]
    print(f"{i:<6} {name:<25} {score:>10.4f}   {value:>6}")

if result.outlier_indicators:
    print(f"\n⚠ Outlier Indicators Detected ({len(result.outlier_indicators)} total):")
    print(f"{'Indicator':<25} {'Value':<8}")
    print("-" * 35)
    for name, value in result.outlier_indicators[:5]:
        print(f"{name:<25} {value:>6.2f}")
else:
    print("\n✓ No outlier indicators detected for this instance")

# Step 4: Compare multiple instances
print("\n4. Comparing Multiple Instances...")
print("-" * 80)

# Analyze first 5 instances
instances_to_analyze = [0, 1, 2, 3, 4]
print(f"\nTop 3 indicators for each of the first {len(instances_to_analyze)} instances:\n")

for idx in instances_to_analyze:
    result = importance_calc.get_row_importance(row_idx=idx, top_k=3)
    print(f"Instance {idx}:")
    for rank, (name, score) in enumerate(result.top_indicators, 1):
        print(f"  {rank}. {name}: {score:.4f}")
    print()

# Step 5: Create a summary DataFrame
print("\n5. Creating Summary DataFrame...")
summary_df = importance_calc.to_dataframe(row_indices=range(10), top_k=3)
print("\nSummary of first 10 instances (top 3 indicators each):")
print(summary_df.to_string(index=False))

# Step 6: Find instances with unusual patterns
print("\n6. Finding Instances with Unusual Patterns...")
print("-" * 80)

outlier_counts = []
for idx in range(n_samples):
    result = importance_calc.get_row_importance(idx)
    outlier_counts.append((idx, len(result.outlier_indicators)))

# Sort by number of outliers
outlier_counts.sort(key=lambda x: x[1], reverse=True)

print(f"\nTop 5 instances with most outlier indicators:")
print(f"{'Instance':<12} {'Outlier Count':<15}")
print("-" * 30)
for idx, count in outlier_counts[:5]:
    print(f"{idx:<12} {count:<15}")

# Step 7: Identify most important indicators globally
print("\n7. Global Indicator Importance Analysis...")
print("-" * 80)

# Aggregate importance across all instances
importance_aggregates = {name: [] for name in indicator_names}

for idx in range(n_samples):
    result = importance_calc.get_row_importance(idx, top_k=n_indicators)
    for name, score in result.top_indicators:
        importance_aggregates[name].append(abs(score))

# Compute average absolute importance
avg_importance = {
    name: np.mean(scores) 
    for name, scores in importance_aggregates.items()
}

# Sort by importance
sorted_indicators = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 10 Most Important Indicators (averaged across all instances):")
print(f"{'Rank':<6} {'Indicator':<25} {'Avg Abs Importance':<20} {'Weight':<10}")
print("-" * 65)
for rank, (name, avg_score) in enumerate(sorted_indicators[:10], 1):
    idx = indicator_names.index(name)
    weight = snorkel_weights[idx]
    print(f"{rank:<6} {name:<25} {avg_score:>18.4f}   {weight:>8.2f}")

# Step 8: Practical insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n✓ How to interpret the results:")
print("  1. Importance Score: Higher absolute values indicate stronger influence")
print("  2. Positive scores: Indicators push toward positive prediction")
print("  3. Negative scores: Indicators push toward negative prediction")
print("  4. Outliers: Unusual indicator values that may warrant investigation")

print("\n✓ Practical applications:")
print("  1. Debug weak supervision: Identify which LFs drive incorrect predictions")
print("  2. LF quality assessment: Find LFs that behave inconsistently")
print("  3. Feature engineering: Discover important LF combinations")
print("  4. Data quality: Detect unusual patterns in your data")

print("\n✓ Next steps:")
print("  1. Integrate with your actual Snorkel pipeline")
print("  2. Use with XGBoost for end-to-end explanations (see integration_example.py)")
print("  3. Adjust outlier_threshold based on your data distribution")
print("  4. Visualize results using plot_importance() method")

print("\n" + "=" * 80)
print("Example completed successfully!")
print("=" * 80)
