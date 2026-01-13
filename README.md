# Snorkel Indicator Importance

A SHAP-like importance analysis tool for Snorkel weak supervision labeling functions. This tool helps you understand which labeling functions (indicators) are most influential for each data point in your weak supervision pipeline.

## Overview

When using Snorkel for weak supervision, you define many labeling functions (LFs) that vote on labels. After training the label model, you get weights for each LF, but it's not always clear which LFs are driving decisions for specific instances. This library provides instance-level importance scores similar to SHAP values for your labeling functions.

## Features

- **Instance-level importance**: Understand which LFs matter most for each data point
- **Vectorized operations**: Process thousands of rows in milliseconds (50-100x faster than loops)
- **Outlier detection**: Identify when LFs fire unusually for specific instances
- **Multiple methods**: IQR, z-score, and percentile-based outlier detection
- **Visualization**: Plot importance scores for easy interpretation
- **XGBoost integration**: Combined explanations for Snorkel + XGBoost pipelines
- **Batch analysis**: Analyze multiple instances efficiently
- **Memory efficient**: Compute only what you need, when you need it

## Installation

```bash
# Core dependencies
pip install numpy pandas

# Optional: for visualization
pip install matplotlib

# Optional: for integration examples
pip install snorkel xgboost
```

## Quick Start

### Basic Usage

```python
import numpy as np
from indicator_importance import SnorkelIndicatorImportance

# Your indicator/LF matrix (n_samples, n_lfs)
indicator_matrix = np.array([...])  # From Snorkel's L matrix

# Weights learned by Snorkel's label model
snorkel_weights = np.array([...])  # From label_model.get_weights()

# Labeling function names
lf_names = ["LF_1", "LF_2", ..., "LF_n"]

# Initialize importance calculator
importance_calc = SnorkelIndicatorImportance(
    indicator_matrix=indicator_matrix,
    snorkel_weights=snorkel_weights,
    indicator_names=lf_names,
    outlier_method='iqr',  # 'iqr', 'zscore', or 'percentile'
    outlier_threshold=1.5
)

# Analyze a single instance
result = importance_calc.get_row_importance(row_idx=0, top_k=10)

print("Top 10 Most Important LFs:")
for lf_name, score in result.top_indicators:
    print(f"  {lf_name}: {score:.4f}")

print(f"\nOutlier LFs: {len(result.outlier_indicators)}")
for lf_name, value in result.outlier_indicators:
    print(f"  {lf_name}: {value:.4f}")
```

## Vectorized Operations (⚡ Fast!)

The library is optimized for processing thousands of rows at once using vectorized NumPy operations. This is **50-100x faster** than looping through rows.

### Get ALL importance scores at once

```python
# Compute importance for ALL rows in one operation
all_importance = importance_calc.get_all_importance_scores(normalize=True)
# Returns: (n_samples, n_indicators) matrix

# Or as a DataFrame with indicator names
importance_df = importance_calc.get_all_importance_scores(
    normalize=True, 
    as_dataframe=True
)
```

### Get top-k indicators for ALL rows

```python
# Efficiently get top-k indicators for all rows
top_indices, top_scores = importance_calc.get_top_k_matrix(top_k=10)
# Returns: (n_samples, top_k) matrices

# Access results for any row
row_0_top_indicators = [lf_names[i] for i in top_indices[0]]
row_0_top_scores = top_scores[0]
```

### Detect outliers for ALL rows

```python
# Detect outliers across all rows at once
all_outliers = importance_calc.detect_outliers_vectorized()
# Returns: (n_samples, n_indicators) boolean matrix

# Count outliers per row
outlier_counts = np.sum(all_outliers, axis=1)
```

### Performance Comparison

```python
import time

# OLD WAY: Loop through rows (slow)
start = time.time()
for i in range(1000):
    result = importance_calc.get_row_importance(i)
loop_time = time.time() - start
print(f"Loop: {loop_time:.2f}s")

# NEW WAY: Vectorized (fast!)
start = time.time()
results = importance_calc.explain_predictions(row_indices=list(range(1000)))
vectorized_time = time.time() - start
print(f"Vectorized: {vectorized_time:.2f}s")

print(f"Speedup: {loop_time/vectorized_time:.1f}x faster! 🚀")
# Typical output: Speedup: 50-100x faster!
```

### Analyze Multiple Instances

```python
# Get explanations for multiple instances
explanations = importance_calc.explain_predictions(
    row_indices=[0, 1, 2, 3, 4],
    top_k=5
)

# Create a summary DataFrame
summary_df = importance_calc.to_dataframe(
    row_indices=range(100),
    top_k=5
)
print(summary_df.head())
```

### Visualize Importance

```python
# Plot importance for a single instance
fig = importance_calc.plot_importance(
    row_idx=0,
    top_k=15,
    figsize=(10, 6)
)
```

## How It Works

### Importance Score Calculation

For each instance, the importance score combines:

1. **Indicator Value**: How strongly the LF fires for this instance
2. **Snorkel Weight**: The global importance of this LF (learned by Snorkel)

```
importance_score = indicator_value * snorkel_weight
```

Scores are normalized by default so they sum to 1, making them interpretable as relative importance.

### Outlier Detection

Three methods are available:

1. **IQR (Interquartile Range)**: Default method
   - Detects values outside [Q1 - threshold×IQR, Q3 + threshold×IQR]
   - Robust to extreme outliers
   
2. **Z-score**: 
   - Detects values with |z-score| > threshold
   - Sensitive to distribution shape
   
3. **Percentile**:
   - Detects values in top/bottom threshold percentile
   - Good for skewed distributions

## Integration with XGBoost

For complete pipeline explanations (Snorkel → XGBoost):

```python
from integration_example import SnorkelXGBoostExplainer

explainer = SnorkelXGBoostExplainer(
    label_model=trained_label_model,
    xgboost_model=trained_xgb_model,
    L_matrix=L_matrix,  # Snorkel's label matrix
    lf_names=lf_names,
    feature_matrix=X,  # Features for XGBoost
    feature_names=feature_names
)

# Get comprehensive explanation
explanation = explainer.explain_instance(row_idx=0)

# Generate human-readable report
report = explainer.create_explanation_report(row_idx=0)
print(report)
```

## API Reference

### SnorkelIndicatorImportance

#### Constructor Parameters

- `indicator_matrix`: (np.ndarray or pd.DataFrame) Matrix of indicator values, shape (n_samples, n_indicators)
- `snorkel_weights`: (np.ndarray, dict, or pd.Series) Learned weights from Snorkel
- `indicator_names`: (list of str, optional) Names of indicators
- `outlier_method`: (str, default='iqr') Method for outlier detection: 'iqr', 'zscore', or 'percentile'
- `outlier_threshold`: (float, default=1.5) Threshold for outlier detection

#### Key Methods

**`compute_importance_scores(row_idx, normalize=True, consider_sign=True)`**
- Compute raw importance scores for a single row
- Returns: np.ndarray of importance scores

**`compute_importance_scores_vectorized(row_indices=None, normalize=True, consider_sign=True)`**
- Compute importance scores for multiple rows at once (VECTORIZED, FAST)
- If row_indices is None, computes for all rows
- Returns: np.ndarray of shape (n_rows, n_indicators)

**`detect_outliers(row_idx)`**
- Detect which indicators are outliers for a row
- Returns: Boolean array indicating outliers

**`detect_outliers_vectorized(row_indices=None)`**
- Detect outliers for multiple rows at once (VECTORIZED)
- Returns: Boolean matrix of shape (n_rows, n_indicators)

**`get_row_importance(row_idx, top_k=10, normalize=True, include_outliers=True)`**
- Get comprehensive importance analysis for a row
- Returns: IndicatorImportance object containing:
  - `top_indicators`: List of (name, score) tuples
  - `outlier_indicators`: List of (name, value) tuples
  - `all_scores`: Dictionary of all scores

**`explain_predictions(row_indices=None, top_k=10, normalize=True)`**
- Explain multiple instances (uses vectorization internally for speed)
- Returns: List of IndicatorImportance objects

**`get_top_k_matrix(row_indices=None, top_k=10, normalize=True)`**
- Get top-k indicator indices and scores for multiple rows (VECTORIZED)
- Most efficient way to get top indicators for many rows
- Returns: Tuple of (top_k_indices, top_k_scores), both shape (n_rows, top_k)

**`get_all_importance_scores(normalize=True, as_dataframe=False)`**
- Get importance scores for ALL rows and ALL indicators (VECTORIZED)
- Most efficient way to get complete importance matrix
- Returns: np.ndarray or pd.DataFrame of shape (n_samples, n_indicators)

**`to_dataframe(row_indices=None, top_k=5)`**
- Create DataFrame summary
- Returns: pd.DataFrame with top indicators per row

**`plot_importance(row_idx, top_k=15, figsize=(10,6), title=None)`**
- Visualize importance scores
- Returns: matplotlib Figure object

## Use Cases

### 1. Debugging Weak Supervision

Identify which LFs are driving incorrect labels:

```python
# Find instances where prediction is wrong
wrong_predictions = (predicted_labels != true_labels)
wrong_indices = np.where(wrong_predictions)[0]

# Analyze top mistakes
for idx in wrong_indices[:10]:
    result = importance_calc.get_row_importance(idx, top_k=5)
    print(f"\nInstance {idx} (True: {true_labels[idx]}, Pred: {predicted_labels[idx]})")
    print("Top contributing LFs:")
    for lf_name, score in result.top_indicators:
        print(f"  {lf_name}: {score:.4f}")
```

### 2. LF Quality Assessment

Find LFs that frequently appear as outliers:

```python
outlier_counts = {}
for idx in range(len(indicator_matrix)):
    result = importance_calc.get_row_importance(idx)
    for lf_name, _ in result.outlier_indicators:
        outlier_counts[lf_name] = outlier_counts.get(lf_name, 0) + 1

print("LFs with most outlier behavior:")
for lf_name, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {lf_name}: {count} instances")
```

### 3. Instance Similarity

Find instances with similar LF importance patterns:

```python
from scipy.spatial.distance import cosine

def find_similar_instances(target_idx, importance_calc, top_n=5):
    target_scores = importance_calc.compute_importance_scores(target_idx)
    
    similarities = []
    for idx in range(len(importance_calc.indicator_matrix)):
        if idx == target_idx:
            continue
        scores = importance_calc.compute_importance_scores(idx)
        sim = 1 - cosine(target_scores, scores)
        similarities.append((idx, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

similar = find_similar_instances(0, importance_calc)
print(f"Instances most similar to instance 0:")
for idx, sim in similar:
    print(f"  Instance {idx}: similarity = {sim:.4f}")
```

### 4. Feature Engineering Insights

Identify which LF combinations are important:

```python
from collections import Counter

# Track which LF pairs co-occur in top-k
cooccurrences = Counter()

for idx in range(len(indicator_matrix)):
    result = importance_calc.get_row_importance(idx, top_k=5)
    top_lfs = [name for name, _ in result.top_indicators]
    
    # Count all pairs
    for i in range(len(top_lfs)):
        for j in range(i+1, len(top_lfs)):
            pair = tuple(sorted([top_lfs[i], top_lfs[j]]))
            cooccurrences[pair] += 1

print("Most common LF pairs in top-5:")
for pair, count in cooccurrences.most_common(10):
    print(f"  {pair}: {count} instances")
```

## Performance Tips

1. **Batch Processing**: Use `explain_predictions()` for multiple instances instead of looping
2. **Memory**: For large datasets, process in chunks
3. **Caching**: Store computed statistics if analyzing the same dataset multiple times
4. **Outlier Method**: IQR is fastest, z-score requires computing all statistics

## Comparison with SHAP

| Feature | SHAP | Indicator Importance |
|---------|------|---------------------|
| Use case | Model predictions | Weak supervision LFs |
| Granularity | Feature-level | LF-level |
| Computation | Game theory | Weighted indicators |
| Outliers | Not built-in | Built-in detection |
| Integration | Model-agnostic | Snorkel-specific |

## Common Issues

### Issue: Weights sum to zero
**Solution**: This can happen if Snorkel learned equal positive/negative weights. Use `consider_sign=False` in `compute_importance_scores()` to use absolute values.

### Issue: All indicators marked as outliers
**Solution**: Adjust `outlier_threshold` parameter. Try 2.0 or 2.5 for IQR method, or 3.0 for z-score method.

### Issue: ImportError for plotting
**Solution**: Install matplotlib: `pip install matplotlib`

## Contributing

Contributions are welcome! Areas for improvement:
- Additional outlier detection methods
- Support for multi-class weak supervision
- Integration with other weak supervision frameworks
- Performance optimizations for large-scale data

## License

MIT License

## Citation

If you use this in research, please cite:

```bibtex
@software{snorkel_indicator_importance,
  title = {Snorkel Indicator Importance: Instance-level Explanations for Weak Supervision},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/snorkel-indicator-importance}
}
```

## Related Work

- [Snorkel](https://www.snorkel.org/): Weak supervision framework
- [SHAP](https://github.com/slundberg/shap): Model explanation library
- [LIME](https://github.com/marcotcr/lime): Local interpretable model explanations
