# Handling Clustered Weights - Quick Start Guide

## The Problem

When Snorkel learns similar weights for many labeling functions (LFs), traditional cosine similarity becomes less effective at distinguishing between instances. This happens because:

```python
importance = indicator_value × weight

# When weights cluster around similar values:
# weight_1 ≈ weight_2 ≈ ... ≈ weight_n ≈ 0.5
# Then importance differences come mainly from indicator_values
# But if those also cluster, everything looks similar!
```

**Symptom**: Cosine similarities cluster tightly (e.g., all between 0.70-0.75), making it hard to distinguish truly similar instances from merely correlated ones.

## Quick Diagnosis

```python
import numpy as np

# Check if your weights cluster
weight_std = np.std(snorkel_weights)

if weight_std < 0.2:
    print("⚠️ Weights are clustered - use alternative methods!")
```

## Solutions

### 1. Use Pattern-Based Similarity (Best for Clustered Weights)

Focus on **which** indicators fired, ignoring weights entirely:

```python
similar = calc.find_similar_instances(
    target_idx=0,
    top_k=10,
    method='hamming'  # or 'pattern'
)
```

**When to use**: Weights are very similar (std < 0.2)

### 2. Use Deviation-Based Importance

Emphasize indicators that fire **unusually** for this instance:

```python
# Compute deviation importance (emphasizes unusual patterns)
deviation_scores = calc.compute_deviation_importance(normalize=True)

# Find similar instances based on deviation
similar = calc.find_similar_instances(
    target_idx=0,
    top_k=10,
    method='deviation'
)
```

**When to use**: Looking for outliers or unique instances

### 3. Use Combined Method (Recommended Default)

Weighted combination of importance + pattern + deviation:

```python
similar = calc.find_similar_instances(
    target_idx=0,
    top_k=10,
    method='combined'  # DEFAULT - works well in all cases
)
```

**When to use**: As your default - robust across scenarios

## Comparison of Methods

| Method | Focuses On | Best When | Computation |
|--------|-----------|-----------|-------------|
| `cosine` | Weighted importance | Weights are diverse | Fast |
| `hamming` | Which indicators fired | Weights cluster | Fast |
| `pattern` | Jaccard similarity | Sparse indicators | Medium |
| `deviation` | Unusual patterns | Finding outliers | Fast |
| `combined` | All three aspects | General purpose | Medium |

## Essential Visualizations

### 1. Indicator Pattern Heatmap

Shows which instances share similar patterns:

```python
fig = calc.plot_indicator_heatmap(
    row_indices=list(range(100)),
    use_importance=False,  # Use raw patterns when weights cluster
    cluster_rows=True,
    cluster_cols=True,
    figsize=(14, 10)
)
plt.savefig('pattern_heatmap.png')
```

**Insight**: Clustering reveals natural groupings despite similar weights.

### 2. Similarity Matrix Comparison

Compare discrimination quality of different methods:

```python
# Poor discrimination (when weights cluster)
fig1 = calc.plot_similarity_matrix(
    row_indices=list(range(50)),
    method='cosine'
)

# Better discrimination
fig2 = calc.plot_similarity_matrix(
    row_indices=list(range(50)),
    method='hamming'
)
```

**Insight**: Visual proof that alternative methods work better.

### 3. PCA Projection

See how instances cluster in pattern space:

```python
# Color by outlier count or labels
outlier_counts = np.sum(calc.detect_outliers_vectorized(), axis=1)

fig = calc.plot_pattern_pca(
    use_importance=False,  # Use patterns, not weighted
    color_by=outlier_counts,
    figsize=(12, 9)
)
```

**Insight**: Understand pattern diversity and identify outlier clusters.

### 4. Instance Comparison

Compare similar instances side-by-side:

```python
# Get similar instances
similar = calc.find_similar_instances(0, top_k=3, method='combined')
similar_rows = [idx for idx, _ in similar]

# Compare visually
fig = calc.plot_instance_comparison(
    row_indices=[0] + similar_rows,
    top_k=10
)
```

**Insight**: Understand subtle differences between "similar" instances.

### 5. Radar Chart

Show importance profile for a single instance:

```python
fig = calc.plot_indicator_radar(
    row_idx=0,
    top_k=10
)
```

**Insight**: Visual summary of which indicators matter most.

## Complete Example

```python
from indicator_importance import SnorkelIndicatorImportance
import numpy as np
import matplotlib.pyplot as plt

# Initialize
calc = SnorkelIndicatorImportance(
    indicator_matrix=L_matrix,  # Your Snorkel L matrix
    snorkel_weights=label_model.get_weights(),
    indicator_names=lf_names
)

# Check weight clustering
weight_std = np.std(calc.weights)
print(f"Weight std: {weight_std:.3f}")

if weight_std < 0.2:
    print("⚠️ Using alternative methods for clustered weights")
    similarity_method = 'combined'
else:
    print("✓ Weights are diverse, cosine similarity OK")
    similarity_method = 'cosine'

# Find similar instances
target_idx = 42
similar = calc.find_similar_instances(
    target_idx=target_idx,
    top_k=10,
    method=similarity_method
)

print(f"\nTop 10 similar to row {target_idx}:")
for rank, (idx, score) in enumerate(similar, 1):
    print(f"  {rank}. Row {idx}: {score:.4f}")

# Visualize patterns
fig = calc.plot_indicator_heatmap(
    row_indices=list(range(100)),
    use_importance=(weight_std >= 0.2),  # Adapt based on clustering
    cluster_rows=True,
    cluster_cols=True
)
plt.savefig('patterns.png')

# PCA projection
outliers = np.sum(calc.detect_outliers_vectorized(), axis=1)
fig = calc.plot_pattern_pca(
    use_importance=(weight_std >= 0.2),
    color_by=outliers
)
plt.savefig('pca.png')
```

## API Reference

### New Methods for Clustered Weights

**`find_similar_instances(target_idx, top_k, method='combined')`**
- Find instances most similar to target
- Methods: 'cosine', 'euclidean', 'manhattan', 'hamming', 'pattern', 'deviation', 'combined'
- Returns: List of (index, similarity_score) tuples

**`get_indicator_patterns(binarize=True)`**
- Get which indicators fired (ignores weights)
- Returns: Binary or raw pattern matrix

**`compute_deviation_importance(normalize=True)`**
- Compute importance based on deviation from mean
- Returns: Deviation importance matrix

**`compute_instance_diversity(method='hamming')`**
- Compute how unique each instance is
- Returns: Diversity scores per instance

### Visualization Methods

**`plot_indicator_heatmap(...)`**
- Shows pattern matrix across instances
- Params: `use_importance`, `cluster_rows`, `cluster_cols`

**`plot_similarity_matrix(...)`**
- Shows pairwise similarities
- Params: `method`, `cluster`

**`plot_pattern_pca(...)`**
- 2D PCA projection of patterns
- Params: `use_importance`, `color_by`

**`plot_instance_comparison(...)`**
- Side-by-side comparison of instances
- Params: `row_indices`, `top_k`

**`plot_indicator_radar(...)`**
- Radar chart for single instance
- Params: `row_idx`, `top_k`

## Decision Tree

```
Start: Do you have clustered weights?
│
├─ Yes (std < 0.2)
│  ├─ Use method='hamming' or 'combined'
│  ├─ Visualize with use_importance=False
│  └─ Focus on pattern-based analysis
│
└─ No (std >= 0.2)
   ├─ Use method='cosine' (default)
   ├─ Visualize with use_importance=True
   └─ Standard importance analysis works well
```

## Performance Tips

1. **Hamming distance** is fastest for pattern comparison
2. **Combined method** is slower but more robust (recommended)
3. **Similarity matrices** are O(n²) - use on subsets (n < 100)
4. **PCA projections** are fast even for large datasets

## Common Mistakes to Avoid

❌ **Don't**: Use cosine similarity when weights cluster
```python
# Bad: Weights cluster, cosine fails
similar = calc.find_similar_instances(0, method='cosine')  
```

✅ **Do**: Use pattern or combined methods
```python
# Good: Works well with clustered weights
similar = calc.find_similar_instances(0, method='combined')
```

❌ **Don't**: Ignore weight clustering in your analysis
```python
# Bad: Assuming all similarity methods work equally
```

✅ **Do**: Check weight std and adapt your approach
```python
weight_std = np.std(calc.weights)
method = 'combined' if weight_std < 0.2 else 'cosine'
```

❌ **Don't**: Use importance-weighted visualizations when weights cluster
```python
# Bad: Hard to see patterns
fig = calc.plot_indicator_heatmap(use_importance=True)
```

✅ **Do**: Use pattern-based visualizations
```python
# Good: Shows actual patterns
fig = calc.plot_indicator_heatmap(use_importance=False)
```

## Example Output Interpretation

### Cosine Similarity (Clustered Weights)
```
Row 1: similarity = 0.742
Row 2: similarity = 0.738
Row 3: similarity = 0.735
...
Std: 0.018  ← Too clustered! Can't discriminate well
```

### Hamming Distance (Pattern-Based)
```
Row 1: distance = 0.225
Row 2: distance = 0.350
Row 3: distance = 0.475
...
Std: 0.087  ← Better discrimination!
```

## Further Reading

- See `clustered_weights_example.py` for complete working example
- See `use_cases.py` for production use cases
- See `benchmark.py` for performance comparisons

## Quick Troubleshooting

**Q: All my similarities are very high (>0.9) or very low (<0.1)**
A: Try 'hamming' or 'pattern' method instead of 'cosine'

**Q: Similarity matrix looks uniform**
A: Weights are likely clustered - use method='hamming' or 'pattern'

**Q: PCA projection shows no structure**
A: Set `use_importance=False` to see pattern structure

**Q: Combined method is slow**
A: Use 'hamming' for production; combined is thorough but slower

**Q: How do I know if weights cluster?**
A: `np.std(weights) < 0.2` indicates clustering
