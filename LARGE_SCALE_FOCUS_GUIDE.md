# Large-Scale Analysis with Focus Rows - Complete Guide

## For Your 5M × 160 Dataset

This script is specifically designed for your use case: **5 million rows** with **~160 indicators** where you want **full analysis** (no skipping) and **detailed focus on specific rows**.

## Quick Start

```python
from large_scale_analysis import analyze_large_dataset
import pandas as pd

# Your data
df = pd.read_csv('your_data.csv')  # 5M rows
L_matrix = df[indicator_columns].values  # or your Snorkel L matrix
weights = label_model.get_weights()
lf_names = list(indicator_columns)

# Rows you're interested in (by position in matrix or pandas index)
interesting_rows = [42, 100, 500, 1000, 5000, 10000]

# Run complete analysis
results = analyze_large_dataset(
    indicator_matrix=L_matrix,
    snorkel_weights=weights,
    indicator_names=lf_names,
    pandas_index=df.index.values,  # Uses your actual row labels!
    focus_rows=interesting_rows,
    viz_sample_size=10000,        # Large sample for good structure
    sampling_method='clustering',  # Smart sampling
    output_dir="./analysis_5M"
)
```

## What You Get

### Global Analysis (All 5M Rows)
1. **Weight analysis** - Clustering detection
2. **All importance scores** - Complete 5M × 160 matrix
3. **Outlier detection** - Across all 5M rows  
4. **Top 20 indicators** - Globally important LFs

### Global Visualizations (Smart Sampled)
1. **Pattern heatmap** - Clustered to show structure
2. **PCA projection** - Colored by outlier count
3. **Top indicators chart** - Bar chart of top 10
4. **Outlier distribution** - Histogram across all rows

### Focus Row Analysis (Per Row)
For each of your focus rows, you get **4 detailed visualizations**:

1. **Top 20 indicators bar chart** - Most important for this row
2. **Radar chart** - Importance profile
3. **Comparison** - Side-by-side with 3 most similar rows
4. **Heatmap** - Indicator values vs similar instances

**Plus detailed JSON data** with:
- Top 30 indicators
- All outliers
- Top 10 similar instances
- Deviation importance

## Key Features

### 1. Pandas Index Support

```python
# Your DataFrame
df = pd.DataFrame(L_matrix, index=your_custom_index)

results = analyze_large_dataset(
    indicator_matrix=df,  # Can pass DataFrame directly!
    pandas_index=df.index.values,  # Or extract index
    ...
)

# All labels use your pandas index
# visualizations show "your_index_value" not "row_42"
```

### 2. Smart Sampling for Visualizations

Three methods to preserve structure:

**Clustering (Recommended for 5M rows)**
```python
sampling_method='clustering'
# - Uses MiniBatchKMeans on indicator patterns
# - Samples from each cluster
# - Preserves diversity and structure
# - Best for understanding global patterns
```

**Stratified (By Outliers)**
```python
sampling_method='stratified'
# - Divides by outlier count quartiles
# - Samples evenly from each
# - Good for balanced view
```

**Random**
```python
sampling_method='random'
# - Simple random sampling
# - Fastest but may miss structure
```

### 3. Flexible Sample Sizes

```python
# For 5M rows, we recommend:
viz_sample_size=10000  # 10K sample captures structure well

# Too small (<1000): Misses patterns
# Sweet spot (5K-20K): Good structure + fast
# Too large (>50K): Slow with minimal benefit
```

### 4. Focus Rows Can Be Anything

```python
# By position in matrix
focus_rows=[0, 100, 1000]

# If using pandas index
focus_rows = [df.index.get_loc(idx) for idx in ['ID_42', 'ID_100']]

# High outlier rows
outlier_counts = calc.detect_outliers_vectorized().sum(axis=1)
focus_rows = np.argsort(outlier_counts)[-10:]  # Top 10 outlier rows

# Random sample
focus_rows = np.random.choice(len(L_matrix), size=20, replace=False)
```

## Performance

### Expected Runtime (5M × 160)

| Step | Time | Rate |
|------|------|------|
| Importance (all rows) | ~30s | 166K rows/sec |
| Outliers (all rows) | ~40s | 125K rows/sec |
| Focus rows (4 rows) | ~10s | 4 viz/row |
| Global viz (10K sample) | ~30s | Clustering + plots |
| **Total** | **~2 minutes** | ⚡ Fast! |

### Memory Usage

| Component | Memory |
|-----------|--------|
| Input matrix (5M × 160) | ~800 MB |
| Importance matrix | ~3.2 GB |
| Outliers | ~800 MB |
| **Peak** | **~6-8 GB** |

**Recommendation**: 16 GB RAM minimum

## Complete Example

```python
from large_scale_analysis import LargeScaleAnalysis
import pandas as pd
import numpy as np

# Load your data
print("Loading data...")
df = pd.read_csv('snorkel_results.csv')  # 5M rows
L_matrix = df[[c for c in df.columns if c.startswith('LF_')]].values
lf_names = [c for c in df.columns if c.startswith('LF_')]

# Load weights
import pickle
with open('label_model.pkl', 'rb') as f:
    label_model = pickle.load(f)
weights = label_model.get_weights()

# Identify interesting rows
# Example: High outlier count rows
print("Pre-screening for interesting rows...")
from indicator_importance import SnorkelIndicatorImportance

quick_calc = SnorkelIndicatorImportance(L_matrix, weights, lf_names)
outlier_counts = quick_calc.detect_outliers_vectorized().sum(axis=1)

# Top 20 outlier rows
focus_rows = np.argsort(outlier_counts)[-20:].tolist()
print(f"Focus on {len(focus_rows)} high-outlier rows")

# Run full analysis
print("Running full analysis...")
analysis = LargeScaleAnalysis(
    indicator_matrix=L_matrix,
    snorkel_weights=weights,
    indicator_names=lf_names,
    pandas_index=df.index.values,
    output_dir="./production_analysis"
)

results = analysis.run_full_analysis(
    focus_rows=focus_rows,
    viz_sample_size=15000,  # Larger sample for production
    sampling_method='clustering',
    diversity_sample_size=5000
)

print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"Top indicator: {results['top_indicators'][0]['name']}")
print(f"Weights clustered: {results['weights']['clustered']}")
print(f"Outlier rate: {results['outliers']['pct_with_outliers']:.1f}%")
print(f"\nFocus rows analyzed: {len(results['focus_analysis'])}")
for focus in results['focus_analysis'][:5]:
    print(f"  {focus['row_label']}: {focus['n_outliers']} outliers")
```

## Output Structure

```
analysis_5M/
├── results.json              # Complete results
├── report.txt                # Human-readable summary
├── global_heatmap.png        # Pattern heatmap (clustered)
├── global_pca.png            # PCA projection
├── global_top_indicators.png # Top 10 bar chart
├── global_outliers.png       # Outlier distribution
└── focus_rows/
    ├── row_42_top20.png           # 4 viz per row
    ├── row_42_radar.png
    ├── row_42_comparison.png
    ├── row_42_heatmap.png
    ├── row_100_top20.png
    ├── ...
```

## Advanced Usage

### Progressive Analysis

```python
# Step 1: Quick overview (small viz sample)
results1 = analysis.run_full_analysis(
    focus_rows=[],
    viz_sample_size=1000,
    sampling_method='stratified'
)

# Step 2: Based on results, deep dive on specific rows
interesting_rows = identify_interesting_rows(results1)
results2 = analysis.run_full_analysis(
    focus_rows=interesting_rows,
    viz_sample_size=10000,
    sampling_method='clustering'
)
```

### Analyze Specific Row

```python
# Just analyze one row in detail
focus_analysis = analysis.analyze_focus_row(
    row_idx=42,
    create_visualizations=True
)

print(focus_analysis['top_20_indicators'])
print(focus_analysis['top_10_similar'])
```

### Custom Sampling

```python
# Manual control over viz sample
from sklearn.cluster import MiniBatchKMeans

patterns = calc.get_indicator_patterns(binarize=True)
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000)
clusters = kmeans.fit_predict(patterns)

# One from each cluster
viz_indices = []
for i in range(100):
    cluster_members = np.where(clusters == i)[0]
    if len(cluster_members) > 0:
        viz_indices.append(np.random.choice(cluster_members))

viz_indices = np.array(viz_indices)
```

## Tips for 5M Rows

1. **Start with smaller viz samples** (1K) to test, then scale up
2. **Use clustering sampling** for best structure preservation
3. **Pre-identify focus rows** using outlier counts or errors
4. **Run on dedicated machine** with 16+ GB RAM
5. **Save intermediate results** (importance matrix is valuable)
6. **Process in chunks** if memory is tight:

```python
# For 10M+ rows
chunk_size = 1000000
for i in range(0, len(L_matrix), chunk_size):
    chunk = L_matrix[i:i+chunk_size]
    chunk_results = analyze_large_dataset(...)
    # Combine results
```

## Troubleshooting

**Q: Out of memory**
```python
# Reduce viz sample size
viz_sample_size=5000  # instead of 10000

# Or process fewer focus rows at once
focus_rows=interesting_rows[:10]  # first 10 only
```

**Q: Clustering too slow**
```python
# Use stratified instead
sampling_method='stratified'

# Or reduce viz sample
viz_sample_size=5000
```

**Q: Want more focus rows**
```python
# No problem! Analyze as many as you want
# Each gets 4 visualizations + JSON data
focus_rows=list(range(100))  # 100 rows = 400 visualizations
```

**Q: Custom row labels not showing**
```python
# Make sure you pass pandas_index
results = analyze_large_dataset(
    ...,
    pandas_index=df.index.values  # ← This!
)
```

## Comparison: This vs comprehensive_analysis.py

| Feature | comprehensive_analysis.py | large_scale_analysis.py |
|---------|--------------------------|-------------------------|
| Target size | <100K rows | Millions of rows |
| Focus rows | ❌ No | ✅ Yes, detailed |
| Pandas index | ❌ No | ✅ Yes |
| Smart sampling | ❌ Random only | ✅ Clustering + stratified |
| Auto-skip steps | ✅ Yes | ❌ No (runs everything) |
| Focus visualizations | ❌ No | ✅ 4 per row |
| Best for | Quick analysis | Production, deep dives |

## Next Steps

1. **Test on subset first** - Run on 10K rows to verify setup
2. **Identify focus rows** - Use outliers, errors, or random sample
3. **Run full analysis** - Takes ~2-3 minutes for 5M rows
4. **Review focus row viz** - Each row gets 4 detailed plots
5. **Iterate** - Add more focus rows as needed

Your 5M × 160 dataset will run in about **2-3 minutes** with **10K viz samples** and produce **comprehensive results** plus **detailed analysis of all your focus rows**! 🚀
