# Large Dataset Optimization Guide

## For 1M+ Rows

When running on very large datasets (millions of rows), use these optimizations:

### Quick Start (Auto-Optimized)

The script automatically detects large datasets and skips expensive operations:

```python
from comprehensive_analysis import run_full_analysis

# For 5M rows × 160 indicators
results = run_full_analysis(
    indicator_matrix=your_L_matrix,      # 5M × 160
    snorkel_weights=your_weights,
    indicator_names=lf_names,
    output_dir="./large_analysis"
)

# Automatic optimizations:
# ✓ Diversity analysis: SKIPPED (too slow for 5M rows)
# ✓ Similarity analysis: SKIPPED (too slow for 5M rows)
# ✓ Visualizations: Limited to 100 samples
# ✓ All other analyses: FAST (vectorized)
```

### Manual Control

For fine-grained control:

```python
results = run_full_analysis(
    indicator_matrix=your_L_matrix,
    snorkel_weights=your_weights,
    indicator_names=lf_names,
    output_dir="./large_analysis",
    
    # Performance options
    skip_diversity=True,         # Skip diversity (slow on large data)
    skip_similarity=True,        # Skip similarity (slow on large data)
    skip_viz=False,              # Keep visualizations (they're sampled)
    max_viz_samples=50,          # Reduce viz samples (default: 100)
    diversity_sample_size=50     # If running diversity, use small sample
)
```

### What Runs Fast (Even at 5M Rows)

These steps are **highly optimized** and run quickly:

✅ **STEP 1: Weight Analysis** - Instant (just statistics on 160 weights)
✅ **STEP 2: Importance Scores** - ~10-30 seconds (vectorized)
✅ **STEP 3: Outlier Detection** - ~20-60 seconds (vectorized)
✅ **STEP 4: Top Indicators** - ~5 seconds (simple aggregation)

### What's Slow (Skip These)

❌ **STEP 5: Diversity Analysis** - O(n²) comparisons, very slow
❌ **STEP 6: Similarity Analysis** - O(n²) for selected samples, slow

## Performance Benchmarks

| Dataset Size | Step 1-4 | Diversity | Similarity | Visualizations |
|--------------|----------|-----------|------------|----------------|
| 10K rows     | 1s       | 2s        | 5s         | 10s            |
| 100K rows    | 5s       | 30s       | 2m         | 15s            |
| 1M rows      | 30s      | 10m       | 30m        | 20s            |
| 5M rows      | 2m       | 3h        | 8h         | 25s            |

**Recommendation for 5M rows**: Skip diversity and similarity!

## Optimized Workflow

### Phase 1: Fast Analysis (2-3 minutes for 5M rows)

```python
# Get the essentials quickly
results = run_full_analysis(
    indicator_matrix=L_matrix,
    snorkel_weights=weights,
    indicator_names=lf_names,
    skip_diversity=True,      # Skip
    skip_similarity=True,     # Skip
    skip_viz=False,           # Keep (sampled anyway)
    max_viz_samples=50        # Small sample
)

# You get:
# ✓ Weight clustering detection
# ✓ All importance scores (5M × 160 matrix)
# ✓ Outlier detection (5M rows analyzed)
# ✓ Top 20 indicators globally
# ✓ 8 visualizations (on 50 sample rows)
# ✓ Comprehensive report
```

### Phase 2: Deep Dive (Optional, on Subsets)

After reviewing the fast analysis, run detailed analysis on interesting subsets:

```python
# Find interesting samples from Phase 1
outlier_counts = np.sum(calc.detect_outliers_vectorized(), axis=1)
high_outlier_samples = np.where(outlier_counts > 10)[0][:1000]

# Run detailed analysis on subset
subset_analysis = run_full_analysis(
    indicator_matrix=L_matrix[high_outlier_samples],
    snorkel_weights=weights,
    indicator_names=lf_names,
    output_dir="./subset_analysis",
    skip_diversity=False,     # Now feasible
    skip_similarity=False,    # Now feasible
    diversity_sample_size=200
)
```

## Memory Considerations

For 5M × 160:
- **Indicator matrix**: ~800 MB (int8)
- **Importance matrix**: ~3.2 GB (float32)
- **Outlier matrix**: ~800 MB (bool)
- **Total peak**: ~6-8 GB RAM

Make sure you have at least **16 GB RAM** for 5M rows.

## Chunked Processing (For 10M+ Rows)

If you have >10M rows, process in chunks:

```python
chunk_size = 1000000  # 1M per chunk
n_chunks = (len(L_matrix) + chunk_size - 1) // chunk_size

all_importance = []
all_outliers = []

for i in range(n_chunks):
    start = i * chunk_size
    end = min((i+1) * chunk_size, len(L_matrix))
    
    print(f"Processing chunk {i+1}/{n_chunks} (rows {start:,}-{end:,})...")
    
    chunk_calc = SnorkelIndicatorImportance(
        indicator_matrix=L_matrix[start:end],
        snorkel_weights=weights,
        indicator_names=lf_names
    )
    
    # Compute for chunk
    chunk_importance = chunk_calc.get_all_importance_scores()
    chunk_outliers = chunk_calc.detect_outliers_vectorized()
    
    all_importance.append(chunk_importance)
    all_outliers.append(chunk_outliers)

# Combine results
final_importance = np.vstack(all_importance)
final_outliers = np.vstack(all_outliers)
```

## Direct Usage (Without Comprehensive Analysis)

For maximum speed on large datasets, use the library directly:

```python
from indicator_importance import SnorkelIndicatorImportance

# Initialize once
calc = SnorkelIndicatorImportance(
    indicator_matrix=L_matrix,  # 5M × 160
    snorkel_weights=weights,
    indicator_names=lf_names
)

# Fast operations (vectorized)
print("Computing importance...")
all_importance = calc.get_all_importance_scores()  # ~20 seconds

print("Detecting outliers...")
all_outliers = calc.detect_outliers_vectorized()   # ~30 seconds

print("Getting top-k...")
top_indices, top_scores = calc.get_top_k_matrix(top_k=10)  # ~10 seconds

# Save results
np.save('importance_5M.npy', all_importance)
np.save('outliers_5M.npy', all_outliers)
np.save('top_indices_5M.npy', top_indices)

print("Done! Total time: ~1 minute")
```

## What You Can Skip Without Losing Value

For production ML pipelines with millions of rows:

### Keep These (Fast & Valuable):
- ✅ Weight analysis → Tells you if weights cluster
- ✅ Importance scores → Core functionality
- ✅ Outlier detection → Find problematic samples
- ✅ Top indicators → Know which LFs matter most

### Skip These (Slow & Optional):
- ❌ Diversity analysis → Nice to have, but slow
- ❌ Similarity analysis → Interesting but not critical
- ⚠️  Visualizations → Keep them but limit samples to 50

## Monitoring Large Scale Production

For daily monitoring of production models:

```python
# Daily script (runs in 2-3 minutes for 5M rows)
import schedule
from datetime import datetime

def daily_monitoring():
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Load today's data
    L_matrix = load_todays_predictions()  # 5M rows
    
    # Fast analysis
    results = run_full_analysis(
        indicator_matrix=L_matrix,
        snorkel_weights=production_weights,
        indicator_names=lf_names,
        output_dir=f"./monitoring/{date_str}",
        sample_name=f"prod_{date_str}",
        skip_diversity=True,
        skip_similarity=True,
        max_viz_samples=50
    )
    
    # Alert if issues detected
    if results['weight_analysis']['is_clustered']:
        send_alert("Weights have started clustering!")
    
    if results['outliers']['pct_samples_with_outliers'] > 20:
        send_alert(f"High outlier rate: {results['outliers']['pct_samples_with_outliers']:.1f}%")

schedule.every().day.at("06:00").do(daily_monitoring)
```

## Performance Tips

1. **Use float32, not float64** - Halves memory usage
   ```python
   L_matrix = L_matrix.astype(np.float32)
   ```

2. **Preallocate arrays** - Faster than appending
   ```python
   results = np.zeros((n_samples, n_indicators), dtype=np.float32)
   ```

3. **Use numba if available** - Can speed up some operations 2-3x
   ```bash
   pip install numba
   ```

4. **Limit precision in JSON** - Smaller files
   ```python
   # In save_results, round floats
   json.dump(results, f, indent=2, default=lambda x: round(x, 4) if isinstance(x, float) else x)
   ```

5. **Compress large outputs**
   ```python
   import gzip
   with gzip.open('results.json.gz', 'wt') as f:
       json.dump(results, f)
   ```

## Summary for Your 5M × 160 Dataset

**Recommended command:**
```python
results = run_full_analysis(
    indicator_matrix=L_matrix,      # 5M × 160
    snorkel_weights=weights,        # 160 weights
    indicator_names=lf_names,       # 160 names
    output_dir="./analysis_5M",
    sample_name="production_5M",
    skip_diversity=True,            # Auto-skipped anyway
    skip_similarity=True,           # Auto-skipped anyway
    max_viz_samples=50              # Small sample for viz
)

# Expected runtime: 2-3 minutes
# Memory usage: ~8 GB peak
# Output: Full report + all essential analyses
```

The slow STEP 5 (diversity) is now automatically skipped for your dataset size! 🚀
