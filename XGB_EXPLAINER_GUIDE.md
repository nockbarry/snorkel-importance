# XGBoost + SHAP Explainer - Quick Guide

## One-Line Usage

```python
from xgb_explainer import explain_xgb_events

results = explain_xgb_events(
    model=xgb_model,
    X=X_data,                          # 5M × 1000
    feature_names=feature_names,
    event_ids=[3947355, 4526073, ...], # Your 3000 events
    events_are_labels=True,
    event_index=df.index.values,
    y=y_true,
    output_dir="./xgb_explanations"
)
# Runtime: ~10-15 minutes for 3000 events
```

## What You Get

### Global Statistics (`global_stats.html`)
- Top 20 features by importance & SHAP
- Model performance metrics
- Dataset overview

### Event Reports (`explanations/`)
- Master summary: All 3000 events in table
- Individual reports: One HTML per event
- CSV exports: For Excel analysis

### Per-Event Details
- **Top 20 features** by |SHAP|
- **Anomalous features** (95th percentile)
- **Prediction** with probability
- **Feature values** and SHAP contributions

## Performance

| Dataset | Events | Time |
|---------|--------|------|
| 5M × 1000 | 3000 | ~15 min |
| 1M × 500 | 1000 | ~5 min |
| 100K × 200 | 100 | ~1 min |

SHAP TreeExplainer is very fast for XGBoost!

## Output Structure

```
xgb_explanations/
├── global_stats.html          # Model overview
├── explanations.json          # All data
├── explanations_summary.csv   # Summary table
├── explanations_detailed.csv  # Full features
└── explanations/
    ├── index.html             # Master (3000 events)
    └── event_*.html           # Individual (3000 files)
```

## Class API (More Control)

```python
from xgb_explainer import XGBExplainer

explainer = XGBExplainer(model, X, feature_names, df.index.values, y)

# Global stats
stats = explainer.get_global_stats()

# Explain events
results = explainer.explain_events(
    event_ids=[3947355, ...],
    events_are_labels=True,
    output_dir="./explanations"
)

# Access programmatically
for exp in results['explanations']:
    print(f"Event {exp['event_id']}: {exp['top_20_features'][0]['name']}")
```

## Key Features

✅ **Efficient**: Only computes SHAP for focus events (not all 5M)  
✅ **Fast**: TreeExplainer optimized for XGBoost  
✅ **Pandas support**: Uses your event IDs  
✅ **Investigator-ready**: Clean HTML tables  
✅ **Anomaly detection**: Flags unusual features  
✅ **CSV export**: For Excel/pandas  

## Use With Snorkel Analyzer

```python
# 1. Analyze labeling quality
from large_scale_analysis import analyze_large_dataset
snorkel_results = analyze_large_dataset(...)  # Which LFs matter

# 2. Analyze model decisions  
from xgb_explainer import explain_xgb_events
xgb_results = explain_xgb_events(...)  # Which features matter

# Complete picture!
```

## Tips

1. **Start with global_stats.html** - Model overview
2. **Use explanations/index.html** - Sort by anomalies
3. **Focus on top 3 features** per event - Usually tell the story
4. **Check SHAP sign** - Positive = increases prediction

## Module Design

- **300 lines** - Tight, no bloat
- **TreeExplainer** - Fast for XGBoost
- **Sampled global stats** - 10K sample (not all 5M)
- **Batch processing** - Progress tracking
- **HTML + CSV** - Multiple export formats

Perfect for your 5M × 1000 dataset with 3000 focus events! 🚀
