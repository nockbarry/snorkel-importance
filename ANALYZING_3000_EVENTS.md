# Analyzing 3000 Events for Investigators

## Your Exact Use Case

You have **~3000 events** (rows with pandas index like `[3742123, 4526073, ...]`) that need detailed analysis and investigator-ready reports.

## The Fix: Pandas Index Labels

The issue was that you were passing **pandas index values** but the code expected **positional indices**. Now fixed!

## Quick Start

```python
from large_scale_analysis import analyze_large_dataset
import pandas as pd

# Your data
df = pd.read_csv('events.csv')  # Has index with values like 3742123, 4526073
L_matrix = df[indicator_cols].values
weights = label_model.get_weights()

# Your 3000 events (pandas index values, not positions!)
focus_events = [3742123, 4526073, 5123456, ...]  # ~3000 values

# Run analysis - IMPORTANT: focus_rows_are_labels=True
results = analyze_large_dataset(
    indicator_matrix=L_matrix,
    snorkel_weights=weights,
    indicator_names=lf_names,
    pandas_index=df.index.values,      # ← Pass your index
    focus_rows=focus_events,           # ← Your event IDs
    focus_rows_are_labels=True,        # ← KEY: These are pandas index values!
    create_focus_visualizations=False, # ← See note below
    create_investigator_reports=True,  # ← Creates HTML reports
    output_dir="./investigation_3000"
)
```

## Key Parameters

### `focus_rows_are_labels=True` (IMPORTANT!)

```python
# If True (default): focus_rows are pandas index values
focus_rows=[3742123, 4526073]  # Your actual event IDs

# If False: focus_rows are positional indices
focus_rows=[0, 1, 2]  # Positions in the matrix
```

### `create_focus_visualizations`

For **3000 events**, you have choices:

**Option 1: Skip visualizations (FAST)**
```python
create_focus_visualizations=False  # No images, just reports
# Runtime: ~5-10 minutes for 3000 events
# Output: HTML reports + tables only
```

**Option 2: Create visualizations (THOROUGH but SLOW)**
```python
create_focus_visualizations=True  # 4 images per event = 12,000 images!
# Runtime: ~2-3 hours for 3000 events
# Output: HTML reports + tables + all visualizations
# Disk space: ~500 MB
```

**Option 3: Selective (RECOMMENDED)**
```python
# Analyze all 3000 first (without viz)
results1 = analyze_large_dataset(..., create_focus_visualizations=False)

# Then create viz for top 100 high-priority events
high_priority = identify_top_100(results1)
results2 = analyze_large_dataset(..., focus_rows=high_priority, create_focus_visualizations=True)
```

## What You Get

### Master Summary (`investigator_reports/index.html`)

Interactive HTML table with ALL 3000 events:
- Event ID (clickable links)
- Outlier count (color-coded by severity)
- Top 3 indicators
- Sortable columns
- Statistics dashboard

### Individual Event Reports (`investigator_reports/event_XXX.html`)

For EACH of the 3000 events:
- **Outlier Alert** - Unusual indicators highlighted
- **Top 20 Indicators Table** - Most important for this event
- **Deviation Table** - What's unusual about this event
- **Similar Events Table** - 10 most similar with links
- **Visualizations** (if enabled) - 4 detailed charts

### CSV Exports

- `events_summary.csv` - One row per event with top indicators
- `events_detailed.csv` - All indicators for all events

## Complete Example

```python
from large_scale_analysis import analyze_large_dataset
import pandas as pd

# Load your data
print("Loading data...")
df = pd.read_csv('all_events.csv', index_col=0)  # index_col=0 if first column is index
print(f"Loaded {len(df):,} total events")

# Get your Snorkel data
L_matrix = df[[c for c in df.columns if c.startswith('LF_')]].values
lf_names = [c for c in df.columns if c.startswith('LF_')]

# Load weights
weights = label_model.get_weights()  # or however you have them

# Your 3000 events of interest (pandas index values)
focus_events = df.loc[df['needs_investigation'] == True].index.tolist()
print(f"Focus on {len(focus_events)} events for investigation")

# Run analysis
print("\nRunning analysis...")
results = analyze_large_dataset(
    indicator_matrix=L_matrix,
    snorkel_weights=weights,
    indicator_names=lf_names,
    pandas_index=df.index.values,        # Your actual index
    focus_rows=focus_events,             # Your 3000 event IDs
    focus_rows_are_labels=True,          # These are pandas index values
    create_focus_visualizations=False,   # Skip viz for speed
    create_investigator_reports=True,    # Create HTML reports
    viz_sample_size=10000,               # For global viz
    sampling_method='clustering',
    output_dir="./investigation_reports"
)

print("\n" + "=" * 80)
print("✅ COMPLETE!")
print("=" * 80)
print("\nInvestigator Reports:")
print("  📊 Master summary: investigation_reports/investigator_reports/index.html")
print(f"  📄 {len(results['focus_analysis'])} event reports created")
print("  📊 CSV exports: events_summary.csv, events_detailed.csv")
```

## Output Structure

```
investigation_reports/
├── investigator_reports/
│   ├── index.html           # ← START HERE: Master summary of all 3000
│   ├── event_3742123.html   # ← Individual reports (3000 files)
│   ├── event_4526073.html
│   ├── ... (3000 total)
│   ├── events_summary.csv   # ← For Excel/analysis
│   └── events_detailed.csv  # ← Full data export
├── focus_rows/              # ← Only if visualizations enabled
│   ├── 3742123_top20.png    # (12,000 images if all enabled)
│   ├── 3742123_radar.png
│   └── ...
├── global_heatmap.png       # ← Global analysis visualizations
├── global_pca.png
├── results.json             # ← All data in JSON
└── report.txt               # ← Summary report
```

## Performance Expectations

### For 3000 Events (5M total rows)

| Configuration | Runtime | Output |
|--------------|---------|--------|
| Reports only (no viz) | ~10 min | 3000 HTML + CSV |
| Reports + viz | ~2-3 hours | + 12,000 images |
| Selective (100 viz) | ~15 min | 3000 HTML + 400 images |

### Memory

- 5M × 160 dataset: ~8 GB RAM
- Recommend: 16 GB+ RAM

## Investigator Workflow

1. **Open master summary** (`investigator_reports/index.html`)
   - See all 3000 events sorted by outlier count
   - Statistics dashboard shows overall patterns
   
2. **Click on high-priority events**
   - Events with most outliers appear first
   - Each link opens detailed event report

3. **Review individual event**
   - See which indicators fired unusually
   - Check top 20 most important indicators
   - Find similar past events

4. **Export for further analysis**
   - Open `events_summary.csv` in Excel
   - Filter, sort, pivot as needed
   - `events_detailed.csv` has all indicators

## Handling "Out of Bounds" Error

If you see "out of bounds" warnings:

```python
# ❌ WRONG - treating pandas index as positions
focus_rows=[3742123, 4526073]
focus_rows_are_labels=False  # Tries to access position 3742123!

# ✅ CORRECT - using pandas index values
focus_rows=[3742123, 4526073]
focus_rows_are_labels=True   # Converts 3742123 to correct position
pandas_index=df.index.values # Needed for conversion!
```

The script will show warnings for any labels not found:
```
⚠️  Warning: 5 labels not found in index:
    3742123
    4526073
    ...
```

## Selective Visualization Strategy

For 3000 events, create viz only for high-priority:

```python
# Step 1: Analyze all 3000 (no viz)
results_all = analyze_large_dataset(
    ...,
    focus_rows=all_3000_events,
    create_focus_visualizations=False
)

# Step 2: Identify top 100 by outlier count
outlier_counts = [(f['row_label'], f['n_outliers']) for f in results_all['focus_analysis']]
top_100 = [label for label, _ in sorted(outlier_counts, key=lambda x: x[1], reverse=True)[:100]]

# Step 3: Create viz for top 100 only
results_top = analyze_large_dataset(
    ...,
    focus_rows=top_100,
    focus_rows_are_labels=True,
    create_focus_visualizations=True,
    output_dir="./top_100_detailed"
)
```

## Tips for Investigators

1. **Start with master summary** - Gives overview of all events
2. **Sort by outliers** - Most unusual events first
3. **Look for patterns** - Similar events may have common issues
4. **Use CSV exports** - For quantitative analysis in Excel
5. **Bookmark events** - Save interesting event URLs
6. **Compare similar events** - Use "Similar Events" table

## Customization

### Add Custom Metrics

```python
# After getting results, enhance reports
for event in results['focus_analysis']:
    event_id = event['row_label']
    
    # Add your custom data
    event['custom_risk_score'] = calculate_risk(event_id)
    event['investigator_notes'] = get_notes(event_id)
    
    # Regenerate report with custom data
    # (or modify HTML template)
```

### Filter Events by Criteria

```python
# Only high-outlier events
high_outlier = [e for e in results['focus_analysis'] if e['n_outliers'] > 10]

# Only events with specific indicator
problematic = [e for e in results['focus_analysis'] 
               if any(ind['name'] == 'LF_suspicious' for ind in e['top_20_indicators'])]
```

## Summary

**Key changes for your use case:**
1. ✅ Pass `focus_rows_are_labels=True` 
2. ✅ Provide `pandas_index=df.index.values`
3. ✅ Use your actual pandas index values in `focus_rows`
4. ✅ Set `create_focus_visualizations=False` for speed (or selective)
5. ✅ Get investigator-ready HTML reports + CSV exports

**Expected output:**
- Master summary HTML with all 3000 events
- 3000 individual event reports
- 2 CSV files for analysis
- Runtime: ~10 minutes without viz, ~2-3 hours with viz

Your investigators can now review all 3000 events through interactive HTML reports! 📊
