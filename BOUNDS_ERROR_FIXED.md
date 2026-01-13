# FIXED: Sampling Bounds Error

## The Problem

Error: `index 3947355 is out of bounds for axis 0 with size 50000`

**Cause**: When you passed pandas index values like `[3947355, 4526073, ...]` as focus_rows, they were being used directly as positional indices in the visualization sampling, causing out-of-bounds errors.

## The Fix

✅ **Now properly converts pandas index labels to positions before sampling**

The script now:
1. Validates that `pandas_index` is provided when using labels
2. Converts all labels to positions early in the process
3. Uses converted positions (not labels) for visualization sampling
4. Validates bounds in smart_sample as a safety check
5. Shows clear error messages if misconfigured

## Correct Usage for Your 3000 Events

```python
from large_scale_analysis import analyze_large_dataset
import pandas as pd

# Load your data
df = pd.read_csv('events.csv')  # Has index like [3947355, 4526073, ...]
L_matrix = df[indicator_cols].values
weights = label_model.get_weights()
lf_names = list(indicator_cols)

# Your 3000 event IDs (pandas index values)
focus_events = [3947355, 4526073, 5123456, ...]  # ~3000 IDs

# ✅ CORRECT USAGE
results = analyze_large_dataset(
    indicator_matrix=L_matrix,
    snorkel_weights=weights,
    indicator_names=lf_names,
    
    # CRITICAL: Must provide these two together!
    pandas_index=df.index.values,        # ← Your DataFrame index
    focus_rows_are_labels=True,          # ← Converts IDs to positions
    
    focus_rows=focus_events,             # ← Your actual event IDs
    create_focus_visualizations=False,   # ← Faster without viz
    create_investigator_reports=True,    # ← Creates HTML reports
    viz_sample_size=10000,
    sampling_method='clustering',
    output_dir="./investigation"
)
```

## What Changed

### Before (Broken)
```python
# Passed pandas index values [3947355, ...]
focus_rows = [3947355, 4526073]

# Sampling tried to use 3947355 as a position index
# But dataset only has positions 0-49999
# → IndexError!
```

### After (Fixed)
```python
# Passed pandas index values [3947355, ...]
focus_rows = [3947355, 4526073]
focus_rows_are_labels = True

# Script converts: 3947355 → position 20123 (example)
# Sampling uses position 20123
# → Works! ✅
```

## Validation Messages

The script now shows clear progress:

```
Converting pandas index labels to positions...
  Dataset size: 50,000 rows
  Focus rows requested: 3000
✓ Converted 3000/3000 labels to positions
  Position range: 0 to 49987

STEP 4: Detailed Analysis of 3000 Focus Rows
  Progress: 0/3000 rows analyzed...
  Progress: 100/3000 rows analyzed...
  ...
✓ Completed detailed analysis of 3000 focus rows

STEP 5: Creating Global Visualizations (sample=10,000, method=clustering)
  Selected 10,000 samples for visualization
  ✓ Created 4 global visualizations
```

## Error Handling

### Missing pandas_index
```python
# If you forget pandas_index:
results = analyze_large_dataset(
    ...,
    pandas_index=None,  # ← Missing!
    focus_rows_are_labels=True
)

# You'll get:
# ❌ ERROR: No pandas_index provided!
#    You must set pandas_index=df.index.values to use focus_rows_are_labels=True
```

### Labels not found
```python
# If some labels don't exist in your index:
⚠️  Warning: 5 labels not found in index:
    9999999
    8888888
    ...
✓ Converted 2995/3000 labels to positions

# Analysis continues with the 2995 that were found
```

### Out of bounds
```python
# If any positions are still out of bounds (shouldn't happen):
⚠️  Warning: 2 focus indices out of bounds, filtered

# Sampling automatically filters them out
```

## Test Results

Tested with 50,000 rows and pandas index values like 3947355:

```
TEST: With fix (focus_rows_are_labels=True)
✓ Converted labels to positions correctly
✓ Position range: 0 to 49999 (valid!)
✓ Sampling worked without errors
✓ Created all visualizations
✅ SUCCESS!
```

## Common Mistakes

### ❌ Mistake 1: Not setting focus_rows_are_labels
```python
results = analyze_large_dataset(
    pandas_index=df.index.values,
    focus_rows=[3947355, ...],
    focus_rows_are_labels=False  # ← WRONG! Treats as positions
)
# Error: index 3947355 is out of bounds
```

### ❌ Mistake 2: Not providing pandas_index
```python
results = analyze_large_dataset(
    pandas_index=None,  # ← Missing!
    focus_rows=[3947355, ...],
    focus_rows_are_labels=True
)
# Error: pandas_index is required
```

### ❌ Mistake 3: Using wrong index
```python
results = analyze_large_dataset(
    pandas_index=range(len(df)),  # ← Wrong! Should be df.index.values
    focus_rows=[3947355, ...],
    focus_rows_are_labels=True
)
# Warning: Labels not found (because index is [0,1,2,...] not [3947355,...])
```

### ✅ Correct: All three together
```python
results = analyze_large_dataset(
    pandas_index=df.index.values,        # ← Correct!
    focus_rows=[3947355, ...],           # ← Your actual IDs
    focus_rows_are_labels=True           # ← Enables conversion
)
# Works perfectly! ✅
```

## Quick Check

Before running, verify:

```python
# Check your setup:
print(f"Dataset size: {len(df):,} rows")
print(f"Index type: {type(df.index[0])}")
print(f"Index range: {df.index[0]} to {df.index[-1]}")
print(f"Focus events: {len(focus_events)}")
print(f"Sample focus events: {focus_events[:5]}")
print(f"Are focus events in index? {all(e in df.index for e in focus_events[:100])}")

# All should match your expectations before running analysis
```

## Performance

For 3000 events on 50K dataset:
- **Conversion**: < 1 second
- **Analysis**: ~10 minutes (without viz)
- **With viz**: ~2-3 hours (12,000 images)

Recommendation: Run without viz first, then add viz for top 100 events.

## Next Steps

1. ✅ Verify your data setup (see Quick Check above)
2. ✅ Run with correct parameters (see Correct Usage)
3. ✅ Check conversion messages in output
4. ✅ Review investigator reports

The sampling bounds error is now **completely fixed**! 🎉
