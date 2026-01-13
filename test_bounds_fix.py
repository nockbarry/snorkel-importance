"""
Test to reproduce and verify fix for sampling bounds error.

Error: "index 3947355 is out of bounds for axis 0 with size 50000"
Cause: Pandas index values being used as positions in sampling
"""

import numpy as np
from large_scale_analysis import analyze_large_dataset

print("=" * 80)
print("REPRODUCING SAMPLING BOUNDS ERROR")
print("=" * 80)

# Simulate user's scenario
np.random.seed(42)
n_samples = 50000  # User has 50K rows
n_indicators = 160

# Pandas index with large values (like user's 3947355)
# This simulates a DataFrame with non-sequential index
pandas_index = np.array([3000000 + i for i in range(n_samples)])
print(f"\nDataset: {n_samples:,} rows")
print(f"Pandas index range: {pandas_index[0]:,} to {pandas_index[-1]:,}")

# Create data
L_matrix = np.random.choice([-1, 0, 1], size=(n_samples, n_indicators))
weights = np.random.randn(n_indicators) * 0.5
lf_names = [f"LF_{i}" for i in range(n_indicators)]

# Focus on some events (using pandas index values, not positions!)
focus_events = [3000000, 3010000, 3947355, 3948000, 3949000]
print(f"\nFocus events (pandas index values): {focus_events}")

print("\n" + "=" * 80)
print("TEST 1: Without fix (focus_rows_are_labels=False)")
print("=" * 80)
try:
    results = analyze_large_dataset(
        indicator_matrix=L_matrix,
        snorkel_weights=weights,
        indicator_names=lf_names,
        pandas_index=pandas_index,
        focus_rows=focus_events,
        focus_rows_are_labels=False,  # WRONG: treats labels as positions
        create_focus_visualizations=False,
        create_investigator_reports=False,
        viz_sample_size=100,
        output_dir="./test_bounds_error_wrong"
    )
    print("❌ ERROR: Should have failed but didn't!")
except (IndexError, ValueError) as e:
    print(f"✓ Got expected error: {type(e).__name__}")
    print(f"  Message: {str(e)[:100]}")

print("\n" + "=" * 80)
print("TEST 2: With fix (focus_rows_are_labels=True)")
print("=" * 80)
try:
    results = analyze_large_dataset(
        indicator_matrix=L_matrix,
        snorkel_weights=weights,
        indicator_names=lf_names,
        pandas_index=pandas_index,
        focus_rows=focus_events,
        focus_rows_are_labels=True,  # CORRECT: converts labels to positions
        create_focus_visualizations=False,
        create_investigator_reports=False,
        viz_sample_size=100,
        output_dir="./test_bounds_error_fixed"
    )
    print(f"\n✅ SUCCESS!")
    print(f"   Analyzed {len(results['focus_analysis'])} focus rows")
    print(f"   Created {len(results['metadata'])} metadata entries")
    
    # Verify the positions were converted correctly
    for focus in results['focus_analysis']:
        label = focus['row_label']
        idx = focus['row_index']
        print(f"   Event {label}: position {idx} (valid: {0 <= idx < n_samples})")
    
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST 3: Without pandas_index (should error)")
print("=" * 80)
try:
    results = analyze_large_dataset(
        indicator_matrix=L_matrix,
        snorkel_weights=weights,
        indicator_names=lf_names,
        pandas_index=None,  # Missing!
        focus_rows=focus_events,
        focus_rows_are_labels=True,
        create_focus_visualizations=False,
        create_investigator_reports=False,
        viz_sample_size=100,
        output_dir="./test_bounds_error_no_index"
    )
    print("❌ ERROR: Should have failed but didn't!")
except ValueError as e:
    print(f"✓ Got expected error: {type(e).__name__}")
    print(f"  Message: {str(e)[:100]}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
To fix "index 3947355 is out of bounds for axis 0 with size 50000":

✅ DO THIS:
    results = analyze_large_dataset(
        ...,
        pandas_index=df.index.values,      # ← Must provide!
        focus_rows=[3947355, ...],         # ← Your pandas index values
        focus_rows_are_labels=True,        # ← Must set True!
    )

❌ DON'T DO THIS:
    results = analyze_large_dataset(
        ...,
        pandas_index=None,                 # ← Missing!
        focus_rows=[3947355, ...],         # ← Will be treated as positions!
        focus_rows_are_labels=False,       # ← Wrong!
    )

The script now:
1. Validates that pandas_index is provided when needed
2. Converts labels to positions before any sampling
3. Checks bounds in smart_sample
4. Shows clear error messages if misconfigured
""")
