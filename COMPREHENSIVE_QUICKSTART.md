# Comprehensive Analysis - Quick Start

## 🚀 One Command to Run Everything

This is the **all-in-one script** that runs every analysis, generates every visualization, and creates a complete report.

## Basic Usage

```python
from comprehensive_analysis import run_full_analysis

# Run complete analysis with your data
results = run_full_analysis(
    indicator_matrix=your_L_matrix,      # From Snorkel
    snorkel_weights=your_weights,        # From label_model.get_weights()
    indicator_names=your_lf_names,       # List of LF names
    output_dir="./my_analysis",          # Where to save results
    sample_name="my_project"             # Name for output files
)
```

That's it! The script will:
- ✅ Detect if weights are clustered
- ✅ Run all importance calculations
- ✅ Detect outliers across all samples
- ✅ Identify top indicators
- ✅ Analyze instance diversity and similarity
- ✅ Generate 8 comprehensive visualizations
- ✅ Create a detailed text report
- ✅ Save all results to JSON

## What You Get

After running, you'll have a complete directory with:

```
my_analysis/
├── my_project_report.txt                    # Comprehensive text report
├── my_project_results.json                  # All data in JSON format
├── my_project_weight_distribution.png       # Weight histogram
├── my_project_top_indicators.png            # Bar chart of top 10 LFs
├── my_project_outlier_distribution.png      # Outlier counts histogram
├── my_project_pattern_heatmap.png           # Clustered heatmap
├── my_project_pca_projection.png            # 2D PCA of patterns
├── my_project_similarity_matrix.png         # Instance similarity
├── my_project_instance_comparison.png       # Side-by-side comparison
└── my_project_radar_sample_0.png            # Radar chart for sample 0
```

## Complete Example

```python
import numpy as np
from comprehensive_analysis import run_full_analysis

# Your data (from Snorkel)
indicator_matrix = L_matrix  # Shape: (n_samples, n_lfs)
snorkel_weights = label_model.get_weights()
lf_names = ["LF_1", "LF_2", ...]

# Run everything
results = run_full_analysis(
    indicator_matrix=indicator_matrix,
    snorkel_weights=snorkel_weights,
    indicator_names=lf_names,
    output_dir="./snorkel_analysis",
    sample_name="production_run",
    max_viz_samples=100  # Limit visualization samples for performance
)

# Results are also returned as a dictionary
print(f"Top indicator: {results['top_indicators']['global_top_20'][0]['name']}")
print(f"Weights clustered: {results['weight_analysis']['is_clustered']}")
```

## Using with Your Snorkel Pipeline

```python
from snorkel.labeling import LabelModel
from comprehensive_analysis import run_full_analysis

# 1. Train your Snorkel label model (your existing code)
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train)

# 2. Extract what you need
indicator_matrix = L_train  # Your label matrix
snorkel_weights = label_model.get_weights()
lf_names = [lf.name for lf in labeling_functions]  # Your LF names

# 3. Run comprehensive analysis
results = run_full_analysis(
    indicator_matrix=indicator_matrix,
    snorkel_weights=snorkel_weights,
    indicator_names=lf_names,
    output_dir="./analysis_results"
)

# 4. Review the report
with open("./analysis_results/analysis_report.txt", 'r') as f:
    print(f.read())
```

## Advanced: Using the Class Interface

For more control, use the `ComprehensiveAnalysis` class:

```python
from comprehensive_analysis import ComprehensiveAnalysis

# Initialize
analysis = ComprehensiveAnalysis(
    indicator_matrix=indicator_matrix,
    snorkel_weights=snorkel_weights,
    indicator_names=lf_names,
    output_dir="./custom_analysis",
    sample_name="detailed_run"
)

# Run full analysis
results = analysis.run_all(max_viz_samples=200)

# Access intermediate results
print(f"Weight std: {analysis.results['weight_analysis']['std']}")
print(f"Total outliers: {analysis.results['outliers']['total_outliers']}")

# Print the report
analysis.print_report()
```

## What the Script Does Automatically

### 1. Weight Clustering Detection
```
Automatically detects if weights cluster (std < 0.2)
→ Adapts similarity methods accordingly
→ Recommends appropriate analysis techniques
```

### 2. Complete Importance Analysis
```
• Standard importance (indicator × weight)
• Deviation importance (unusual patterns)
• Top-k indicators per instance
• Global importance ranking
```

### 3. Outlier Detection
```
• Identifies outlier indicators for each sample
• Ranks most problematic samples
• Shows which LFs most often behave unusually
```

### 4. Similarity Analysis
```
• Automatically chooses best method (cosine vs combined)
• Analyzes representative samples
• Computes discrimination quality
```

### 5. Comprehensive Visualizations
```
8 different visualizations covering:
• Weight distribution
• Top indicators
• Outlier patterns
• Heatmaps with clustering
• PCA projections
• Similarity matrices
• Instance comparisons
• Radar charts
```

### 6. Detailed Report
```
Text report includes:
• Weight analysis and clustering detection
• Top 10 most important indicators
• Outlier statistics and problematic samples
• Diversity analysis
• Similarity quality metrics
• Actionable recommendations
```

## Performance Notes

- **Small datasets** (<1K samples): Everything runs in seconds
- **Medium datasets** (1K-10K): Analysis in ~30 seconds
- **Large datasets** (>10K): Set `max_viz_samples=100` to limit visualization time

## Customization

```python
# Customize visualization samples
results = run_full_analysis(
    ...,
    max_viz_samples=50  # Fewer samples = faster (default: 100)
)

# Different output location
results = run_full_analysis(
    ...,
    output_dir="./reports/january_2026"
)

# Custom naming
results = run_full_analysis(
    ...,
    sample_name="model_v3_validation"
)
```

## Troubleshooting

**Q: Script is slow**
```python
# Reduce visualization samples
results = run_full_analysis(..., max_viz_samples=50)
```

**Q: Want to skip some visualizations**
```python
# Use the class interface and call methods selectively
analysis = ComprehensiveAnalysis(...)
analysis._analyze_weights()
analysis._compute_importance_scores()
# ... call only what you need
```

**Q: Need results programmatically**
```python
results = run_full_analysis(...)

# Access specific results
top_lf = results['top_indicators']['global_top_20'][0]
is_clustered = results['weight_analysis']['is_clustered']
outlier_samples = results['outliers']['top_outlier_samples']
```

## Output Formats

### JSON Results (`results.json`)
Complete machine-readable results including:
- All statistics and metrics
- Top indicators with scores
- Outlier information
- Similarity analysis results
- File paths to visualizations

### Text Report (`report.txt`)
Human-readable summary with:
- Executive summary
- Key findings
- Statistical tables
- Actionable recommendations

### Visualizations (8 PNG files)
Ready-to-include in presentations or papers

## Integration Examples

### Use in Jupyter Notebook
```python
from comprehensive_analysis import run_full_analysis
from IPython.display import Image, display

# Run analysis
results = run_full_analysis(...)

# Display visualizations inline
for viz_path in results['visualizations']:
    display(Image(filename=viz_path))
```

### Schedule Regular Analysis
```python
import schedule
import time

def daily_analysis():
    # Load latest data
    L_matrix = load_latest_data()
    weights = load_latest_weights()
    
    # Run analysis
    run_full_analysis(
        indicator_matrix=L_matrix,
        snorkel_weights=weights,
        indicator_names=lf_names,
        output_dir=f"./daily_reports/{datetime.now().strftime('%Y-%m-%d')}"
    )

# Schedule daily at 6 AM
schedule.every().day.at("06:00").do(daily_analysis)
```

### Export to Dashboard
```python
import json

results = run_full_analysis(...)

# Extract key metrics
dashboard_data = {
    'date': datetime.now().isoformat(),
    'n_samples': results['metadata']['n_samples'],
    'top_indicator': results['top_indicators']['global_top_20'][0]['name'],
    'weights_clustered': results['weight_analysis']['is_clustered'],
    'outlier_rate': results['outliers']['pct_samples_with_outliers'],
    'visualizations': results['visualizations']
}

# Send to dashboard API
# requests.post('https://dashboard.com/api/metrics', json=dashboard_data)
```

## Best Practices

1. **Run after model training**: Analyze immediately after training your Snorkel model
2. **Version your analyses**: Use descriptive `sample_name` like "model_v2_20260113"
3. **Review the report**: Always read the generated report for recommendations
4. **Share visualizations**: The PNGs are presentation-ready
5. **Track over time**: Save results to compare across model versions

## Example Output Snippet

```
================================================================================
COMPREHENSIVE INDICATOR IMPORTANCE ANALYSIS REPORT
================================================================================
Generated: 2026-01-13T10:30:00
Sample: production_model_v3
Dataset: 5,000 samples × 120 indicators

================================================================================
1. WEIGHT ANALYSIS
================================================================================
Mean:   0.512
Std:    0.189
Median: 0.543
Range:  [-0.342, 2.145]

⚠️  WEIGHT CLUSTERING DETECTED
   Severity: MEDIUM
   Recommendation: Use pattern-based or combined similarity methods

================================================================================
2. TOP 10 MOST IMPORTANT INDICATORS
================================================================================
Rank   Name                                Avg Importance   Weight    
----------------------------------------------------------------------
1      LF_accuracy_keyword_match                   0.0892     2.145
2      LF_regex_email_pattern                      0.0654     1.234
...
```

## Next Steps

After running the comprehensive analysis:

1. **Read the report** (`_report.txt`) for insights
2. **Review visualizations** to understand patterns
3. **Check JSON results** for programmatic access
4. **Investigate outliers** shown in the analysis
5. **Iterate on your LFs** based on findings

That's it! One script does everything. 🎉
