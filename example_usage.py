"""
Usage Example for XGBExplainer
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from model_explainer import XGBExplainer, quick_explain

# Example: Load your data
# X = pd.read_csv('features.csv')
# y = pd.read_csv('labels.csv')

# For demonstration, create sample data
np.random.seed(42)
n_samples, n_features = 5000, 1000
X = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f'feature_{i}' for i in range(n_features)]
)
y = (X.iloc[:, :10].sum(axis=1) + np.random.randn(n_samples) * 0.5 > 0).astype(int)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X, y)

# =============================================================================
# Method 1: Use quick_explain for one-shot analysis
# =============================================================================
event_ids_to_explain = [0, 100, 500, 1000, 2000]

report = quick_explain(
    model=model,
    X=X,
    event_ids=event_ids_to_explain,
    top_k_global=20,
    top_k_events=10
)

print("=" * 80)
print("MODEL SUMMARY")
print("=" * 80)
for key, value in report['model_summary'].items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

print("\n" + "=" * 80)
print("TOP GLOBAL FEATURES")
print("=" * 80)
print(report['global_importance'].to_string(index=False))

print("\n" + "=" * 80)
print("EVENT-LEVEL EXPLANATIONS")
print("=" * 80)
print(report['event_explanations'].to_string(index=False))

# =============================================================================
# Method 2: Use XGBExplainer class for more control
# =============================================================================
explainer = XGBExplainer(model, X)

# Get global importance
global_imp = explainer.global_feature_importance(top_k=15)
print("\n" + "=" * 80)
print("GLOBAL IMPORTANCE (Top 15)")
print("=" * 80)
print(global_imp)

# Explain specific events
events_df = explainer.explain_events(event_ids=[42, 123, 456], top_k=8)
print("\n" + "=" * 80)
print("SPECIFIC EVENT EXPLANATIONS")
print("=" * 80)
for event_id in [42, 123, 456]:
    print(f"\nEvent {event_id}:")
    event_data = events_df[events_df['event_id'] == event_id]
    print(event_data[['rank', 'feature', 'shap_value', 'feature_value']].to_string(index=False))

# Get model summary statistics
summary = explainer.model_summary()
print("\n" + "=" * 80)
print("MODEL STATISTICS")
print("=" * 80)
for k, v in summary.items():
    print(f"{k}: {v}")
