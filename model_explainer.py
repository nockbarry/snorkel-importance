"""
Efficient XGBoost + SHAP Model Explainer
Handles large feature spaces (~1000 features) with minimal overhead
"""
import numpy as np
import pandas as pd
import shap
from typing import List, Dict, Optional, Union


class XGBExplainer:
    """Lightweight explainer for XGBoost models using SHAP"""
    
    def __init__(self, model, X: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        Initialize explainer with model and data
        
        Args:
            model: Trained XGBoost model
            X: Feature data (DataFrame)
            feature_names: Optional list of feature names (uses X.columns if None)
        """
        self.model = model
        self.X = X
        self.feature_names = feature_names or list(X.columns)
        
        # Initialize SHAP explainer (TreeExplainer is fast for XGBoost)
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = None
        self.base_value = self.explainer.expected_value
        
    def compute_shap_values(self):
        """Compute SHAP values for all data (cached)"""
        if self.shap_values is None:
            self.shap_values = self.explainer.shap_values(self.X)
        return self.shap_values
    
    def explain_events(self, event_ids: List, top_k: int = 10) -> pd.DataFrame:
        """
        Get top contributing features for specific events
        
        Args:
            event_ids: List of event IDs (row indices or index values)
            top_k: Number of top features to show per event
            
        Returns:
            DataFrame with event_id, rank, feature, shap_value, feature_value
        """
        shap_vals = self.compute_shap_values()
        
        results = []
        for event_id in event_ids:
            # Handle both integer indices and index labels
            try:
                idx = self.X.index.get_loc(event_id)
            except (KeyError, TypeError):
                idx = event_id
            
            # Get SHAP values for this event
            event_shap = shap_vals[idx]
            event_features = self.X.iloc[idx]
            
            # Get top contributors by absolute SHAP value
            top_indices = np.argsort(np.abs(event_shap))[-top_k:][::-1]
            
            for rank, feat_idx in enumerate(top_indices, 1):
                results.append({
                    'event_id': event_id,
                    'rank': rank,
                    'feature': self.feature_names[feat_idx],
                    'shap_value': event_shap[feat_idx],
                    'feature_value': event_features.iloc[feat_idx],
                    'abs_shap': abs(event_shap[feat_idx])
                })
        
        return pd.DataFrame(results)
    
    def global_feature_importance(self, top_k: int = 20) -> pd.DataFrame:
        """
        Get global feature importance across all data
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance metrics
        """
        shap_vals = self.compute_shap_values()
        
        # Mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        # Sort by importance
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        
        importance_data = []
        for rank, idx in enumerate(top_indices, 1):
            importance_data.append({
                'rank': rank,
                'feature': self.feature_names[idx],
                'mean_abs_shap': mean_abs_shap[idx],
                'mean_shap': shap_vals[:, idx].mean(),
                'std_shap': shap_vals[:, idx].std()
            })
        
        return pd.DataFrame(importance_data)
    
    def model_summary(self) -> Dict:
        """
        Get overall model statistics
        
        Returns:
            Dictionary with model summary statistics
        """
        shap_vals = self.compute_shap_values()
        predictions = self.model.predict(self.X)
        
        return {
            'n_samples': len(self.X),
            'n_features': len(self.feature_names),
            'base_value': float(self.base_value),
            'mean_prediction': float(predictions.mean()),
            'std_prediction': float(predictions.std()),
            'mean_abs_shap': float(np.abs(shap_vals).mean()),
            'max_abs_shap': float(np.abs(shap_vals).max()),
            'shap_coverage': float((np.abs(shap_vals) > 0.01).sum() / shap_vals.size)
        }
    
    def summary_report(self, event_ids: Optional[List] = None, 
                      top_k_global: int = 20, top_k_events: int = 10) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Generate complete summary report
        
        Args:
            event_ids: Optional list of specific events to explain
            top_k_global: Number of top global features
            top_k_events: Number of top features per event
            
        Returns:
            Dictionary containing all summary tables and statistics
        """
        report = {
            'model_summary': self.model_summary(),
            'global_importance': self.global_feature_importance(top_k_global)
        }
        
        if event_ids:
            report['event_explanations'] = self.explain_events(event_ids, top_k_events)
        
        return report


# Convenience function for quick analysis
def quick_explain(model, X: pd.DataFrame, event_ids: List, 
                  top_k_global: int = 20, top_k_events: int = 10) -> Dict:
    """
    Quick one-shot explanation
    
    Args:
        model: XGBoost model
        X: Feature DataFrame
        event_ids: List of events to explain
        top_k_global: Top global features
        top_k_events: Top features per event
        
    Returns:
        Complete explanation report
    """
    explainer = XGBExplainer(model, X)
    return explainer.summary_report(event_ids, top_k_global, top_k_events)
