"""
Integration example: Using SnorkelIndicatorImportance with Snorkel and XGBoost

This example shows how to:
1. Train a label model in Snorkel
2. Extract the learned weights
3. Use IndicatorImportance to explain which LFs matter for each instance
4. Integrate with XGBoost predictions
"""

import numpy as np
import pandas as pd
from typing import List, Optional

# Note: This example requires snorkel and xgboost installed
# pip install snorkel xgboost

from indicator_importance import SnorkelIndicatorImportance


def extract_snorkel_weights(label_model):
    """
    Extract weights from a trained Snorkel LabelModel.
    
    Parameters:
    -----------
    label_model : snorkel.labeling.LabelModel
        Trained Snorkel label model
    
    Returns:
    --------
    np.ndarray : Weights for each labeling function
    """
    # Snorkel stores weights in the model
    # The exact extraction method depends on your Snorkel version
    # For most versions, you can access them like this:
    
    try:
        # Method 1: Direct access (newer Snorkel versions)
        weights = label_model.get_weights()
    except AttributeError:
        # Method 2: Access from model parameters (older versions)
        weights = label_model.mu.weight.detach().numpy().flatten()
    
    return weights


def get_lf_matrix_from_snorkel(L_matrix):
    """
    Convert Snorkel's L matrix (label matrix) to indicator matrix.
    
    Parameters:
    -----------
    L_matrix : np.ndarray
        Snorkel label matrix of shape (n_samples, n_lfs)
        Values are typically -1 (negative), 0 (abstain), 1 (positive)
    
    Returns:
    --------
    np.ndarray : Processed indicator matrix
    """
    # Option 1: Use raw L matrix
    indicator_matrix = L_matrix.copy()
    
    # Option 2: Binarize to show which LFs voted (abstain=0, voted=1)
    # indicator_matrix = (L_matrix != 0).astype(float)
    
    # Option 3: Create separate indicators for positive/negative votes
    # This doubles the number of indicators but gives more granularity
    # pos_indicators = (L_matrix == 1).astype(float)
    # neg_indicators = (L_matrix == -1).astype(float)
    # indicator_matrix = np.hstack([pos_indicators, neg_indicators])
    
    return indicator_matrix


class SnorkelXGBoostExplainer:
    """
    Combined explainer for Snorkel + XGBoost pipelines.
    
    This class helps understand:
    1. Which labeling functions are important for each instance (Snorkel level)
    2. Which features are important for predictions (XGBoost level)
    """
    
    def __init__(
        self,
        label_model,
        xgboost_model,
        L_matrix: np.ndarray,
        lf_names: List[str],
        feature_matrix: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the explainer.
        
        Parameters:
        -----------
        label_model : snorkel.labeling.LabelModel
            Trained Snorkel label model
        xgboost_model : xgboost.XGBClassifier or xgboost.Booster
            Trained XGBoost model
        L_matrix : np.ndarray
            Snorkel label matrix (n_samples, n_lfs)
        lf_names : list of str
            Names of labeling functions
        feature_matrix : np.ndarray, optional
            Feature matrix used for XGBoost training
        feature_names : list of str, optional
            Names of features in XGBoost model
        """
        self.label_model = label_model
        self.xgboost_model = xgboost_model
        self.L_matrix = L_matrix
        self.lf_names = lf_names
        self.feature_matrix = feature_matrix
        self.feature_names = feature_names
        
        # Extract Snorkel weights
        self.snorkel_weights = extract_snorkel_weights(label_model)
        
        # Convert L matrix to indicator matrix
        self.indicator_matrix = get_lf_matrix_from_snorkel(L_matrix)
        
        # Initialize indicator importance calculator
        self.indicator_importance = SnorkelIndicatorImportance(
            indicator_matrix=self.indicator_matrix,
            snorkel_weights=self.snorkel_weights,
            indicator_names=lf_names,
            outlier_method='iqr',
            outlier_threshold=1.5
        )
    
    def explain_instance(
        self,
        row_idx: int,
        top_k_lfs: int = 10,
        include_xgboost_importance: bool = True,
        top_k_features: int = 10
    ) -> dict:
        """
        Explain predictions for a single instance at both Snorkel and XGBoost levels.
        
        Parameters:
        -----------
        row_idx : int
            Index of the instance to explain
        top_k_lfs : int, default=10
            Number of top labeling functions to return
        include_xgboost_importance : bool, default=True
            Whether to include XGBoost feature importance
        top_k_features : int, default=10
            Number of top XGBoost features to return
        
        Returns:
        --------
        dict : Explanation containing both Snorkel and XGBoost insights
        """
        explanation = {}
        
        # Get Snorkel-level importance
        lf_importance = self.indicator_importance.get_row_importance(
            row_idx, top_k=top_k_lfs
        )
        
        explanation['snorkel'] = {
            'top_lfs': lf_importance.top_indicators,
            'outlier_lfs': lf_importance.outlier_indicators,
            'lf_votes': self.L_matrix[row_idx],
            'predicted_label': self.label_model.predict(
                self.L_matrix[row_idx:row_idx+1]
            )[0] if hasattr(self.label_model, 'predict') else None
        }
        
        # Get XGBoost-level importance if requested
        if include_xgboost_importance and self.feature_matrix is not None:
            explanation['xgboost'] = self._get_xgboost_importance(
                row_idx, top_k_features
            )
        
        return explanation
    
    def _get_xgboost_importance(self, row_idx: int, top_k: int) -> dict:
        """Get XGBoost feature importance for a single instance."""
        import xgboost as xgb
        
        xgb_explanation = {}
        
        # Get prediction
        X_instance = self.feature_matrix[row_idx:row_idx+1]
        
        if hasattr(self.xgboost_model, 'predict_proba'):
            prediction = self.xgboost_model.predict_proba(X_instance)[0]
            xgb_explanation['prediction_proba'] = prediction
        else:
            prediction = self.xgboost_model.predict(X_instance)[0]
            xgb_explanation['prediction'] = prediction
        
        # Get global feature importance
        if hasattr(self.xgboost_model, 'feature_importances_'):
            importance_scores = self.xgboost_model.feature_importances_
        else:
            # For Booster objects
            importance_dict = self.xgboost_model.get_score(importance_type='gain')
            importance_scores = np.array([
                importance_dict.get(f'f{i}', 0) 
                for i in range(self.feature_matrix.shape[1])
            ])
        
        # Combine with instance values to get instance-level importance
        instance_values = self.feature_matrix[row_idx]
        instance_importance = instance_values * importance_scores
        
        # Get top-k features
        top_indices = np.argsort(np.abs(instance_importance))[-top_k:][::-1]
        
        if self.feature_names:
            top_features = [
                (self.feature_names[idx], instance_importance[idx], instance_values[idx])
                for idx in top_indices
            ]
        else:
            top_features = [
                (f'feature_{idx}', instance_importance[idx], instance_values[idx])
                for idx in top_indices
            ]
        
        xgb_explanation['top_features'] = top_features
        
        return xgb_explanation
    
    def create_explanation_report(self, row_idx: int) -> str:
        """
        Create a human-readable explanation report for an instance.
        
        Parameters:
        -----------
        row_idx : int
            Index of the instance to explain
        
        Returns:
        --------
        str : Formatted explanation report
        """
        explanation = self.explain_instance(row_idx)
        
        report = f"=" * 80 + "\n"
        report += f"Explanation for Instance {row_idx}\n"
        report += f"=" * 80 + "\n\n"
        
        # Snorkel section
        report += "WEAK SUPERVISION LAYER (Snorkel)\n"
        report += "-" * 80 + "\n"
        
        if explanation['snorkel']['predicted_label'] is not None:
            report += f"Snorkel Predicted Label: {explanation['snorkel']['predicted_label']}\n\n"
        
        report += "Top Contributing Labeling Functions:\n"
        for i, (lf_name, score) in enumerate(explanation['snorkel']['top_lfs'], 1):
            vote = explanation['snorkel']['lf_votes'][
                self.lf_names.index(lf_name)
            ]
            report += f"  {i}. {lf_name}\n"
            report += f"     Importance Score: {score:.4f}\n"
            report += f"     Vote: {vote}\n"
        
        if explanation['snorkel']['outlier_lfs']:
            report += f"\nOutlier Labeling Functions ({len(explanation['snorkel']['outlier_lfs'])} detected):\n"
            for lf_name, value in explanation['snorkel']['outlier_lfs'][:5]:
                report += f"  - {lf_name}: {value:.4f}\n"
        
        # XGBoost section
        if 'xgboost' in explanation:
            report += "\n" + "=" * 80 + "\n"
            report += "PREDICTION LAYER (XGBoost)\n"
            report += "-" * 80 + "\n"
            
            if 'prediction_proba' in explanation['xgboost']:
                probs = explanation['xgboost']['prediction_proba']
                report += f"Prediction Probabilities: {probs}\n\n"
            
            report += "Top Contributing Features:\n"
            for i, (feat_name, importance, value) in enumerate(
                explanation['xgboost']['top_features'], 1
            ):
                report += f"  {i}. {feat_name}\n"
                report += f"     Importance: {importance:.4f}\n"
                report += f"     Value: {value:.4f}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    def analyze_batch(
        self,
        row_indices: Optional[List[int]] = None,
        top_k_lfs: int = 5
    ) -> pd.DataFrame:
        """
        Analyze a batch of instances and return summary DataFrame.
        
        Parameters:
        -----------
        row_indices : list of int, optional
            Indices to analyze. If None, analyzes first 100 instances.
        top_k_lfs : int, default=5
            Number of top LFs to include per instance
        
        Returns:
        --------
        pd.DataFrame : Summary of explanations
        """
        if row_indices is None:
            row_indices = range(min(len(self.L_matrix), 100))
        
        summaries = []
        for idx in row_indices:
            explanation = self.explain_instance(idx, top_k_lfs=top_k_lfs)
            
            summary = {
                'instance_id': idx,
                'snorkel_label': explanation['snorkel']['predicted_label']
            }
            
            # Add top LFs
            for i, (lf_name, score) in enumerate(
                explanation['snorkel']['top_lfs'], 1
            ):
                summary[f'top_lf_{i}'] = lf_name
                summary[f'top_lf_{i}_score'] = score
            
            # Add XGBoost prediction if available
            if 'xgboost' in explanation:
                if 'prediction_proba' in explanation['xgboost']:
                    probs = explanation['xgboost']['prediction_proba']
                    summary['xgb_pred_class'] = np.argmax(probs)
                    summary['xgb_pred_confidence'] = np.max(probs)
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


def example_complete_workflow():
    """
    Example showing a complete Snorkel + XGBoost workflow with explanations.
    """
    # This is a simplified example - in practice you would:
    # 1. Define your labeling functions
    # 2. Apply them to get L_matrix
    # 3. Train label model
    # 4. Get probabilistic labels
    # 5. Train XGBoost on features + probabilistic labels
    # 6. Use this explainer to understand predictions
    
    np.random.seed(42)
    
    # Simulate data
    n_samples = 500
    n_lfs = 30
    n_features = 20
    
    # Simulate L matrix (labeling function outputs)
    L_matrix = np.random.choice([-1, 0, 1], size=(n_samples, n_lfs))
    lf_names = [f"LF_{i}" for i in range(n_lfs)]
    
    # Simulate feature matrix
    X = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Simulate ground truth labels
    y = np.random.choice([0, 1], size=n_samples)
    
    print("Creating mock Snorkel label model...")
    # In practice, you would train a real label model:
    # from snorkel.labeling import LabelModel
    # label_model = LabelModel(cardinality=2, verbose=True)
    # label_model.fit(L_matrix)
    
    # For this example, we'll create a mock label model
    class MockLabelModel:
        def __init__(self, n_lfs):
            self.weights = np.random.randn(n_lfs) * 0.5
        
        def predict(self, L):
            # Simple weighted voting
            votes = np.dot(L, self.weights)
            return (votes > 0).astype(int)
        
        def get_weights(self):
            return self.weights
    
    label_model = MockLabelModel(n_lfs)
    
    print("Training XGBoost model...")
    # Train XGBoost
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    xgb_model.fit(X, y)
    
    print("\nInitializing explainer...")
    explainer = SnorkelXGBoostExplainer(
        label_model=label_model,
        xgboost_model=xgb_model,
        L_matrix=L_matrix,
        lf_names=lf_names,
        feature_matrix=X,
        feature_names=feature_names
    )
    
    # Explain a single instance
    print("\n" + "=" * 80)
    print("Single Instance Explanation")
    print("=" * 80)
    report = explainer.create_explanation_report(row_idx=0)
    print(report)
    
    # Analyze a batch
    print("\nGenerating batch analysis...")
    batch_df = explainer.analyze_batch(row_indices=range(10), top_k_lfs=3)
    print("\nBatch Analysis Summary:")
    print(batch_df.head())
    
    return explainer


if __name__ == "__main__":
    try:
        explainer = example_complete_workflow()
        print("\n✓ Example completed successfully!")
    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("Install required packages: pip install snorkel xgboost")
