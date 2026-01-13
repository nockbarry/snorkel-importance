import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class IndicatorImportance:
    """Container for indicator importance results for a single row."""
    row_index: int
    top_indicators: List[Tuple[str, float]]  # (indicator_name, importance_score)
    outlier_indicators: List[Tuple[str, float]]  # (indicator_name, value)
    all_scores: Dict[str, float]  # All importance scores for this row


class SnorkelIndicatorImportance:
    """
    Computes SHAP-like importance values for Snorkel labeling function indicators.
    
    This class helps identify which weak supervision indicators are most influential
    for each data point, combining:
    1. The indicator values (how strongly each LF fires)
    2. The learned Snorkel weights (global importance of each LF)
    3. Statistical outlier detection for unusual indicator patterns
    """
    
    def __init__(
        self,
        indicator_matrix: Union[np.ndarray, pd.DataFrame],
        snorkel_weights: Union[np.ndarray, Dict[str, float], pd.Series],
        indicator_names: Optional[List[str]] = None,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5
    ):
        """
        Initialize the importance calculator.
        
        Parameters:
        -----------
        indicator_matrix : np.ndarray or pd.DataFrame
            Matrix of shape (n_samples, n_indicators) containing indicator/LF outputs
        snorkel_weights : np.ndarray, dict, or pd.Series
            Learned weights from Snorkel for each labeling function
        indicator_names : list of str, optional
            Names of indicators. If None and indicator_matrix is DataFrame, uses column names
        outlier_method : str, default='iqr'
            Method for detecting outliers: 'iqr', 'zscore', or 'percentile'
        outlier_threshold : float, default=1.5
            Threshold for outlier detection (IQR multiplier, z-score, or percentile)
        """
        # Convert indicator matrix to numpy array
        if isinstance(indicator_matrix, pd.DataFrame):
            if indicator_names is None:
                indicator_names = list(indicator_matrix.columns)
            self.indicator_matrix = indicator_matrix.values
        else:
            self.indicator_matrix = np.array(indicator_matrix)
        
        # Set indicator names
        if indicator_names is None:
            indicator_names = [f"indicator_{i}" for i in range(self.indicator_matrix.shape[1])]
        self.indicator_names = indicator_names
        
        # Convert weights to numpy array
        if isinstance(snorkel_weights, dict):
            self.weights = np.array([snorkel_weights[name] for name in indicator_names])
        elif isinstance(snorkel_weights, pd.Series):
            self.weights = snorkel_weights.values
        else:
            self.weights = np.array(snorkel_weights)
        
        # Validate dimensions
        assert self.indicator_matrix.shape[1] == len(self.weights), \
            "Number of indicators must match number of weights"
        assert len(self.indicator_names) == len(self.weights), \
            "Number of indicator names must match number of weights"
        
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        # Compute statistics for outlier detection
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute statistics needed for outlier detection."""
        self.means = np.mean(self.indicator_matrix, axis=0)
        self.stds = np.std(self.indicator_matrix, axis=0)
        self.medians = np.median(self.indicator_matrix, axis=0)
        
        # IQR statistics
        self.q25 = np.percentile(self.indicator_matrix, 25, axis=0)
        self.q75 = np.percentile(self.indicator_matrix, 75, axis=0)
        self.iqr = self.q75 - self.q25
    
    def compute_importance_scores(
        self,
        row_idx: int,
        normalize: bool = True,
        consider_sign: bool = True
    ) -> np.ndarray:
        """
        Compute importance scores for a single row.
        
        The importance score combines:
        - The magnitude of the indicator value (how strongly it fires)
        - The Snorkel weight (how important this LF is globally)
        - Optional: The sign/direction of the indicator
        
        Parameters:
        -----------
        row_idx : int
            Index of the row to compute importance for
        normalize : bool, default=True
            Whether to normalize scores to sum to 1
        consider_sign : bool, default=True
            Whether to preserve the sign of indicator * weight product
        
        Returns:
        --------
        np.ndarray : Importance scores for each indicator
        """
        row_values = self.indicator_matrix[row_idx]
        
        # Compute base importance: indicator_value * weight
        importance = row_values * self.weights
        
        if not consider_sign:
            importance = np.abs(importance)
        
        # Normalize if requested
        if normalize and np.sum(np.abs(importance)) > 0:
            importance = importance / np.sum(np.abs(importance))
        
        return importance
    
    def detect_outliers(self, row_idx: int) -> np.ndarray:
        """
        Detect which indicators are outliers for this row.
        
        Parameters:
        -----------
        row_idx : int
            Index of the row to check
        
        Returns:
        --------
        np.ndarray : Boolean array indicating which indicators are outliers
        """
        row_values = self.indicator_matrix[row_idx]
        
        if self.outlier_method == 'iqr':
            # IQR method: values outside [Q1 - threshold*IQR, Q3 + threshold*IQR]
            lower_bound = self.q25 - self.outlier_threshold * self.iqr
            upper_bound = self.q75 + self.outlier_threshold * self.iqr
            outliers = (row_values < lower_bound) | (row_values > upper_bound)
        
        elif self.outlier_method == 'zscore':
            # Z-score method: |z-score| > threshold
            with np.errstate(divide='ignore', invalid='ignore'):
                z_scores = np.abs((row_values - self.means) / self.stds)
                z_scores = np.nan_to_num(z_scores, nan=0.0)
            outliers = z_scores > self.outlier_threshold
        
        elif self.outlier_method == 'percentile':
            # Percentile method: values in top/bottom threshold percentile
            lower_threshold = np.percentile(self.indicator_matrix, self.outlier_threshold, axis=0)
            upper_threshold = np.percentile(self.indicator_matrix, 100 - self.outlier_threshold, axis=0)
            outliers = (row_values < lower_threshold) | (row_values > upper_threshold)
        
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")
        
        return outliers
    
    def get_row_importance(
        self,
        row_idx: int,
        top_k: int = 10,
        normalize: bool = True,
        include_outliers: bool = True
    ) -> IndicatorImportance:
        """
        Get comprehensive importance analysis for a single row.
        
        Parameters:
        -----------
        row_idx : int
            Index of the row to analyze
        top_k : int, default=10
            Number of top indicators to return
        normalize : bool, default=True
            Whether to normalize importance scores
        include_outliers : bool, default=True
            Whether to detect and include outlier indicators
        
        Returns:
        --------
        IndicatorImportance : Object containing all importance information
        """
        # Compute importance scores
        importance_scores = self.compute_importance_scores(
            row_idx, normalize=normalize, consider_sign=True
        )
        
        # Get top-k indicators by absolute importance
        top_indices = np.argsort(np.abs(importance_scores))[-top_k:][::-1]
        top_indicators = [
            (self.indicator_names[idx], importance_scores[idx])
            for idx in top_indices
        ]
        
        # Detect outliers if requested
        outlier_indicators = []
        if include_outliers:
            outliers = self.detect_outliers(row_idx)
            outlier_indices = np.where(outliers)[0]
            outlier_indicators = [
                (self.indicator_names[idx], self.indicator_matrix[row_idx, idx])
                for idx in outlier_indices
            ]
        
        # Create dictionary of all scores
        all_scores = {
            name: score
            for name, score in zip(self.indicator_names, importance_scores)
        }
        
        return IndicatorImportance(
            row_index=row_idx,
            top_indicators=top_indicators,
            outlier_indicators=outlier_indicators,
            all_scores=all_scores
        )
    
    def explain_predictions(
        self,
        row_indices: Optional[List[int]] = None,
        top_k: int = 10,
        normalize: bool = True
    ) -> List[IndicatorImportance]:
        """
        Get importance explanations for multiple rows.
        
        Parameters:
        -----------
        row_indices : list of int, optional
            Indices to explain. If None, explains all rows.
        top_k : int, default=10
            Number of top indicators to return per row
        normalize : bool, default=True
            Whether to normalize importance scores
        
        Returns:
        --------
        list of IndicatorImportance : Explanations for each row
        """
        if row_indices is None:
            row_indices = range(self.indicator_matrix.shape[0])
        
        results = []
        for idx in row_indices:
            result = self.get_row_importance(idx, top_k=top_k, normalize=normalize)
            results.append(result)
        
        return results
    
    def to_dataframe(
        self,
        row_indices: Optional[List[int]] = None,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Create a DataFrame summary of indicator importance for multiple rows.
        
        Parameters:
        -----------
        row_indices : list of int, optional
            Indices to include. If None, includes all rows.
        top_k : int, default=5
            Number of top indicators to include per row
        
        Returns:
        --------
        pd.DataFrame : Summary with top indicators and their scores
        """
        if row_indices is None:
            row_indices = range(min(self.indicator_matrix.shape[0], 100))
        
        results = []
        for idx in row_indices:
            importance = self.get_row_importance(idx, top_k=top_k)
            
            row_data = {'row_index': idx}
            for i, (name, score) in enumerate(importance.top_indicators, 1):
                row_data[f'top_{i}_indicator'] = name
                row_data[f'top_{i}_score'] = score
            
            results.append(row_data)
        
        return pd.DataFrame(results)
    
    def plot_importance(
        self,
        row_idx: int,
        top_k: int = 15,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None
    ):
        """
        Plot the importance scores for a single row.
        
        Parameters:
        -----------
        row_idx : int
            Index of the row to plot
        top_k : int, default=15
            Number of top indicators to show
        figsize : tuple, default=(10, 6)
            Figure size
        title : str, optional
            Plot title
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        importance = self.get_row_importance(row_idx, top_k=top_k)
        
        names = [name for name, _ in importance.top_indicators]
        scores = [score for _, score in importance.top_indicators]
        
        # Create color map: positive = green, negative = red
        colors = ['green' if s > 0 else 'red' for s in scores]
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(names))
        
        ax.barh(y_pos, scores, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Importance Score')
        ax.set_title(title or f'Top {top_k} Indicator Importance for Row {row_idx}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return fig


def example_usage():
    """Example demonstrating how to use the SnorkelIndicatorImportance class."""
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_indicators = 50
    
    # Simulate indicator matrix (e.g., output from Snorkel labeling functions)
    indicator_matrix = np.random.randn(n_samples, n_indicators)
    
    # Simulate Snorkel weights (importance of each labeling function)
    snorkel_weights = np.random.randn(n_indicators) * 0.5
    
    # Create indicator names
    indicator_names = [f"LF_{i}" for i in range(n_indicators)]
    
    # Initialize importance calculator
    importance_calc = SnorkelIndicatorImportance(
        indicator_matrix=indicator_matrix,
        snorkel_weights=snorkel_weights,
        indicator_names=indicator_names,
        outlier_method='iqr',
        outlier_threshold=1.5
    )
    
    # Analyze a single row
    row_idx = 0
    result = importance_calc.get_row_importance(row_idx, top_k=10)
    
    print(f"Analysis for Row {row_idx}:")
    print("\nTop 10 Most Important Indicators:")
    for name, score in result.top_indicators:
        print(f"  {name}: {score:.4f}")
    
    print(f"\nOutlier Indicators ({len(result.outlier_indicators)} found):")
    for name, value in result.outlier_indicators[:5]:  # Show first 5
        print(f"  {name}: {value:.4f}")
    
    # Analyze multiple rows
    explanations = importance_calc.explain_predictions(
        row_indices=[0, 1, 2],
        top_k=5
    )
    
    print(f"\n\nExplanations for {len(explanations)} rows generated.")
    
    # Create summary DataFrame
    df_summary = importance_calc.to_dataframe(row_indices=range(10), top_k=3)
    print("\nSummary DataFrame (first 5 rows):")
    print(df_summary.head())
    
    return importance_calc, result


if __name__ == "__main__":
    importance_calc, result = example_usage()
