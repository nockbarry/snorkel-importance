import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Literal
from dataclasses import dataclass
from scipy.spatial.distance import cosine, euclidean, cityblock, hamming
from scipy.stats import entropy


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
    
    def compute_importance_scores_vectorized(
        self,
        row_indices: Optional[np.ndarray] = None,
        normalize: bool = True,
        consider_sign: bool = True
    ) -> np.ndarray:
        """
        Compute importance scores for multiple rows at once (vectorized).
        
        This is much faster than calling compute_importance_scores in a loop.
        
        Parameters:
        -----------
        row_indices : np.ndarray, optional
            Indices of rows to compute importance for. If None, computes for all rows.
        normalize : bool, default=True
            Whether to normalize scores to sum to 1 per row
        consider_sign : bool, default=True
            Whether to preserve the sign of indicator * weight product
        
        Returns:
        --------
        np.ndarray : Importance scores matrix of shape (n_rows, n_indicators)
        """
        # Select rows
        if row_indices is None:
            data = self.indicator_matrix
        else:
            data = self.indicator_matrix[row_indices]
        
        # Vectorized computation: element-wise multiply each row by weights
        # Broadcasting: (n_rows, n_indicators) * (n_indicators,) -> (n_rows, n_indicators)
        importance = data * self.weights[np.newaxis, :]
        
        if not consider_sign:
            importance = np.abs(importance)
        
        # Normalize each row independently if requested
        if normalize:
            row_sums = np.sum(np.abs(importance), axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.where(row_sums > 0, row_sums, 1)
            importance = importance / row_sums
        
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
    
    def detect_outliers_vectorized(
        self,
        row_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Detect outliers for multiple rows at once (vectorized).
        
        Parameters:
        -----------
        row_indices : np.ndarray, optional
            Indices of rows to check. If None, checks all rows.
        
        Returns:
        --------
        np.ndarray : Boolean matrix of shape (n_rows, n_indicators) indicating outliers
        """
        # Select rows
        if row_indices is None:
            data = self.indicator_matrix
        else:
            data = self.indicator_matrix[row_indices]
        
        if self.outlier_method == 'iqr':
            # IQR method: vectorized across all rows
            lower_bound = self.q25 - self.outlier_threshold * self.iqr
            upper_bound = self.q75 + self.outlier_threshold * self.iqr
            # Broadcasting: (n_rows, n_indicators) compared to (n_indicators,)
            outliers = (data < lower_bound[np.newaxis, :]) | (data > upper_bound[np.newaxis, :])
        
        elif self.outlier_method == 'zscore':
            # Z-score method: vectorized
            with np.errstate(divide='ignore', invalid='ignore'):
                z_scores = np.abs((data - self.means[np.newaxis, :]) / self.stds[np.newaxis, :])
                z_scores = np.nan_to_num(z_scores, nan=0.0)
            outliers = z_scores > self.outlier_threshold
        
        elif self.outlier_method == 'percentile':
            # Percentile method: vectorized
            lower_threshold = np.percentile(self.indicator_matrix, self.outlier_threshold, axis=0)
            upper_threshold = np.percentile(self.indicator_matrix, 100 - self.outlier_threshold, axis=0)
            outliers = (data < lower_threshold[np.newaxis, :]) | (data > upper_threshold[np.newaxis, :])
        
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
            row_indices = list(range(self.indicator_matrix.shape[0]))
        
        # Convert to numpy array for vectorized operations
        row_indices_array = np.array(row_indices)
        
        # Compute importance scores for all rows at once (VECTORIZED)
        importance_matrix = self.compute_importance_scores_vectorized(
            row_indices=row_indices_array,
            normalize=normalize,
            consider_sign=True
        )
        
        # Detect outliers for all rows at once (VECTORIZED)
        outliers_matrix = self.detect_outliers_vectorized(row_indices=row_indices_array)
        
        # Build results for each row
        results = []
        for i, orig_idx in enumerate(row_indices):
            importance_scores = importance_matrix[i]
            outliers = outliers_matrix[i]
            
            # Get top-k indicators by absolute importance
            top_indices = np.argsort(np.abs(importance_scores))[-top_k:][::-1]
            top_indicators = [
                (self.indicator_names[idx], importance_scores[idx])
                for idx in top_indices
            ]
            
            # Get outlier indicators
            outlier_indices = np.where(outliers)[0]
            outlier_indicators = [
                (self.indicator_names[idx], self.indicator_matrix[orig_idx, idx])
                for idx in outlier_indices
            ]
            
            # Create dictionary of all scores
            all_scores = {
                name: score
                for name, score in zip(self.indicator_names, importance_scores)
            }
            
            results.append(IndicatorImportance(
                row_index=orig_idx,
                top_indicators=top_indicators,
                outlier_indicators=outlier_indicators,
                all_scores=all_scores
            ))
        
        return results
    
    def get_top_k_matrix(
        self,
        row_indices: Optional[np.ndarray] = None,
        top_k: int = 10,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-k indicator indices and scores for multiple rows (vectorized).
        
        This is the most efficient way to get top indicators for many rows.
        
        Parameters:
        -----------
        row_indices : np.ndarray, optional
            Indices of rows to analyze. If None, analyzes all rows.
        top_k : int, default=10
            Number of top indicators to return per row
        normalize : bool, default=True
            Whether to normalize importance scores
        
        Returns:
        --------
        top_k_indices : np.ndarray
            Matrix of shape (n_rows, top_k) with indicator indices
        top_k_scores : np.ndarray
            Matrix of shape (n_rows, top_k) with importance scores
        """
        # Compute all importance scores at once
        importance_matrix = self.compute_importance_scores_vectorized(
            row_indices=row_indices,
            normalize=normalize,
            consider_sign=True
        )
        
        # Use numpy's argpartition for efficient top-k selection
        # This is faster than full sorting when k << n_indicators
        n_indicators = importance_matrix.shape[1]
        
        if top_k >= n_indicators:
            # If requesting all indicators, just sort
            top_k_indices = np.argsort(np.abs(importance_matrix), axis=1)[:, ::-1]
        else:
            # Use argpartition for efficiency (O(n) vs O(n log n))
            # Get indices of top-k by absolute value
            partition_indices = np.argpartition(
                np.abs(importance_matrix), 
                -top_k, 
                axis=1
            )[:, -top_k:]
            
            # Sort these top-k indices by their actual values
            # For each row, get the scores of the top-k indicators
            rows = np.arange(importance_matrix.shape[0])[:, np.newaxis]
            top_k_values = np.abs(importance_matrix[rows, partition_indices])
            
            # Get sorting order within the top-k
            sort_order = np.argsort(top_k_values, axis=1)[:, ::-1]
            
            # Rearrange partition_indices according to sort_order
            top_k_indices = partition_indices[rows, sort_order]
        
        # Extract the actual scores
        rows = np.arange(importance_matrix.shape[0])[:, np.newaxis]
        top_k_scores = importance_matrix[rows, top_k_indices[:, :top_k]]
        
        return top_k_indices[:, :top_k], top_k_scores
    
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
            row_indices = list(range(min(self.indicator_matrix.shape[0], 100)))
        
        # Use vectorized method for efficiency
        row_indices_array = np.array(row_indices)
        top_indices, top_scores = self.get_top_k_matrix(
            row_indices=row_indices_array,
            top_k=top_k,
            normalize=True
        )
        
        # Build DataFrame
        results = []
        for i, orig_idx in enumerate(row_indices):
            row_data = {'row_index': orig_idx}
            for j in range(top_k):
                indicator_idx = top_indices[i, j]
                row_data[f'top_{j+1}_indicator'] = self.indicator_names[indicator_idx]
                row_data[f'top_{j+1}_score'] = top_scores[i, j]
            results.append(row_data)
        
        return pd.DataFrame(results)
    
    def get_all_importance_scores(
        self,
        normalize: bool = True,
        as_dataframe: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get importance scores for ALL rows and ALL indicators (vectorized).
        
        This is the most efficient way to get the complete importance matrix.
        
        Parameters:
        -----------
        normalize : bool, default=True
            Whether to normalize scores per row
        as_dataframe : bool, default=False
            If True, return as DataFrame with indicator names as columns
        
        Returns:
        --------
        np.ndarray or pd.DataFrame : 
            Importance matrix of shape (n_samples, n_indicators)
        """
        importance_matrix = self.compute_importance_scores_vectorized(
            row_indices=None,
            normalize=normalize,
            consider_sign=True
        )
        
        if as_dataframe:
            return pd.DataFrame(
                importance_matrix,
                columns=self.indicator_names
            )
        else:
            return importance_matrix
    
    def get_indicator_patterns(
        self,
        binarize: bool = True,
        threshold: float = 0.0
    ) -> np.ndarray:
        """
        Get indicator firing patterns (which indicators are active).
        
        This focuses on WHICH indicators fired, ignoring weights entirely.
        Useful when weights are clustered and less discriminative.
        
        Parameters:
        -----------
        binarize : bool, default=True
            If True, returns binary pattern (fired or not)
            If False, returns raw indicator values
        threshold : float, default=0.0
            For binarization, indicators > threshold are marked as 1
        
        Returns:
        --------
        np.ndarray : Pattern matrix of shape (n_samples, n_indicators)
        """
        if binarize:
            return (self.indicator_matrix != threshold).astype(int)
        else:
            return self.indicator_matrix.copy()
    
    def compute_deviation_importance(
        self,
        row_indices: Optional[np.ndarray] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute importance based on deviation from typical patterns.
        
        This emphasizes indicators that fire unusually for an instance,
        helping distinguish instances when weights are similar.
        
        Formula: importance = weight × (indicator_value - mean_indicator_value)
        
        Parameters:
        -----------
        row_indices : np.ndarray, optional
            Indices to compute for. If None, computes for all rows.
        normalize : bool, default=True
            Whether to normalize per row
        
        Returns:
        --------
        np.ndarray : Deviation-based importance of shape (n_rows, n_indicators)
        """
        if row_indices is None:
            data = self.indicator_matrix
        else:
            data = self.indicator_matrix[row_indices]
        
        # Compute deviation from mean
        deviations = data - self.means[np.newaxis, :]
        
        # Weight by importance
        deviation_importance = deviations * self.weights[np.newaxis, :]
        
        if normalize:
            row_sums = np.sum(np.abs(deviation_importance), axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1)
            deviation_importance = deviation_importance / row_sums
        
        return deviation_importance
    
    def find_similar_instances(
        self,
        target_idx: int,
        top_k: int = 10,
        method: Literal['cosine', 'euclidean', 'manhattan', 'hamming', 'pattern', 'deviation', 'combined'] = 'combined',
        use_importance: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Find instances most similar to a target instance.
        
        When weights are clustered, use 'pattern', 'hamming', or 'combined' methods
        for better discrimination.
        
        Parameters:
        -----------
        target_idx : int
            Index of target instance
        top_k : int, default=10
            Number of similar instances to return
        method : str, default='combined'
            Similarity method:
            - 'cosine': Cosine similarity on importance scores
            - 'euclidean': Euclidean distance on importance scores
            - 'manhattan': Manhattan distance on importance scores
            - 'hamming': Hamming distance on indicator patterns (which fired)
            - 'pattern': Jaccard similarity on firing patterns
            - 'deviation': Similarity based on deviation importance
            - 'combined': Weighted combination of multiple methods
        use_importance : bool, default=True
            Whether to use weighted importance or raw indicators
        
        Returns:
        --------
        list of (index, similarity) : Top-k most similar instances
        """
        if method == 'combined':
            # Use weighted combination of methods
            # This works well when weights are clustered
            scores_importance = self._compute_similarities(target_idx, 'cosine', True)
            scores_pattern = self._compute_similarities(target_idx, 'hamming', False)
            scores_deviation = self._compute_deviation_similarities(target_idx)
            
            # Normalize all scores to [0, 1]
            scores_importance = (scores_importance - scores_importance.min()) / (scores_importance.max() - scores_importance.min() + 1e-10)
            scores_pattern = (scores_pattern - scores_pattern.min()) / (scores_pattern.max() - scores_pattern.min() + 1e-10)
            scores_deviation = (scores_deviation - scores_deviation.min()) / (scores_deviation.max() - scores_deviation.min() + 1e-10)
            
            # Weighted combination (emphasize pattern and deviation when weights cluster)
            combined_scores = 0.3 * scores_importance + 0.4 * scores_pattern + 0.3 * scores_deviation
            
            # Get top-k (excluding self)
            combined_scores[target_idx] = -np.inf
            top_indices = np.argsort(combined_scores)[-top_k:][::-1]
            
            return [(int(idx), float(combined_scores[idx])) for idx in top_indices]
        
        elif method == 'deviation':
            scores = self._compute_deviation_similarities(target_idx)
        else:
            scores = self._compute_similarities(target_idx, method, use_importance)
        
        # Exclude self
        scores[target_idx] = -np.inf if method in ['cosine', 'pattern'] else np.inf
        
        # Get top-k
        if method in ['euclidean', 'manhattan', 'hamming']:
            # Lower is better for distances
            top_indices = np.argsort(scores)[:top_k]
        else:
            # Higher is better for similarities
            top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def _compute_similarities(
        self,
        target_idx: int,
        method: str,
        use_importance: bool
    ) -> np.ndarray:
        """Helper to compute similarity scores."""
        if use_importance:
            all_data = self.get_all_importance_scores(normalize=True)
        else:
            all_data = self.get_indicator_patterns(binarize=(method == 'hamming'))
        
        target = all_data[target_idx]
        
        if method == 'cosine':
            # Compute cosine similarity for all rows
            similarities = np.zeros(len(all_data))
            for i in range(len(all_data)):
                if i == target_idx:
                    similarities[i] = 1.0
                else:
                    similarities[i] = 1 - cosine(target, all_data[i])
            return similarities
        
        elif method == 'euclidean':
            return np.linalg.norm(all_data - target, axis=1)
        
        elif method == 'manhattan':
            return np.sum(np.abs(all_data - target), axis=1)
        
        elif method == 'hamming':
            # Hamming distance on binary patterns
            return np.sum(all_data != target, axis=1) / all_data.shape[1]
        
        elif method == 'pattern':
            # Jaccard similarity: intersection / union of active indicators
            target_active = set(np.where(target != 0)[0])
            similarities = np.zeros(len(all_data))
            
            for i in range(len(all_data)):
                row_active = set(np.where(all_data[i] != 0)[0])
                if len(target_active) == 0 and len(row_active) == 0:
                    similarities[i] = 1.0
                elif len(target_active.union(row_active)) == 0:
                    similarities[i] = 0.0
                else:
                    similarities[i] = len(target_active.intersection(row_active)) / len(target_active.union(row_active))
            
            return similarities
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_deviation_similarities(self, target_idx: int) -> np.ndarray:
        """Compute similarities based on deviation importance."""
        all_deviation = self.compute_deviation_importance(normalize=True)
        target = all_deviation[target_idx]
        
        # Use cosine similarity on deviation importance
        similarities = np.zeros(len(all_deviation))
        for i in range(len(all_deviation)):
            if i == target_idx:
                similarities[i] = 1.0
            else:
                similarities[i] = 1 - cosine(target, all_deviation[i])
        
        return similarities
    
    def compute_instance_diversity(
        self,
        row_indices: Optional[List[int]] = None,
        method: str = 'hamming',
        sample_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute diversity/uniqueness score for each instance.
        
        Higher scores indicate more unique instances (different from others).
        For large datasets, uses sampling for efficiency.
        
        Parameters:
        -----------
        row_indices : list of int, optional
            Indices to analyze. If None, analyzes all rows.
        method : str, default='hamming'
            Method for computing diversity
        sample_size : int, optional
            Number of samples to compare against (for large datasets).
            If None, uses min(1000, n_samples)
        
        Returns:
        --------
        np.ndarray : Diversity scores, one per instance
        """
        if row_indices is None:
            row_indices = list(range(self.indicator_matrix.shape[0]))
        
        patterns = self.get_indicator_patterns(binarize=True)
        
        if method == 'hamming':
            # For efficiency, compare against a sample rather than all instances
            if sample_size is None:
                sample_size = min(1000, len(patterns))
            
            # Sample instances to compare against
            compare_indices = np.random.choice(
                len(patterns),
                size=min(sample_size, len(patterns)),
                replace=False
            )
            compare_patterns = patterns[compare_indices]
            
            # Vectorized diversity computation
            diversity_scores = np.zeros(len(row_indices))
            
            # Process in batches for memory efficiency
            batch_size = 100
            for i in range(0, len(row_indices), batch_size):
                batch_end = min(i + batch_size, len(row_indices))
                batch_indices = row_indices[i:batch_end]
                batch_patterns = patterns[batch_indices]
                
                # Compute Hamming distances: (batch_size, n_indicators) vs (sample_size, n_indicators)
                # Result: (batch_size, sample_size)
                distances = np.mean(batch_patterns[:, np.newaxis, :] != compare_patterns[np.newaxis, :, :], axis=2)
                
                # Average distance to sampled instances
                diversity_scores[i:batch_end] = np.mean(distances, axis=1)
            
            return diversity_scores
        
        elif method == 'variance':
            # Fast alternative: variance of indicator values
            # Higher variance = more unique pattern
            diversity_scores = np.var(patterns[row_indices], axis=1)
            return diversity_scores
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
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
    
    def plot_indicator_heatmap(
        self,
        row_indices: Optional[List[int]] = None,
        indicator_indices: Optional[List[int]] = None,
        use_importance: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        cluster_rows: bool = True,
        cluster_cols: bool = True,
        cmap: str = 'RdBu_r'
    ):
        """
        Plot heatmap of indicator patterns or importance across instances.
        
        Useful for visualizing patterns when weights are similar.
        Clustering helps identify groups of similar instances.
        
        Parameters:
        -----------
        row_indices : list of int, optional
            Instances to include. If None, uses first 100 or all if less.
        indicator_indices : list of int, optional
            Indicators to include. If None, uses all.
        use_importance : bool, default=True
            If True, shows importance scores. If False, shows raw indicators.
        figsize : tuple, default=(12, 8)
            Figure size
        cluster_rows : bool, default=True
            Whether to cluster rows (instances)
        cluster_cols : bool, default=True
            Whether to cluster columns (indicators)
        cmap : str, default='RdBu_r'
            Colormap
        
        Returns:
        --------
        Figure : matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn required. Install with: pip install matplotlib seaborn")
        
        # Select data
        if row_indices is None:
            row_indices = list(range(min(100, self.indicator_matrix.shape[0])))
        
        if indicator_indices is None:
            indicator_indices = list(range(self.indicator_matrix.shape[1]))
        
        if use_importance:
            data = self.compute_importance_scores_vectorized(
                row_indices=np.array(row_indices),
                normalize=True
            )[:, indicator_indices]
        else:
            data = self.indicator_matrix[np.ix_(row_indices, indicator_indices)]
        
        # Create labels
        row_labels = [f"Row {i}" for i in row_indices]
        col_labels = [self.indicator_names[i] for i in indicator_indices]
        
        # Plot
        if cluster_rows or cluster_cols:
            from scipy.cluster import hierarchy
            from scipy.spatial.distance import pdist, squareform
            
            # Cluster rows
            if cluster_rows:
                row_linkage = hierarchy.linkage(data, method='ward')
                row_order = hierarchy.dendrogram(row_linkage, no_plot=True)['leaves']
                data = data[row_order, :]
                row_labels = [row_labels[i] for i in row_order]
            
            # Cluster columns
            if cluster_cols:
                col_linkage = hierarchy.linkage(data.T, method='ward')
                col_order = hierarchy.dendrogram(col_linkage, no_plot=True)['leaves']
                data = data[:, col_order]
                col_labels = [col_labels[i] for i in col_order]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest')
        
        # Set ticks
        if len(row_indices) <= 50:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=8)
        
        if len(indicator_indices) <= 30:
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
        
        ax.set_xlabel('Indicators')
        ax.set_ylabel('Instances')
        ax.set_title('Indicator Pattern Heatmap' + (' (Clustered)' if cluster_rows or cluster_cols else ''))
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Importance' if use_importance else 'Value')
        
        plt.tight_layout()
        return fig
    
    def plot_similarity_matrix(
        self,
        row_indices: Optional[List[int]] = None,
        method: str = 'combined',
        figsize: Tuple[int, int] = (10, 8),
        cluster: bool = True
    ):
        """
        Plot similarity matrix between instances.
        
        Shows which instances are similar/different. Use 'combined' or 'pattern'
        method when weights are clustered.
        
        Parameters:
        -----------
        row_indices : list of int, optional
            Instances to compare. If None, uses first 100.
        method : str, default='combined'
            Similarity method (see find_similar_instances)
        figsize : tuple, default=(10, 8)
            Figure size
        cluster : bool, default=True
            Whether to reorder by hierarchical clustering
        
        Returns:
        --------
        Figure : matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        
        if row_indices is None:
            row_indices = list(range(min(100, self.indicator_matrix.shape[0])))
        
        n = len(row_indices)
        similarity_matrix = np.zeros((n, n))
        
        # Compute pairwise similarities
        print(f"Computing {n}x{n} similarity matrix...")
        for i, idx_i in enumerate(row_indices):
            if method == 'combined':
                # Use the combined similarity method
                for j, idx_j in enumerate(row_indices):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Get similarity using find_similar_instances
                        similar = self.find_similar_instances(
                            target_idx=idx_i,
                            top_k=n,
                            method='combined'
                        )
                        # Find the score for idx_j
                        for sim_idx, score in similar:
                            if sim_idx == idx_j:
                                similarity_matrix[i, j] = score
                                break
            else:
                similarities = self._compute_similarities(idx_i, method, True)
                for j, idx_j in enumerate(row_indices):
                    similarity_matrix[i, j] = similarities[idx_j]
        
        # Cluster if requested
        if cluster:
            from scipy.cluster import hierarchy
            linkage = hierarchy.linkage(similarity_matrix, method='ward')
            order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']
            similarity_matrix = similarity_matrix[np.ix_(order, order)]
            row_indices = [row_indices[i] for i in order]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        
        ax.set_xlabel('Instance')
        ax.set_ylabel('Instance')
        ax.set_title(f'Instance Similarity Matrix ({method} method)' + 
                    (' - Clustered' if cluster else ''))
        
        plt.colorbar(im, ax=ax, label='Similarity')
        plt.tight_layout()
        
        return fig
    
    def plot_instance_comparison(
        self,
        row_indices: List[int],
        top_k: int = 10,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Compare indicator patterns across multiple instances side-by-side.
        
        Useful for understanding how similar instances differ.
        
        Parameters:
        -----------
        row_indices : list of int
            Instances to compare (max 5 recommended)
        top_k : int, default=10
            Number of top indicators to show per instance
        figsize : tuple, default=(12, 6)
            Figure size
        
        Returns:
        --------
        Figure : matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        
        n_instances = len(row_indices)
        
        fig, axes = plt.subplots(1, n_instances, figsize=figsize, sharey=True)
        if n_instances == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, row_indices):
            result = self.get_row_importance(idx, top_k=top_k)
            
            names = [name[:20] for name, _ in result.top_indicators]  # Truncate names
            scores = [score for _, score in result.top_indicators]
            colors = ['green' if s > 0 else 'red' for s in scores]
            
            y_pos = np.arange(len(names))
            ax.barh(y_pos, scores, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            if ax == axes[0]:
                ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('Score', fontsize=8)
            ax.set_title(f'Row {idx}', fontsize=10)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        fig.suptitle('Instance Comparison', fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def plot_indicator_radar(
        self,
        row_idx: int,
        top_k: int = 10,
        figsize: Tuple[int, int] = (8, 8)
    ):
        """
        Plot radar/spider chart for a single instance.
        
        Shows the importance profile across indicators.
        
        Parameters:
        -----------
        row_idx : int
            Instance to plot
        top_k : int, default=10
            Number of indicators to include
        figsize : tuple, default=(8, 8)
            Figure size
        
        Returns:
        --------
        Figure : matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        
        result = self.get_row_importance(row_idx, top_k=top_k)
        
        names = [name[:15] for name, _ in result.top_indicators]  # Truncate
        scores = np.abs([score for _, score in result.top_indicators])  # Use absolute
        
        # Normalize to 0-1
        if np.max(scores) > 0:
            scores = scores / np.max(scores)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
        scores = scores.tolist()
        
        # Close the plot
        angles += angles[:1]
        scores += scores[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, scores, 'o-', linewidth=2)
        ax.fill(angles, scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'Indicator Profile - Row {row_idx}', fontsize=12, pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_pattern_pca(
        self,
        row_indices: Optional[List[int]] = None,
        use_importance: bool = True,
        color_by: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot PCA projection of indicator patterns.
        
        Visualizes how instances cluster in pattern space. Useful when
        weights are similar - helps see true pattern diversity.
        
        Parameters:
        -----------
        row_indices : list of int, optional
            Instances to include. If None, uses all.
        use_importance : bool, default=True
            Whether to use importance scores or raw indicators
        color_by : np.ndarray, optional
            Values to color points by (e.g., labels, outlier counts)
        figsize : tuple, default=(10, 8)
            Figure size
        
        Returns:
        --------
        Figure : matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("matplotlib and sklearn required. Install with: pip install matplotlib scikit-learn")
        
        if row_indices is None:
            row_indices = list(range(self.indicator_matrix.shape[0]))
        
        if use_importance:
            data = self.compute_importance_scores_vectorized(
                row_indices=np.array(row_indices),
                normalize=True
            )
        else:
            data = self.indicator_matrix[row_indices]
        
        # Apply PCA
        pca = PCA(n_components=2)
        projected = pca.fit_transform(data)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_by is not None:
            scatter = ax.scatter(projected[:, 0], projected[:, 1], 
                               c=color_by, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(projected[:, 0], projected[:, 1], alpha=0.6)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('Instance Patterns - PCA Projection')
        ax.grid(True, alpha=0.3)
        
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
    
    print("=" * 80)
    print("SINGLE ROW ANALYSIS")
    print("=" * 80)
    
    # Analyze a single row
    row_idx = 0
    result = importance_calc.get_row_importance(row_idx, top_k=10)
    
    print(f"\nAnalysis for Row {row_idx}:")
    print("\nTop 10 Most Important Indicators:")
    for name, score in result.top_indicators:
        print(f"  {name}: {score:.4f}")
    
    print(f"\nOutlier Indicators ({len(result.outlier_indicators)} found):")
    for name, value in result.outlier_indicators[:5]:  # Show first 5
        print(f"  {name}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("VECTORIZED BATCH ANALYSIS")
    print("=" * 80)
    
    # Demonstrate vectorized computation
    print("\n1. Get ALL importance scores at once (vectorized):")
    import time
    
    start = time.time()
    all_importance = importance_calc.get_all_importance_scores(normalize=True)
    elapsed = time.time() - start
    print(f"   Computed importance for ALL {n_samples} rows in {elapsed:.4f} seconds")
    print(f"   Shape: {all_importance.shape}")
    print(f"   Example row 0 scores: {all_importance[0, :5]}")  # First 5 scores
    
    # Get as DataFrame
    print("\n2. Get as DataFrame with indicator names:")
    importance_df = importance_calc.get_all_importance_scores(
        normalize=True, 
        as_dataframe=True
    )
    print(f"   DataFrame shape: {importance_df.shape}")
    print(f"   Columns: {list(importance_df.columns[:5])}... (showing first 5)")
    print("\n   First 3 rows:")
    print(importance_df.head(3))
    
    # Get top-k for all rows efficiently
    print("\n3. Get top-k indicators for all rows (vectorized):")
    start = time.time()
    top_indices, top_scores = importance_calc.get_top_k_matrix(top_k=10)
    elapsed = time.time() - start
    print(f"   Computed top-10 for ALL {n_samples} rows in {elapsed:.4f} seconds")
    print(f"   Top indices shape: {top_indices.shape}")
    print(f"   Top scores shape: {top_scores.shape}")
    print(f"\n   Row 0 top indicators: {[indicator_names[i] for i in top_indices[0, :5]]}")
    print(f"   Row 0 top scores: {top_scores[0, :5]}")
    
    # Vectorized outlier detection
    print("\n4. Detect outliers for all rows (vectorized):")
    start = time.time()
    all_outliers = importance_calc.detect_outliers_vectorized()
    elapsed = time.time() - start
    outlier_counts = np.sum(all_outliers, axis=1)
    print(f"   Detected outliers for ALL {n_samples} rows in {elapsed:.4f} seconds")
    print(f"   Outlier matrix shape: {all_outliers.shape}")
    print(f"   Rows with most outliers: {np.sort(outlier_counts)[-5:]}")
    print(f"   Mean outliers per row: {np.mean(outlier_counts):.2f}")
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Compare loop vs vectorized approach
    n_test = 100
    print(f"\nComparing performance for {n_test} rows:")
    
    # Method 1: Loop (old way)
    print("\n  Method 1: Loop through rows")
    start = time.time()
    for idx in range(n_test):
        _ = importance_calc.get_row_importance(idx, top_k=10)
    loop_time = time.time() - start
    print(f"    Time: {loop_time:.4f} seconds")
    
    # Method 2: Vectorized (new way)
    print("\n  Method 2: Vectorized batch computation")
    start = time.time()
    _ = importance_calc.explain_predictions(row_indices=list(range(n_test)), top_k=10)
    vectorized_time = time.time() - start
    print(f"    Time: {vectorized_time:.4f} seconds")
    
    speedup = loop_time / vectorized_time
    print(f"\n  ⚡ Speedup: {speedup:.1f}x faster with vectorization!")
    
    print("\n" + "=" * 80)
    print("PRACTICAL EXAMPLES")
    print("=" * 80)
    
    # Analyze multiple rows
    explanations = importance_calc.explain_predictions(
        row_indices=[0, 1, 2],
        top_k=5
    )
    
    print(f"\nExplanations for {len(explanations)} rows generated.")
    
    # Create summary DataFrame (now using vectorized method internally)
    df_summary = importance_calc.to_dataframe(row_indices=range(10), top_k=3)
    print("\nSummary DataFrame (first 5 rows):")
    print(df_summary.head())
    
    # Advanced: Find rows with similar importance patterns
    print("\n" + "=" * 80)
    print("ADVANCED: Finding Similar Rows")
    print("=" * 80)
    
    from scipy.spatial.distance import cdist
    
    # Get all importance scores
    all_scores = importance_calc.get_all_importance_scores(normalize=True)
    
    # Find rows most similar to row 0
    target_row = all_scores[0:1, :]  # Keep 2D for cdist
    distances = cdist(target_row, all_scores[1:], metric='cosine')[0]
    most_similar_indices = np.argsort(distances)[:5] + 1  # +1 because we excluded row 0
    
    print(f"\nRows most similar to row 0 (by cosine similarity):")
    for rank, idx in enumerate(most_similar_indices, 1):
        similarity = 1 - distances[idx - 1]
        print(f"  {rank}. Row {idx}: similarity = {similarity:.4f}")
    
    return importance_calc, result


if __name__ == "__main__":
    importance_calc, result = example_usage()
