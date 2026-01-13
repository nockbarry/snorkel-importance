"""
Large-Scale Indicator Importance Analysis

Optimized for datasets with millions of rows.
Supports focus rows for detailed analysis and smart sampling for visualizations.

Usage:
    from large_scale_analysis import LargeScaleAnalysis
    
    analysis = LargeScaleAnalysis(
        indicator_matrix=L_matrix,  # 5M × 160
        snorkel_weights=weights,
        indicator_names=lf_names,
        pandas_index=df.index,  # Optional
        output_dir="./analysis_5M"
    )
    
    # Run with focus rows
    results = analysis.run_full_analysis(
        focus_rows=[42, 100, 500, 1000],  # Rows of interest
        viz_sample_size=10000,             # Large sample for structure
        sampling_method='clustering'       # Smart sampling
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, List, Union
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('.')
from indicator_importance import SnorkelIndicatorImportance


class LargeScaleAnalysis:
    """
    Comprehensive analysis optimized for large datasets (millions of rows).
    """
    
    def __init__(
        self,
        indicator_matrix: Union[np.ndarray, pd.DataFrame],
        snorkel_weights: np.ndarray,
        indicator_names: List[str],
        pandas_index: Optional[np.ndarray] = None,
        output_dir: str = "./large_scale_analysis"
    ):
        """
        Initialize large-scale analysis.
        
        Parameters:
        -----------
        indicator_matrix : np.ndarray or pd.DataFrame
            Matrix of shape (n_samples, n_indicators)
        snorkel_weights : np.ndarray
            Learned weights from Snorkel
        indicator_names : list of str
            Names of indicators/labeling functions
        pandas_index : np.ndarray, optional
            Original DataFrame index for proper row labeling
        output_dir : str
            Output directory
        """
        # Handle DataFrame input
        if isinstance(indicator_matrix, pd.DataFrame):
            if pandas_index is None:
                pandas_index = indicator_matrix.index.values
            indicator_matrix = indicator_matrix.values
        
        self.indicator_matrix = indicator_matrix
        self.snorkel_weights = snorkel_weights
        self.indicator_names = indicator_names
        self.pandas_index = pandas_index
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create index lookup for fast label -> position conversion
        if self.pandas_index is not None:
            self.index_to_position = {label: pos for pos, label in enumerate(self.pandas_index)}
        else:
            self.index_to_position = {}
        
        # Initialize calculator
        self.calc = SnorkelIndicatorImportance(
            indicator_matrix=indicator_matrix,
            snorkel_weights=snorkel_weights,
            indicator_names=indicator_names
        )
        
        print("=" * 80)
        print("LARGE-SCALE INDICATOR IMPORTANCE ANALYSIS")
        print("=" * 80)
        print(f"Dataset: {indicator_matrix.shape[0]:,} samples × {indicator_matrix.shape[1]} indicators")
        print(f"Output: {self.output_dir}")
        print("=" * 80)
        print()
    
    def convert_labels_to_positions(self, labels: List) -> List[int]:
        """
        Convert pandas index labels to positional indices.
        
        Parameters:
        -----------
        labels : list
            List of pandas index values (e.g., [3742123, 4526073])
        
        Returns:
        --------
        list of int : Positional indices
        """
        if not self.index_to_position:
            # No pandas index provided, assume labels are already positions
            return labels
        
        positions = []
        not_found = []
        
        for label in labels:
            if label in self.index_to_position:
                positions.append(self.index_to_position[label])
            else:
                not_found.append(label)
        
        if not_found:
            print(f"⚠️  Warning: {len(not_found)} labels not found in index:")
            for label in not_found[:10]:
                print(f"    {label}")
            if len(not_found) > 10:
                print(f"    ... and {len(not_found) - 10} more")
        
        return positions
    
    def get_row_label(self, row_idx: int) -> str:
        """Get label for a row (uses pandas index if available)."""
        if self.pandas_index is not None:
            return str(self.pandas_index[row_idx])
        return f"row_{row_idx}"
    
    def smart_sample(
        self,
        n_samples: int,
        method: str = 'clustering',
        include_focus: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Smart sampling for visualizations.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to select
        method : str
            'clustering' (best for structure), 'stratified', or 'random'
        include_focus : list of int, optional
            Always include these indices (must be positional indices, not labels!)
        
        Returns:
        --------
        np.ndarray : Selected indices
        """
        total_samples = self.indicator_matrix.shape[0]
        n_samples = min(n_samples, total_samples)
        
        # Validate focus indices if provided
        if include_focus:
            original_count = len(include_focus)
            include_focus = [idx for idx in include_focus if 0 <= idx < total_samples]
            if len(include_focus) < original_count:
                print(f"  ⚠️  Warning: {original_count - len(include_focus)} focus indices out of bounds, filtered")
            if include_focus:
                n_samples = max(n_samples, len(include_focus))
        
        if method == 'random':
            selected = np.random.choice(total_samples, size=n_samples, replace=False)
        
        elif method == 'stratified':
            print(f"  Using stratified sampling (by outlier count)...")
            outlier_counts = np.sum(self.calc.detect_outliers_vectorized(), axis=1)
            
            # Divide into quartiles
            q25, q50, q75 = np.percentile(outlier_counts, [25, 50, 75])
            
            bins = [
                np.where(outlier_counts <= q25)[0],
                np.where((outlier_counts > q25) & (outlier_counts <= q50))[0],
                np.where((outlier_counts > q50) & (outlier_counts <= q75))[0],
                np.where(outlier_counts > q75)[0]
            ]
            
            # Sample from each bin
            per_bin = n_samples // 4
            selected = []
            for bin_indices in bins:
                if len(bin_indices) > 0:
                    selected.extend(
                        np.random.choice(bin_indices, size=min(per_bin, len(bin_indices)), replace=False)
                    )
            selected = np.array(selected)
        
        elif method == 'clustering':
            print(f"  Using clustering-based sampling...")
            from sklearn.cluster import MiniBatchKMeans
            
            # Cluster on patterns
            patterns = self.calc.get_indicator_patterns(binarize=True).astype(np.float32)
            
            # Number of clusters
            n_clusters = min(int(np.sqrt(n_samples)), n_samples // 2)
            print(f"  Creating {n_clusters} clusters...")
            
            # Use MiniBatchKMeans for large datasets
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=min(10000, total_samples),
                verbose=0
            )
            clusters = kmeans.fit_predict(patterns)
            
            # Sample from each cluster
            selected = []
            per_cluster = max(1, n_samples // n_clusters)
            
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                if len(cluster_indices) > 0:
                    n_from_cluster = min(per_cluster, len(cluster_indices))
                    selected.extend(
                        np.random.choice(cluster_indices, size=n_from_cluster, replace=False)
                    )
            
            selected = np.array(selected[:n_samples])
        
        # Ensure focus rows are included
        if include_focus:
            selected = np.unique(np.concatenate([selected, include_focus]))[:n_samples]
        
        return selected
    
    def analyze_focus_row(
        self,
        row_idx: int,
        create_visualizations: bool = True
    ) -> dict:
        """
        Detailed analysis of a single focus row.
        
        Parameters:
        -----------
        row_idx : int
            Row index (position in matrix, not pandas index)
        create_visualizations : bool
            Whether to create detailed visualizations
        
        Returns:
        --------
        dict : Detailed analysis results
        """
        print(f"\n  Analyzing focus row: {self.get_row_label(row_idx)}...")
        
        # Get importance
        importance_result = self.calc.get_row_importance(row_idx, top_k=30)
        
        # Find similar instances
        similar = self.calc.find_similar_instances(
            target_idx=row_idx,
            top_k=10,
            method='combined'
        )
        
        # Deviation importance
        deviation_imp = self.calc.compute_deviation_importance(
            row_indices=np.array([row_idx])
        )[0]
        
        analysis = {
            'row_index': row_idx,
            'row_label': self.get_row_label(row_idx),
            'top_20_indicators': [
                {'name': name, 'score': float(score)}
                for name, score in importance_result.top_indicators[:20]
            ],
            'n_outliers': len(importance_result.outlier_indicators),
            'outlier_indicators': [
                {'name': name, 'value': float(value)}
                for name, value in importance_result.outlier_indicators
            ],
            'top_10_similar': [
                {'index': int(idx), 'label': self.get_row_label(idx), 'score': float(score)}
                for idx, score in similar
            ],
            'top_10_deviation': sorted(
                [(self.indicator_names[i], float(deviation_imp[i])) 
                 for i in range(len(deviation_imp))],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]
        }
        
        if create_visualizations:
            self._create_focus_visualizations(row_idx, importance_result, similar)
        
        return analysis
    
    def _create_focus_visualizations(self, row_idx: int, importance_result, similar):
        """Create 4 detailed visualizations for a focus row."""
        focus_dir = self.output_dir / "focus_rows"
        focus_dir.mkdir(exist_ok=True)
        
        row_label = self.get_row_label(row_idx).replace('/', '_')
        
        # 1. Top 20 indicators bar chart
        names = [name[:30] for name, _ in importance_result.top_indicators[:20]]
        scores = [score for _, score in importance_result.top_indicators[:20]]
        colors = ['green' if s > 0 else 'red' for s in scores]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_title(f'Top 20 Indicators - {row_label}', fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(focus_dir / f"{row_label}_top20.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Radar chart
        fig = self.calc.plot_indicator_radar(row_idx, top_k=15, figsize=(10, 10))
        plt.savefig(focus_dir / f"{row_label}_radar.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Comparison with similar instances
        similar_indices = [row_idx] + [idx for idx, _ in similar[:3]]
        fig = self.calc.plot_instance_comparison(similar_indices, top_k=15, figsize=(18, 8))
        plt.savefig(focus_dir / f"{row_label}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Heatmap: this row vs similar rows
        top_30_ind = [self.indicator_names.index(name) for name, _ in importance_result.top_indicators[:30]]
        data = self.indicator_matrix[np.ix_(similar_indices, top_30_ind)]
        
        fig, ax = plt.subplots(figsize=(16, 6))
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax.set_yticks(np.arange(len(similar_indices)))
        ax.set_yticklabels([self.get_row_label(idx) for idx in similar_indices], fontsize=10)
        ax.set_xticks(np.arange(len(top_30_ind)))
        ax.set_xticklabels([self.indicator_names[i][:25] for i in top_30_ind], rotation=90, fontsize=8)
        ax.set_title(f'Indicator Values: {row_label} vs Similar Instances', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Indicator Value')
        plt.tight_layout()
        plt.savefig(focus_dir / f"{row_label}_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Created 4 visualizations for {row_label}")
    
    def run_full_analysis(
        self,
        focus_rows: Optional[List] = None,
        focus_rows_are_labels: bool = True,
        viz_sample_size: int = 5000,
        sampling_method: str = 'clustering',
        diversity_sample_size: int = 5000,
        create_focus_visualizations: bool = True,
        create_investigator_reports: bool = True
    ) -> dict:
        """
        Run complete analysis optimized for large datasets.
        
        Parameters:
        -----------
        focus_rows : list, optional
            Row identifiers (pandas index values or positions)
        focus_rows_are_labels : bool, default=True
            If True, focus_rows are pandas index values (e.g., [3742123, 4526073])
            If False, focus_rows are positional indices (e.g., [0, 1, 2])
        viz_sample_size : int
            Sample size for global visualizations
        sampling_method : str
            'clustering', 'stratified', or 'random'
        diversity_sample_size : int
            Sample size for diversity analysis
        create_focus_visualizations : bool, default=True
            Whether to create 4 visualizations per focus row (can be slow for 3000 rows)
        create_investigator_reports : bool, default=True
            Whether to create detailed HTML reports for investigators
        
        Returns:
        --------
        dict : Complete results
        """
        # Convert labels to positions if needed
        if focus_rows and focus_rows_are_labels:
            print("\nConverting pandas index labels to positions...")
            print(f"  Dataset size: {len(self.indicator_matrix):,} rows")
            print(f"  Focus rows requested: {len(focus_rows)}")
            
            if not self.index_to_position:
                print("\n❌ ERROR: No pandas_index provided!")
                print("   You must set pandas_index=df.index.values to use focus_rows_are_labels=True")
                raise ValueError("pandas_index is required when focus_rows_are_labels=True")
            
            focus_row_positions = self.convert_labels_to_positions(focus_rows)
            
            if not focus_row_positions:
                print("\n❌ ERROR: No focus rows were successfully converted!")
                print("   Check that your focus_rows values match your pandas_index")
                print(f"   Pandas index range: {self.pandas_index[0]} to {self.pandas_index[-1]}")
                print(f"   Focus rows sample: {focus_rows[:5]}")
                raise ValueError("No focus rows could be converted from labels to positions")
            
            print(f"✓ Converted {len(focus_row_positions)}/{len(focus_rows)} labels to positions")
            print(f"  Position range: {min(focus_row_positions)} to {max(focus_row_positions)}")
            
            focus_rows = focus_row_positions
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_samples': self.indicator_matrix.shape[0],
                'n_indicators': self.indicator_matrix.shape[1],
                'viz_sample_size': viz_sample_size,
                'sampling_method': sampling_method,
                'n_focus_rows': len(focus_rows) if focus_rows else 0
            }
        }
        
        # STEP 1: Weight Analysis
        print("\n" + "=" * 80)
        print("STEP 1: Weight Analysis")
        print("=" * 80)
        
        weight_std = np.std(self.snorkel_weights)
        results['weights'] = {
            'mean': float(np.mean(self.snorkel_weights)),
            'std': float(weight_std),
            'clustered': bool(weight_std < 0.2),
            'range': [float(np.min(self.snorkel_weights)), float(np.max(self.snorkel_weights))]
        }
        print(f"Weight std: {weight_std:.4f} - {'CLUSTERED' if weight_std < 0.2 else 'DIVERSE'}")
        
        # STEP 2: All Importance Scores
        print("\n" + "=" * 80)
        print(f"STEP 2: Computing Importance for All {self.indicator_matrix.shape[0]:,} Rows")
        print("=" * 80)
        
        import time
        start = time.time()
        all_importance = self.calc.get_all_importance_scores(normalize=True)
        elapsed = time.time() - start
        print(f"✓ Computed in {elapsed:.1f}s ({self.indicator_matrix.shape[0]/elapsed:,.0f} rows/sec)")
        
        # Global top indicators
        global_importance = np.mean(np.abs(all_importance), axis=0)
        top_20_indices = np.argsort(global_importance)[-20:][::-1]
        
        results['top_indicators'] = [
            {'name': self.indicator_names[idx], 'avg_importance': float(global_importance[idx])}
            for idx in top_20_indices
        ]
        
        # STEP 3: Outlier Detection
        print("\n" + "=" * 80)
        print(f"STEP 3: Outlier Detection Across All {self.indicator_matrix.shape[0]:,} Rows")
        print("=" * 80)
        
        start = time.time()
        all_outliers = self.calc.detect_outliers_vectorized()
        elapsed = time.time() - start
        outlier_counts = np.sum(all_outliers, axis=1)
        print(f"✓ Detected in {elapsed:.1f}s")
        print(f"  Total outliers: {np.sum(all_outliers):,}")
        print(f"  Mean per row: {np.mean(outlier_counts):.2f}")
        print(f"  Rows with outliers: {np.sum(outlier_counts > 0):,} ({100*np.sum(outlier_counts > 0)/len(outlier_counts):.1f}%)")
        
        results['outliers'] = {
            'total': int(np.sum(all_outliers)),
            'mean_per_row': float(np.mean(outlier_counts)),
            'pct_with_outliers': float(100 * np.sum(outlier_counts > 0) / len(outlier_counts))
        }
        
        # STEP 4: Focus Row Analysis
        focus_row_positions = []  # Track converted positions
        if focus_rows:
            print("\n" + "=" * 80)
            print(f"STEP 4: Detailed Analysis of {len(focus_rows)} Focus Rows")
            print("=" * 80)
            
            if len(focus_rows) > 100:
                print(f"⚠️  Large number of focus rows ({len(focus_rows)})")
                if create_focus_visualizations:
                    print(f"   This will create {len(focus_rows) * 4:,} visualizations")
                    print(f"   Consider setting create_focus_visualizations=False for faster processing")
            
            focus_results = []
            for i, row_idx in enumerate(focus_rows):
                if i % 100 == 0 and i > 0:
                    print(f"  Progress: {i}/{len(focus_rows)} rows analyzed...")
                
                if row_idx < len(self.indicator_matrix):
                    focus_analysis = self.analyze_focus_row(
                        row_idx,
                        create_visualizations=create_focus_visualizations
                    )
                    focus_results.append(focus_analysis)
                    focus_row_positions.append(row_idx)  # Track valid positions
                else:
                    print(f"  Warning: Row {row_idx} (position) out of bounds, skipping")
            
            results['focus_analysis'] = focus_results
            print(f"\n✓ Completed detailed analysis of {len(focus_results)} focus rows")
            
            # Create investigator reports
            if create_investigator_reports:
                print("\n  Creating investigator reports...")
                self._create_investigator_reports(focus_results, results)
                print(f"  ✓ Created investigator reports")
        
        # STEP 5: Global Visualizations
        print("\n" + "=" * 80)
        print(f"STEP 5: Creating Global Visualizations (sample={viz_sample_size:,}, method={sampling_method})")
        print("=" * 80)
        
        # Use converted positions for sampling (not original labels!)
        viz_indices = self.smart_sample(
            viz_sample_size, 
            method=sampling_method, 
            include_focus=focus_row_positions if focus_row_positions else None
        )
        print(f"  Selected {len(viz_indices):,} samples for visualization")
        
        self._create_global_visualizations(viz_indices, all_importance, outlier_counts, results)
        
        # Save results
        print("\n" + "=" * 80)
        print("Saving Results")
        print("=" * 80)
        
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved: results.json")
        
        # Create summary report
        self._create_summary_report(results, focus_rows)
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"  • results.json - All data")
        print(f"  • report.txt - Summary report")
        print(f"  • *.png - Global visualizations")
        if focus_rows:
            print(f"  • focus_rows/ - {len(focus_rows)} × 4 detailed visualizations")
        
        return results
    
    def _create_global_visualizations(self, viz_indices, all_importance, outlier_counts, results):
        """Create global visualizations using smart sampling."""
        
        # 1. Pattern heatmap (clustered)
        print("  1. Pattern heatmap (clustered)...")
        fig = self.calc.plot_indicator_heatmap(
            row_indices=list(viz_indices),
            use_importance=False,
            cluster_rows=True,
            cluster_cols=True,
            figsize=(16, 12)
        )
        plt.savefig(self.output_dir / "global_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. PCA projection
        print("  2. PCA projection...")
        fig = self.calc.plot_pattern_pca(
            row_indices=list(viz_indices),
            use_importance=False,
            color_by=outlier_counts[viz_indices],
            figsize=(14, 10)
        )
        plt.savefig(self.output_dir / "global_pca.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Top indicators
        print("  3. Top indicators chart...")
        top_10 = results['top_indicators'][:10]
        names = [d['name'][:30] for d in top_10]
        values = [d['avg_importance'] for d in top_10]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, alpha=0.8, color='steelblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Average Absolute Importance', fontsize=11)
        ax.set_title('Top 10 Most Important Indicators (Global)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.output_dir / "global_top_indicators.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Outlier distribution
        print("  4. Outlier distribution...")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(outlier_counts, bins=50, alpha=0.7, edgecolor='black', color='coral')
        ax.axvline(np.mean(outlier_counts), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(outlier_counts):.2f}')
        ax.axvline(np.median(outlier_counts), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(outlier_counts):.2f}')
        ax.set_xlabel('Number of Outlier Indicators', fontsize=11)
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title('Distribution of Outlier Counts', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "global_outliers.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Created 4 global visualizations")
    
    def _create_summary_report(self, results, focus_rows):
        """Create summary text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("LARGE-SCALE INDICATOR IMPORTANCE ANALYSIS - SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {results['metadata']['timestamp']}")
        lines.append(f"Dataset: {results['metadata']['n_samples']:,} samples × {results['metadata']['n_indicators']} indicators")
        lines.append("")
        
        lines.append("WEIGHT ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Mean: {results['weights']['mean']:.4f}")
        lines.append(f"Std:  {results['weights']['std']:.4f}")
        lines.append(f"Clustered: {'YES' if results['weights']['clustered'] else 'NO'}")
        lines.append("")
        
        lines.append("TOP 10 INDICATORS")
        lines.append("-" * 80)
        for i, ind in enumerate(results['top_indicators'][:10], 1):
            lines.append(f"{i:2d}. {ind['name']:<40} {ind['avg_importance']:.4f}")
        lines.append("")
        
        lines.append("OUTLIER SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total outliers: {results['outliers']['total']:,}")
        lines.append(f"Mean per row: {results['outliers']['mean_per_row']:.2f}")
        lines.append(f"Samples with outliers: {results['outliers']['pct_with_outliers']:.1f}%")
        lines.append("")
        
        if focus_rows and 'focus_analysis' in results:
            lines.append("FOCUS ROWS SUMMARY")
            lines.append("-" * 80)
            for focus in results['focus_analysis'][:10]:
                lines.append(f"\nRow: {focus['row_label']}")
                lines.append(f"  Outliers: {focus['n_outliers']}")
                lines.append(f"  Top 5 indicators:")
                for ind in focus['top_20_indicators'][:5]:
                    lines.append(f"    - {ind['name']}: {ind['score']:.4f}")
            if len(results['focus_analysis']) > 10:
                lines.append(f"\n... and {len(results['focus_analysis']) - 10} more focus rows")
        
        lines.append("")
        lines.append("=" * 80)
        
        with open(self.output_dir / "report.txt", 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Saved: report.txt")
    
    def _create_investigator_reports(self, focus_results, global_results):
        """
        Create detailed HTML reports for investigators.
        
        Creates:
        1. Individual HTML report for each focus row
        2. Master summary table of all focus rows
        3. CSV export of key metrics
        """
        reports_dir = self.output_dir / "investigator_reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Create master summary table
        self._create_master_summary(focus_results, reports_dir)
        
        # Create individual event reports
        for focus in focus_results:
            self._create_event_report(focus, reports_dir)
        
        # Create CSV export
        self._create_csv_export(focus_results, reports_dir)
    
    def _create_master_summary(self, focus_results, reports_dir):
        """Create master HTML summary of all focus rows."""
        html = []
        html.append('<!DOCTYPE html>')
        html.append('<html><head>')
        html.append('<meta charset="UTF-8">')
        html.append('<title>Investigation Summary - All Events</title>')
        html.append('<style>')
        html.append('''
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }
        th { background: #3498db; color: white; padding: 12px; text-align: left; position: sticky; top: 0; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f8f9fa; }
        .outlier-high { background: #ffebee; color: #c62828; font-weight: bold; }
        .outlier-medium { background: #fff3e0; color: #e65100; }
        .outlier-low { background: #e8f5e9; color: #2e7d32; }
        .score-pos { color: #27ae60; }
        .score-neg { color: #e74c3c; }
        .event-link { color: #3498db; text-decoration: none; font-weight: 600; }
        .event-link:hover { text-decoration: underline; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
        .stat-card { background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 32px; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        ''')
        html.append('</style></head><body>')
        html.append('<div class="container">')
        html.append(f'<h1>Investigation Summary - All Events ({len(focus_results)} total)</h1>')
        html.append(f'<p style="color: #7f8c8d;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        
        # Statistics cards
        html.append('<div class="stats">')
        
        total_outliers = sum(f['n_outliers'] for f in focus_results)
        avg_outliers = total_outliers / len(focus_results) if focus_results else 0
        max_outliers = max(f['n_outliers'] for f in focus_results) if focus_results else 0
        
        html.append(f'''
        <div class="stat-card">
            <div class="stat-value">{len(focus_results)}</div>
            <div class="stat-label">Total Events</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_outliers}</div>
            <div class="stat-label">Total Outliers</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_outliers:.1f}</div>
            <div class="stat-label">Avg Outliers/Event</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{max_outliers}</div>
            <div class="stat-label">Max Outliers</div>
        </div>
        ''')
        html.append('</div>')
        
        # Master table
        html.append('<h2>All Events Summary</h2>')
        html.append('<table>')
        html.append('<tr>')
        html.append('<th>Event ID</th>')
        html.append('<th>Outliers</th>')
        html.append('<th>Top Indicator</th>')
        html.append('<th>Top Score</th>')
        html.append('<th>2nd Indicator</th>')
        html.append('<th>3rd Indicator</th>')
        html.append('<th>Details</th>')
        html.append('</tr>')
        
        for focus in sorted(focus_results, key=lambda x: x['n_outliers'], reverse=True):
            event_id = focus['row_label']
            n_outliers = focus['n_outliers']
            
            # Outlier class
            if n_outliers > avg_outliers + 5:
                outlier_class = 'outlier-high'
            elif n_outliers > avg_outliers:
                outlier_class = 'outlier-medium'
            else:
                outlier_class = 'outlier-low'
            
            top_indicators = focus['top_20_indicators'][:3]
            
            html.append('<tr>')
            html.append(f'<td><strong>{event_id}</strong></td>')
            html.append(f'<td class="{outlier_class}">{n_outliers}</td>')
            
            if top_indicators:
                top1 = top_indicators[0]
                score_class = 'score-pos' if top1['score'] > 0 else 'score-neg'
                html.append(f'<td>{top1["name"][:40]}</td>')
                html.append(f'<td class="{score_class}">{top1["score"]:.3f}</td>')
                
                if len(top_indicators) > 1:
                    html.append(f'<td>{top_indicators[1]["name"][:30]}</td>')
                else:
                    html.append('<td>-</td>')
                
                if len(top_indicators) > 2:
                    html.append(f'<td>{top_indicators[2]["name"][:30]}</td>')
                else:
                    html.append('<td>-</td>')
            else:
                html.append('<td colspan="4">-</td>')
            
            html.append(f'<td><a class="event-link" href="event_{event_id}.html">View Details →</a></td>')
            html.append('</tr>')
        
        html.append('</table>')
        html.append('</div></body></html>')
        
        with open(reports_dir / "index.html", 'w') as f:
            f.write('\n'.join(html))
    
    def _create_event_report(self, focus, reports_dir):
        """Create detailed HTML report for a single event/row."""
        event_id = focus['row_label']
        
        html = []
        html.append('<!DOCTYPE html>')
        html.append('<html><head>')
        html.append('<meta charset="UTF-8">')
        html.append(f'<title>Event {event_id} - Investigation Report</title>')
        html.append('<style>')
        html.append('''
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; background: #ecf0f1; padding: 10px; border-left: 4px solid #3498db; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th { background: #34495e; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f8f9fa; }
        .score-pos { color: #27ae60; font-weight: bold; }
        .score-neg { color: #e74c3c; font-weight: bold; }
        .alert { background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .back-link { display: inline-block; margin-bottom: 20px; color: #3498db; text-decoration: none; }
        .back-link:hover { text-decoration: underline; }
        .metric { background: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .viz-gallery { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }
        .viz-item img { width: 100%; border: 1px solid #ddd; border-radius: 5px; }
        .viz-item p { text-align: center; font-weight: bold; color: #34495e; margin-top: 10px; }
        ''')
        html.append('</style></head><body>')
        html.append('<div class="container">')
        html.append('<a class="back-link" href="index.html">← Back to Summary</a>')
        html.append(f'<h1>Event {event_id} - Detailed Investigation Report</h1>')
        
        # Key metrics
        html.append('<div class="metric">')
        html.append(f'<strong>Outlier Indicators:</strong> {focus["n_outliers"]} detected')
        if focus["n_outliers"] > 5:
            html.append(' <span style="color: #e74c3c;">⚠️ HIGH</span>')
        html.append('</div>')
        
        # Outliers section
        if focus['outlier_indicators']:
            html.append('<div class="alert">')
            html.append('<strong>⚠️ Unusual Indicator Behavior Detected</strong>')
            html.append(f'<p>{focus["n_outliers"]} indicators showed unusual values for this event:</p>')
            html.append('<ul>')
            for outlier in focus['outlier_indicators'][:10]:
                html.append(f'<li><strong>{outlier["name"]}</strong>: {outlier["value"]:.3f}</li>')
            html.append('</ul>')
            html.append('</div>')
        
        # Top 20 indicators table
        html.append('<h2>Top 20 Most Important Indicators for This Event</h2>')
        html.append('<table>')
        html.append('<tr><th>Rank</th><th>Indicator Name</th><th>Importance Score</th></tr>')
        
        for i, indicator in enumerate(focus['top_20_indicators'], 1):
            score = indicator['score']
            score_class = 'score-pos' if score > 0 else 'score-neg'
            html.append(f'<tr>')
            html.append(f'<td>{i}</td>')
            html.append(f'<td>{indicator["name"]}</td>')
            html.append(f'<td class="{score_class}">{score:.4f}</td>')
            html.append(f'</tr>')
        
        html.append('</table>')
        
        # Deviation importance
        html.append('<h2>Top 10 Deviation Importance (Unusual Patterns)</h2>')
        html.append('<p>These indicators behaved most differently from typical patterns:</p>')
        html.append('<table>')
        html.append('<tr><th>Rank</th><th>Indicator Name</th><th>Deviation Score</th></tr>')
        
        for i, (name, score) in enumerate(focus['top_10_deviation'], 1):
            score_class = 'score-pos' if score > 0 else 'score-neg'
            html.append(f'<tr>')
            html.append(f'<td>{i}</td>')
            html.append(f'<td>{name}</td>')
            html.append(f'<td class="{score_class}">{score:.4f}</td>')
            html.append(f'</tr>')
        
        html.append('</table>')
        
        # Similar events
        html.append('<h2>10 Most Similar Events</h2>')
        html.append('<p>Events with similar indicator patterns:</p>')
        html.append('<table>')
        html.append('<tr><th>Rank</th><th>Event ID</th><th>Similarity Score</th></tr>')
        
        for i, similar in enumerate(focus['top_10_similar'], 1):
            html.append(f'<tr>')
            html.append(f'<td>{i}</td>')
            html.append(f'<td><a href="event_{similar["label"]}.html">{similar["label"]}</a></td>')
            html.append(f'<td>{similar["score"]:.4f}</td>')
            html.append(f'</tr>')
        
        html.append('</table>')
        
        # Visualizations (if they exist)
        focus_viz_dir = self.output_dir / "focus_rows"
        if focus_viz_dir.exists():
            viz_files = [
                (f"{event_id}_top20.png", "Top 20 Indicators"),
                (f"{event_id}_radar.png", "Indicator Profile (Radar)"),
                (f"{event_id}_comparison.png", "Comparison with Similar Events"),
                (f"{event_id}_heatmap.png", "Indicator Values Heatmap")
            ]
            
            existing_viz = [(f, title) for f, title in viz_files if (focus_viz_dir / f).exists()]
            
            if existing_viz:
                html.append('<h2>Visualizations</h2>')
                html.append('<div class="viz-gallery">')
                for filename, title in existing_viz:
                    html.append('<div class="viz-item">')
                    html.append(f'<img src="../focus_rows/{filename}" alt="{title}">')
                    html.append(f'<p>{title}</p>')
                    html.append('</div>')
                html.append('</div>')
        
        html.append('<br><a class="back-link" href="index.html">← Back to Summary</a>')
        html.append('</div></body></html>')
        
        filename = f"event_{event_id}.html".replace('/', '_')
        with open(reports_dir / filename, 'w') as f:
            f.write('\n'.join(html))
    
    def _create_csv_export(self, focus_results, reports_dir):
        """Export focus results to CSV for further analysis."""
        import csv
        
        # Main summary CSV
        with open(reports_dir / "events_summary.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Event_ID', 'Outlier_Count',
                'Top1_Indicator', 'Top1_Score',
                'Top2_Indicator', 'Top2_Score',
                'Top3_Indicator', 'Top3_Score'
            ])
            
            for focus in focus_results:
                row = [focus['row_label'], focus['n_outliers']]
                
                for i in range(3):
                    if i < len(focus['top_20_indicators']):
                        ind = focus['top_20_indicators'][i]
                        row.extend([ind['name'], ind['score']])
                    else:
                        row.extend(['', ''])
                
                writer.writerow(row)
        
        # Detailed indicators CSV
        with open(reports_dir / "events_detailed.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Event_ID', 'Rank', 'Indicator_Name', 'Importance_Score'])
            
            for focus in focus_results:
                for i, ind in enumerate(focus['top_20_indicators'], 1):
                    writer.writerow([
                        focus['row_label'],
                        i,
                        ind['name'],
                        ind['score']
                    ])
        
        print(f"    ✓ CSV exports: events_summary.csv, events_detailed.csv")


# Simple function interface
def analyze_large_dataset(
    indicator_matrix: Union[np.ndarray, pd.DataFrame],
    snorkel_weights: np.ndarray,
    indicator_names: List[str],
    focus_rows: Optional[List] = None,
    focus_rows_are_labels: bool = True,
    pandas_index: Optional[np.ndarray] = None,
    output_dir: str = "./large_scale_analysis",
    viz_sample_size: int = 5000,
    sampling_method: str = 'clustering',
    create_focus_visualizations: bool = True,
    create_investigator_reports: bool = True
) -> dict:
    """
    Analyze large dataset with focus on specific rows.
    
    Simple function interface for quick analysis.
    
    Parameters:
    -----------
    indicator_matrix : np.ndarray or pd.DataFrame
        Matrix of indicators/labeling functions
    snorkel_weights : np.ndarray
        Learned weights from Snorkel
    indicator_names : list of str
        Names of indicators
    focus_rows : list, optional
        Row identifiers (pandas index values or positions)
    focus_rows_are_labels : bool, default=True
        If True, focus_rows are pandas index values (e.g., [3742123, 4526073])
        If False, focus_rows are positional indices (e.g., [0, 1, 2])
    pandas_index : np.ndarray, optional
        Original DataFrame index
    output_dir : str
        Output directory
    viz_sample_size : int
        Sample size for global visualizations
    sampling_method : str
        'clustering', 'stratified', or 'random'
    create_focus_visualizations : bool, default=True
        Create 4 visualizations per focus row (set False for 1000+ rows)
    create_investigator_reports : bool, default=True
        Create detailed HTML reports for investigators
    
    Returns:
    --------
    dict : Complete results
    """
    analysis = LargeScaleAnalysis(
        indicator_matrix=indicator_matrix,
        snorkel_weights=snorkel_weights,
        indicator_names=indicator_names,
        pandas_index=pandas_index,
        output_dir=output_dir
    )
    
    return analysis.run_full_analysis(
        focus_rows=focus_rows,
        focus_rows_are_labels=focus_rows_are_labels,
        viz_sample_size=viz_sample_size,
        sampling_method=sampling_method,
        create_focus_visualizations=create_focus_visualizations,
        create_investigator_reports=create_investigator_reports
    )


if __name__ == "__main__":
    # Example with synthetic large dataset
    print("Testing on synthetic large dataset...\n")
    
    np.random.seed(42)
    n_samples = 50000
    n_indicators = 100
    
    # Create data
    L_matrix = np.random.choice([-1, 0, 1], size=(n_samples, n_indicators))
    weights = np.random.randn(n_indicators) * 0.5
    lf_names = [f"LF_{i}" for i in range(n_indicators)]
    
    # Run analysis
    results = analyze_large_dataset(
        indicator_matrix=L_matrix,
        snorkel_weights=weights,
        indicator_names=lf_names,
        focus_rows=[0, 100, 1000, 5000],
        viz_sample_size=1000,
        sampling_method='clustering'
    )
    
    print("\n✅ Test complete!")
