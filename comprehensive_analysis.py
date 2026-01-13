"""
COMPREHENSIVE INDICATOR IMPORTANCE ANALYSIS
All-in-one script for complete Snorkel indicator analysis with automatic
weight clustering detection and adaptive methods.

Usage:
    python comprehensive_analysis.py

Or import and use:
    from comprehensive_analysis import run_full_analysis
    results = run_full_analysis(indicator_matrix, snorkel_weights, indicator_names)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import sys

# Import the indicator importance library
sys.path.append('.')
from indicator_importance import SnorkelIndicatorImportance


class ComprehensiveAnalysis:
    """
    Complete analysis pipeline for Snorkel indicator importance.
    
    Automatically detects weight clustering and adapts analysis methods.
    Generates all visualizations and a comprehensive report.
    """
    
    def __init__(
        self,
        indicator_matrix: np.ndarray,
        snorkel_weights: np.ndarray,
        indicator_names: list,
        output_dir: str = "./analysis_output",
        sample_name: str = "analysis"
    ):
        """
        Initialize comprehensive analysis.
        
        Parameters:
        -----------
        indicator_matrix : np.ndarray
            Matrix of shape (n_samples, n_indicators)
        snorkel_weights : np.ndarray
            Learned weights from Snorkel
        indicator_names : list
            Names of indicators/labeling functions
        output_dir : str
            Directory to save outputs
        sample_name : str
            Name for this analysis (used in filenames)
        """
        self.indicator_matrix = indicator_matrix
        self.snorkel_weights = snorkel_weights
        self.indicator_names = indicator_names
        self.output_dir = Path(output_dir)
        self.sample_name = sample_name
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize calculator
        self.calc = SnorkelIndicatorImportance(
            indicator_matrix=indicator_matrix,
            snorkel_weights=snorkel_weights,
            indicator_names=indicator_names
        )
        
        # Results storage
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_samples': indicator_matrix.shape[0],
                'n_indicators': indicator_matrix.shape[1],
                'sample_name': sample_name
            },
            'weight_analysis': {},
            'similarity_analysis': {},
            'top_indicators': {},
            'outliers': {},
            'diversity': {},
            'visualizations': []
        }
        
        print("=" * 80)
        print(f"COMPREHENSIVE INDICATOR IMPORTANCE ANALYSIS: {sample_name}")
        print("=" * 80)
        print(f"Dataset: {indicator_matrix.shape[0]:,} samples × {indicator_matrix.shape[1]} indicators")
        print(f"Output directory: {self.output_dir}")
        print()
    
    def run_all(
        self, 
        max_viz_samples: int = 100,
        skip_diversity: bool = False,
        skip_similarity: bool = False,
        skip_viz: bool = False,
        diversity_sample_size: int = 100
    ):
        """
        Run complete analysis pipeline.
        
        Parameters:
        -----------
        max_viz_samples : int
            Maximum samples to include in visualizations (for performance)
        skip_diversity : bool, default=False
            Skip diversity analysis (slow for large datasets)
        skip_similarity : bool, default=False
            Skip similarity analysis (slow for large datasets)
        skip_viz : bool, default=False
            Skip all visualizations (for very large datasets)
        diversity_sample_size : int, default=100
            Sample size for diversity calculation (reduced for large datasets)
        """
        print("Starting comprehensive analysis...\n")
        
        # Detect large dataset
        is_large = self.indicator_matrix.shape[0] > 100000
        if is_large:
            print("⚡ LARGE DATASET DETECTED")
            print(f"   {self.indicator_matrix.shape[0]:,} samples - using optimized settings")
            print(f"   Skipping expensive operations by default\n")
            skip_diversity = skip_diversity or is_large
            skip_similarity = skip_similarity or (self.indicator_matrix.shape[0] > 500000)
        
        # Step 1: Analyze weights
        self._analyze_weights()
        
        # Step 2: Compute all importance scores
        self._compute_importance_scores()
        
        # Step 3: Detect outliers
        self._analyze_outliers()
        
        # Step 4: Analyze top indicators
        self._analyze_top_indicators()
        
        # Step 5: Compute diversity (OPTIONAL - can be slow)
        if not skip_diversity:
            self._analyze_diversity(sample_size=diversity_sample_size)
        else:
            print("=" * 80)
            print("STEP 5: Instance Diversity Analysis")
            print("=" * 80)
            print("⚡ SKIPPED (use skip_diversity=False to enable)")
            print("   For large datasets, this step is slow and optional\n")
            self.results['diversity'] = {'skipped': True}
        
        # Step 6: Similarity analysis (OPTIONAL - can be slow)
        if not skip_similarity:
            self._analyze_similarity()
        else:
            print("=" * 80)
            print("STEP 6: Similarity Analysis")
            print("=" * 80)
            print("⚡ SKIPPED (use skip_similarity=False to enable)")
            print("   For large datasets, this step is slow and optional\n")
            self.results['similarity_analysis'] = {'skipped': True}
        
        # Step 7: Generate visualizations (OPTIONAL)
        if not skip_viz:
            self._generate_visualizations(max_samples=max_viz_samples)
        else:
            print("=" * 80)
            print("STEP 7: Visualizations")
            print("=" * 80)
            print("⚡ SKIPPED (use skip_viz=False to enable)\n")
            self.results['visualizations'] = []
        
        # Step 8: Create summary report
        self._create_report()
        
        # Step 9: Save results
        self._save_results()
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"  • Report: {self.sample_name}_report.txt")
        print(f"  • Data: {self.sample_name}_results.json")
        if not skip_viz:
            print(f"  • Visualizations: {len(self.results['visualizations'])} PNG files")
        
        return self.results
    
    def _analyze_weights(self):
        """Analyze weight distribution and detect clustering."""
        print("=" * 80)
        print("STEP 1: Weight Analysis")
        print("=" * 80)
        
        weights = self.snorkel_weights
        
        stats = {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'range': float(np.max(weights) - np.min(weights)),
            'median': float(np.median(weights)),
            'q25': float(np.percentile(weights, 25)),
            'q75': float(np.percentile(weights, 75))
        }
        
        # Detect clustering
        stats['is_clustered'] = stats['std'] < 0.2
        stats['clustering_severity'] = 'high' if stats['std'] < 0.1 else 'medium' if stats['std'] < 0.2 else 'low'
        
        self.results['weight_analysis'] = stats
        
        print(f"Weight Statistics:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
        print()
        
        if stats['is_clustered']:
            print(f"⚠️  WEIGHT CLUSTERING DETECTED (std = {stats['std']:.4f})")
            print(f"   Severity: {stats['clustering_severity'].upper()}")
            print(f"   → Will use pattern-based and combined methods")
            self.similarity_method = 'combined'
        else:
            print(f"✓ Weights are diverse (std = {stats['std']:.4f})")
            print(f"   → Standard cosine similarity will work well")
            self.similarity_method = 'cosine'
        
        print()
    
    def _compute_importance_scores(self):
        """Compute all importance scores."""
        print("=" * 80)
        print("STEP 2: Computing Importance Scores")
        print("=" * 80)
        
        # Standard importance
        print("Computing standard importance scores...")
        all_importance = self.calc.get_all_importance_scores(normalize=True)
        
        # Deviation importance
        print("Computing deviation-based importance...")
        deviation_importance = self.calc.compute_deviation_importance(normalize=True)
        
        # Global statistics
        mean_importance = np.mean(np.abs(all_importance), axis=0)
        std_importance = np.std(np.abs(all_importance), axis=0)
        
        self.results['importance_scores'] = {
            'standard_mean': mean_importance.tolist(),
            'standard_std': std_importance.tolist(),
            'has_deviation': True
        }
        
        # Store for later use
        self.all_importance = all_importance
        self.deviation_importance = deviation_importance
        
        print(f"✓ Computed importance for {all_importance.shape[0]:,} samples")
        print()
    
    def _analyze_outliers(self):
        """Detect and analyze outliers."""
        print("=" * 80)
        print("STEP 3: Outlier Detection")
        print("=" * 80)
        
        print("Detecting outliers across all samples...")
        all_outliers = self.calc.detect_outliers_vectorized()
        outlier_counts = np.sum(all_outliers, axis=1)
        
        stats = {
            'total_outliers': int(np.sum(all_outliers)),
            'mean_per_sample': float(np.mean(outlier_counts)),
            'std_per_sample': float(np.std(outlier_counts)),
            'max_per_sample': int(np.max(outlier_counts)),
            'samples_with_outliers': int(np.sum(outlier_counts > 0)),
            'pct_samples_with_outliers': float(100 * np.sum(outlier_counts > 0) / len(outlier_counts))
        }
        
        # Find most problematic samples
        top_outlier_indices = np.argsort(outlier_counts)[-10:][::-1]
        stats['top_outlier_samples'] = [
            {'index': int(idx), 'count': int(outlier_counts[idx])}
            for idx in top_outlier_indices
        ]
        
        # Outlier frequency by indicator
        outlier_freq_by_indicator = np.sum(all_outliers, axis=0)
        top_outlier_indicators = np.argsort(outlier_freq_by_indicator)[-10:][::-1]
        stats['top_outlier_indicators'] = [
            {'name': self.indicator_names[idx], 'frequency': int(outlier_freq_by_indicator[idx])}
            for idx in top_outlier_indicators
        ]
        
        self.results['outliers'] = stats
        self.outlier_counts = outlier_counts
        
        print(f"✓ Outlier Detection Complete:")
        print(f"  Total outliers: {stats['total_outliers']:,}")
        print(f"  Avg per sample: {stats['mean_per_sample']:.2f}")
        print(f"  Samples with outliers: {stats['samples_with_outliers']:,} ({stats['pct_samples_with_outliers']:.1f}%)")
        print(f"\n  Top 5 most problematic samples:")
        for item in stats['top_outlier_samples'][:5]:
            print(f"    Sample {item['index']}: {item['count']} outliers")
        print()
    
    def _analyze_top_indicators(self):
        """Analyze top indicators globally."""
        print("=" * 80)
        print("STEP 4: Top Indicator Analysis")
        print("=" * 80)
        
        print("Computing global indicator importance...")
        
        # Global importance (averaged across all samples)
        global_importance = np.mean(np.abs(self.all_importance), axis=0)
        top_indices = np.argsort(global_importance)[-20:][::-1]
        
        top_indicators = []
        for rank, idx in enumerate(top_indices, 1):
            top_indicators.append({
                'rank': rank,
                'name': self.indicator_names[idx],
                'avg_importance': float(global_importance[idx]),
                'weight': float(self.snorkel_weights[idx]),
                'std_importance': float(np.std(np.abs(self.all_importance[:, idx])))
            })
        
        self.results['top_indicators'] = {
            'global_top_20': top_indicators,
            'method': 'mean_absolute_importance'
        }
        
        print(f"✓ Top 10 Most Important Indicators (globally):")
        print(f"{'Rank':<6} {'Name':<30} {'Avg Importance':<16} {'Weight':<10}")
        print("-" * 65)
        for item in top_indicators[:10]:
            print(f"{item['rank']:<6} {item['name']:<30} {item['avg_importance']:>14.4f}  {item['weight']:>8.3f}")
        print()
    
    def _analyze_diversity(self, sample_size: int = 100):
        """Analyze instance diversity."""
        print("=" * 80)
        print("STEP 5: Instance Diversity Analysis")
        print("=" * 80)
        
        print(f"Computing instance diversity scores (sample_size={sample_size})...")
        
        # Sample for diversity calculation (much smaller for large datasets)
        analysis_sample_size = min(sample_size, self.indicator_matrix.shape[0])
        sample_indices = np.random.choice(
            self.indicator_matrix.shape[0],
            size=analysis_sample_size,
            replace=False
        )
        
        diversity_scores = self.calc.compute_instance_diversity(
            row_indices=list(sample_indices),
            sample_size=min(500, self.indicator_matrix.shape[0])  # Compare against max 500
        )
        
        stats = {
            'mean_diversity': float(np.mean(diversity_scores)),
            'std_diversity': float(np.std(diversity_scores)),
            'min_diversity': float(np.min(diversity_scores)),
            'max_diversity': float(np.max(diversity_scores)),
            'sample_size': analysis_sample_size
        }
        
        # Find most/least diverse samples
        most_diverse_idx = sample_indices[np.argsort(diversity_scores)[-5:][::-1]]
        least_diverse_idx = sample_indices[np.argsort(diversity_scores)[:5]]
        
        stats['most_diverse_samples'] = [int(idx) for idx in most_diverse_idx]
        stats['least_diverse_samples'] = [int(idx) for idx in least_diverse_idx]
        
        self.results['diversity'] = stats
        
        print(f"✓ Diversity Analysis (on {analysis_sample_size:,} samples):")
        print(f"  Mean diversity: {stats['mean_diversity']:.4f}")
        print(f"  Range: [{stats['min_diversity']:.4f}, {stats['max_diversity']:.4f}]")
        print()
    
    def _analyze_similarity(self):
        """Analyze instance similarity with multiple methods."""
        print("=" * 80)
        print("STEP 6: Similarity Analysis")
        print("=" * 80)
        
        # Pick a few representative samples to analyze
        sample_indices = [0, 10, 50, 100, 200]
        sample_indices = [i for i in sample_indices if i < self.indicator_matrix.shape[0]]
        
        similarity_results = []
        
        print(f"Analyzing similarity for {len(sample_indices)} representative samples...")
        print(f"Using method: {self.similarity_method}")
        print()
        
        for target_idx in sample_indices:
            similar = self.calc.find_similar_instances(
                target_idx=target_idx,
                top_k=10,
                method=self.similarity_method
            )
            
            # Get scores
            scores = [score for _, score in similar]
            
            result = {
                'target_index': target_idx,
                'method': self.similarity_method,
                'top_10_similar': [
                    {'index': int(idx), 'score': float(score)}
                    for idx, score in similar
                ],
                'score_stats': {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'range': float(np.max(scores) - np.min(scores))
                }
            }
            
            similarity_results.append(result)
            
            print(f"  Sample {target_idx}: similarity std = {result['score_stats']['std']:.4f}")
        
        self.results['similarity_analysis'] = {
            'method_used': self.similarity_method,
            'samples_analyzed': similarity_results
        }
        
        print()
    
    def _generate_visualizations(self, max_samples: int = 100):
        """Generate all visualizations."""
        print("=" * 80)
        print("STEP 7: Generating Visualizations")
        print("=" * 80)
        
        viz_samples = min(max_samples, self.indicator_matrix.shape[0])
        
        # 1. Weight distribution
        print("1. Creating weight distribution plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.snorkel_weights, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(self.snorkel_weights), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.snorkel_weights):.3f}')
        ax.axvline(np.median(self.snorkel_weights), color='green', linestyle='--',
                   label=f'Median: {np.median(self.snorkel_weights):.3f}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Snorkel Weight Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = self.output_dir / f"{self.sample_name}_weight_distribution.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        self.results['visualizations'].append(str(fname))
        print(f"   ✓ Saved: {fname.name}")
        
        # 2. Global importance bar chart
        print("2. Creating global importance chart...")
        top_10_data = self.results['top_indicators']['global_top_20'][:10]
        names = [d['name'][:25] for d in top_10_data]
        values = [d['avg_importance'] for d in top_10_data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Average Absolute Importance')
        ax.set_title('Top 10 Most Important Indicators (Global)')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        fname = self.output_dir / f"{self.sample_name}_top_indicators.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        self.results['visualizations'].append(str(fname))
        print(f"   ✓ Saved: {fname.name}")
        
        # 3. Outlier distribution
        print("3. Creating outlier distribution plot...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.outlier_counts, bins=30, alpha=0.7, edgecolor='black', color='coral')
        ax.axvline(np.mean(self.outlier_counts), color='red', linestyle='--',
                   label=f'Mean: {np.mean(self.outlier_counts):.2f}')
        ax.set_xlabel('Number of Outlier Indicators')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Distribution of Outlier Counts per Sample')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = self.output_dir / f"{self.sample_name}_outlier_distribution.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        self.results['visualizations'].append(str(fname))
        print(f"   ✓ Saved: {fname.name}")
        
        # 4. Pattern heatmap
        print(f"4. Creating pattern heatmap (first {viz_samples} samples)...")
        fig = self.calc.plot_indicator_heatmap(
            row_indices=list(range(viz_samples)),
            use_importance=(not self.results['weight_analysis']['is_clustered']),
            cluster_rows=True,
            cluster_cols=True,
            figsize=(14, 10)
        )
        fname = self.output_dir / f"{self.sample_name}_pattern_heatmap.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        self.results['visualizations'].append(str(fname))
        print(f"   ✓ Saved: {fname.name}")
        
        # 5. PCA projection
        print(f"5. Creating PCA projection...")
        fig = self.calc.plot_pattern_pca(
            row_indices=list(range(min(500, self.indicator_matrix.shape[0]))),
            use_importance=(not self.results['weight_analysis']['is_clustered']),
            color_by=self.outlier_counts[:min(500, self.indicator_matrix.shape[0])],
            figsize=(12, 9)
        )
        fname = self.output_dir / f"{self.sample_name}_pca_projection.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        self.results['visualizations'].append(str(fname))
        print(f"   ✓ Saved: {fname.name}")
        
        # 6. Similarity matrix
        if viz_samples <= 100:
            print(f"6. Creating similarity matrix ({viz_samples} samples)...")
            fig = self.calc.plot_similarity_matrix(
                row_indices=list(range(viz_samples)),
                method=self.similarity_method,
                cluster=True,
                figsize=(10, 8)
            )
            fname = self.output_dir / f"{self.sample_name}_similarity_matrix.png"
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()
            self.results['visualizations'].append(str(fname))
            print(f"   ✓ Saved: {fname.name}")
        else:
            print(f"6. Skipping similarity matrix (too many samples: {viz_samples})")
        
        # 7. Instance comparison (first few samples)
        print("7. Creating instance comparison...")
        comparison_indices = list(range(min(4, self.indicator_matrix.shape[0])))
        fig = self.calc.plot_instance_comparison(
            row_indices=comparison_indices,
            top_k=10,
            figsize=(14, 6)
        )
        fname = self.output_dir / f"{self.sample_name}_instance_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        self.results['visualizations'].append(str(fname))
        print(f"   ✓ Saved: {fname.name}")
        
        # 8. Radar chart for first sample
        print("8. Creating radar chart (sample 0)...")
        fig = self.calc.plot_indicator_radar(
            row_idx=0,
            top_k=10,
            figsize=(10, 10)
        )
        fname = self.output_dir / f"{self.sample_name}_radar_sample_0.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        self.results['visualizations'].append(str(fname))
        print(f"   ✓ Saved: {fname.name}")
        
        print(f"\n✓ Generated {len(self.results['visualizations'])} visualizations")
        print()
    
    def _create_report(self):
        """Create comprehensive text report."""
        print("=" * 80)
        print("STEP 8: Creating Summary Report")
        print("=" * 80)
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE INDICATOR IMPORTANCE ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {self.results['metadata']['timestamp']}")
        report_lines.append(f"Sample: {self.results['metadata']['sample_name']}")
        report_lines.append(f"Dataset: {self.results['metadata']['n_samples']:,} samples × {self.results['metadata']['n_indicators']} indicators")
        report_lines.append("")
        
        # Weight Analysis
        report_lines.append("=" * 80)
        report_lines.append("1. WEIGHT ANALYSIS")
        report_lines.append("=" * 80)
        wa = self.results['weight_analysis']
        report_lines.append(f"Mean:   {wa['mean']:.4f}")
        report_lines.append(f"Std:    {wa['std']:.4f}")
        report_lines.append(f"Median: {wa['median']:.4f}")
        report_lines.append(f"Range:  [{wa['min']:.4f}, {wa['max']:.4f}]")
        report_lines.append("")
        
        if wa['is_clustered']:
            report_lines.append(f"⚠️  WEIGHT CLUSTERING DETECTED")
            report_lines.append(f"   Severity: {wa['clustering_severity'].upper()}")
            report_lines.append(f"   Recommendation: Use pattern-based or combined similarity methods")
        else:
            report_lines.append(f"✓ Weights are diverse - standard methods work well")
        report_lines.append("")
        
        # Top Indicators
        report_lines.append("=" * 80)
        report_lines.append("2. TOP 10 MOST IMPORTANT INDICATORS")
        report_lines.append("=" * 80)
        report_lines.append(f"{'Rank':<6} {'Name':<35} {'Avg Importance':<16} {'Weight':<10}")
        report_lines.append("-" * 70)
        for item in self.results['top_indicators']['global_top_20'][:10]:
            report_lines.append(
                f"{item['rank']:<6} {item['name']:<35} {item['avg_importance']:>14.4f}  {item['weight']:>8.3f}"
            )
        report_lines.append("")
        
        # Outliers
        report_lines.append("=" * 80)
        report_lines.append("3. OUTLIER ANALYSIS")
        report_lines.append("=" * 80)
        ol = self.results['outliers']
        report_lines.append(f"Total outliers detected: {ol['total_outliers']:,}")
        report_lines.append(f"Average per sample: {ol['mean_per_sample']:.2f}")
        report_lines.append(f"Samples with outliers: {ol['samples_with_outliers']:,} ({ol['pct_samples_with_outliers']:.1f}%)")
        report_lines.append("")
        report_lines.append("Top 5 Most Problematic Samples:")
        for item in ol['top_outlier_samples'][:5]:
            report_lines.append(f"  Sample {item['index']}: {item['count']} outliers")
        report_lines.append("")
        report_lines.append("Top 5 Indicators with Most Outliers:")
        for item in ol['top_outlier_indicators'][:5]:
            report_lines.append(f"  {item['name']}: {item['frequency']} times")
        report_lines.append("")
        
        # Diversity
        report_lines.append("=" * 80)
        report_lines.append("4. INSTANCE DIVERSITY")
        report_lines.append("=" * 80)
        div = self.results['diversity']
        report_lines.append(f"Mean diversity score: {div['mean_diversity']:.4f}")
        report_lines.append(f"Range: [{div['min_diversity']:.4f}, {div['max_diversity']:.4f}]")
        report_lines.append(f"(Based on {div['sample_size']:,} samples)")
        report_lines.append("")
        
        # Similarity
        report_lines.append("=" * 80)
        report_lines.append("5. SIMILARITY ANALYSIS")
        report_lines.append("=" * 80)
        sim = self.results['similarity_analysis']
        report_lines.append(f"Method used: {sim['method_used']}")
        report_lines.append("")
        report_lines.append("Similarity score statistics for representative samples:")
        for sample in sim['samples_analyzed'][:5]:
            stats = sample['score_stats']
            report_lines.append(
                f"  Sample {sample['target_index']}: "
                f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, range={stats['range']:.4f}"
            )
        report_lines.append("")
        
        # Recommendations
        report_lines.append("=" * 80)
        report_lines.append("6. RECOMMENDATIONS")
        report_lines.append("=" * 80)
        
        if wa['is_clustered']:
            report_lines.append("Based on weight clustering:")
            report_lines.append("  1. Use 'combined' or 'hamming' method for find_similar_instances()")
            report_lines.append("  2. Focus on pattern-based analysis (which indicators fired)")
            report_lines.append("  3. Use deviation importance for outlier detection")
            report_lines.append("  4. Review pattern heatmap for natural groupings")
        else:
            report_lines.append("Based on weight distribution:")
            report_lines.append("  1. Standard cosine similarity is appropriate")
            report_lines.append("  2. Importance-weighted analysis is effective")
            report_lines.append("  3. Consider top indicators for feature engineering")
        
        report_lines.append("")
        report_lines.append(f"High outlier samples ({len([s for s in ol['top_outlier_samples'] if s['count'] > ol['mean_per_sample'] + 2*ol['std_per_sample']])} found):")
        report_lines.append("  • Review these samples manually")
        report_lines.append("  • Check for data quality issues")
        report_lines.append("  • May indicate edge cases worth investigating")
        report_lines.append("")
        
        # Files generated
        report_lines.append("=" * 80)
        report_lines.append("7. OUTPUT FILES")
        report_lines.append("=" * 80)
        report_lines.append(f"Report: {self.sample_name}_report.txt")
        report_lines.append(f"Data: {self.sample_name}_results.json")
        report_lines.append(f"\nVisualizations ({len(self.results['visualizations'])} files):")
        for viz_path in self.results['visualizations']:
            report_lines.append(f"  • {Path(viz_path).name}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Save report
        report_path = self.output_dir / f"{self.sample_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.report_text = '\n'.join(report_lines)
        
        print(f"✓ Report created: {report_path.name}")
        print()
    
    def _save_results(self):
        """Save results to JSON."""
        print("=" * 80)
        print("STEP 9: Saving Results")
        print("=" * 80)
        
        # Save JSON results
        results_path = self.output_dir / f"{self.sample_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ Results saved: {results_path.name}")
        print()
    
    def print_report(self):
        """Print the report to console."""
        if hasattr(self, 'report_text'):
            print(self.report_text)


def run_full_analysis(
    indicator_matrix: np.ndarray,
    snorkel_weights: np.ndarray,
    indicator_names: list,
    output_dir: str = "./analysis_output",
    sample_name: str = "analysis",
    max_viz_samples: int = 100,
    skip_diversity: bool = None,  # Auto-detect based on size
    skip_similarity: bool = None,  # Auto-detect based on size
    skip_viz: bool = False,
    diversity_sample_size: int = 100
) -> dict:
    """
    Run complete indicator importance analysis.
    
    Parameters:
    -----------
    indicator_matrix : np.ndarray
        Matrix of shape (n_samples, n_indicators)
    snorkel_weights : np.ndarray
        Learned weights from Snorkel
    indicator_names : list
        Names of indicators/labeling functions
    output_dir : str
        Directory to save outputs
    sample_name : str
        Name for this analysis
    max_viz_samples : int, default=100
        Maximum samples in visualizations
    skip_diversity : bool, optional
        Skip diversity analysis. If None, auto-detects based on dataset size.
    skip_similarity : bool, optional
        Skip similarity analysis. If None, auto-detects based on dataset size.
    skip_viz : bool, default=False
        Skip all visualizations
    diversity_sample_size : int, default=100
        Sample size for diversity calculation (smaller = faster)
    
    Returns:
    --------
    dict : Complete analysis results
    """
    # Auto-detect skipping based on size
    n_samples = indicator_matrix.shape[0]
    if skip_diversity is None:
        skip_diversity = n_samples > 100000
    if skip_similarity is None:
        skip_similarity = n_samples > 500000
    
    if n_samples > 1000000:
        print(f"⚡ VERY LARGE DATASET: {n_samples:,} samples")
        print(f"   Automatically skipping: diversity={skip_diversity}, similarity={skip_similarity}")
        print()
    
    analysis = ComprehensiveAnalysis(
        indicator_matrix=indicator_matrix,
        snorkel_weights=snorkel_weights,
        indicator_names=indicator_names,
        output_dir=output_dir,
        sample_name=sample_name
    )
    
    results = analysis.run_all(
        max_viz_samples=max_viz_samples,
        skip_diversity=skip_diversity,
        skip_similarity=skip_similarity,
        skip_viz=skip_viz,
        diversity_sample_size=diversity_sample_size
    )
    
    # Print report to console
    print("\n" * 2)
    analysis.print_report()
    
    return results


# Example usage
if __name__ == "__main__":
    print("Running comprehensive analysis on synthetic data...\n")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_indicators = 50
    
    # Indicator matrix
    indicator_matrix = np.random.choice(
        [-1, 0, 1],
        size=(n_samples, n_indicators),
        p=[0.2, 0.5, 0.3]
    )
    
    # Add some structure
    for i in range(0, n_samples, 20):
        indicator_matrix[i:i+10, 0:10] = 1
        indicator_matrix[i+10:i+20, 20:30] = -1
    
    # Snorkel weights (clustered for demonstration)
    snorkel_weights = np.random.normal(0.5, 0.15, n_indicators)
    snorkel_weights[0] = 2.0
    snorkel_weights[10] = -1.5
    
    # Indicator names
    indicator_names = [f"LF_{i}_{np.random.choice(['accuracy', 'precision', 'coverage'])}" 
                      for i in range(n_indicators)]
    
    # Run full analysis
    results = run_full_analysis(
        indicator_matrix=indicator_matrix,
        snorkel_weights=snorkel_weights,
        indicator_names=indicator_names,
        output_dir="./comprehensive_analysis_output",
        sample_name="synthetic_demo",
        max_viz_samples=100
    )
    
    print("\n✅ Demo complete! Check the 'comprehensive_analysis_output' directory.")
