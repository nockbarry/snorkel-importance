"""
SHAP + XGBoost Model Explainer

Efficient model explanation for large datasets (5M rows × 1000 features).

Usage:
    from xgb_explainer import explain_xgb_events
    
    results = explain_xgb_events(
        model=xgb_model,
        X=X_data,
        feature_names=feature_names,
        event_ids=[3947355, 4526073, ...],  # Your 3000 events
        events_are_labels=True,
        event_index=df.index.values,
        y=y_true,
        output_dir="./xgb_explanations"
    )
"""

import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Optional, List, Union
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class XGBExplainer:
    """Fast XGBoost + SHAP explainer for large datasets."""
    
    def __init__(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
        event_ids: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ):
        """Initialize explainer with model and data."""
        # Handle DataFrame
        if isinstance(X, pd.DataFrame):
            if event_ids is None:
                event_ids = X.index.values
            X = X.values
        
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.event_ids = event_ids
        self.y = y
        
        # Index lookup
        if self.event_ids is not None:
            self.id_to_position = {eid: pos for pos, eid in enumerate(self.event_ids)}
        else:
            self.id_to_position = {}
        
        print(f"Computing predictions for {len(X):,} samples...")
        self.predictions = self.model.predict(X)
        if hasattr(self.model, 'predict_proba'):
            self.pred_proba = self.model.predict_proba(X)
        else:
            self.pred_proba = None
        
        print("Initializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        print(f"✓ Initialized: {X.shape[0]:,} × {X.shape[1]} features")
    
    def get_event_label(self, position: int) -> str:
        """Get label for an event."""
        return str(self.event_ids[position]) if self.event_ids is not None else str(position)
    
    def convert_labels_to_positions(self, labels: List) -> List[int]:
        """Convert event IDs to positions."""
        if not self.id_to_position:
            return labels
        
        positions = []
        not_found = []
        for label in labels:
            if label in self.id_to_position:
                positions.append(self.id_to_position[label])
            else:
                not_found.append(label)
        
        if not_found:
            print(f"⚠️  {len(not_found)} event IDs not found")
        
        return positions
    
    def get_global_stats(self, sample_size: int = 10000) -> dict:
        """Compute global model statistics (sampled for speed)."""
        print("\nComputing global statistics...")
        
        stats = {
            'n_samples': int(self.X.shape[0]),
            'n_features': int(self.X.shape[1])
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            importance = np.zeros(len(self.feature_names))
        
        importance = importance / importance.sum() if importance.sum() > 0 else importance
        top_20_idx = np.argsort(importance)[-20:][::-1]
        
        stats['top_20_features'] = [
            {'name': self.feature_names[idx], 'importance': float(importance[idx])}
            for idx in top_20_idx
        ]
        
        # Performance metrics
        if self.y is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            stats['performance'] = {'accuracy': float(accuracy_score(self.y, self.predictions))}
            try:
                stats['performance'].update({
                    'precision': float(precision_score(self.y, self.predictions, average='weighted')),
                    'recall': float(recall_score(self.y, self.predictions, average='weighted')),
                    'f1': float(f1_score(self.y, self.predictions, average='weighted'))
                })
            except:
                pass
        
        # SHAP importance (sampled)
        sample_idx = np.random.choice(len(self.X), size=min(sample_size, len(self.X)), replace=False)
        shap_values = self.explainer.shap_values(self.X[sample_idx])
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_20_shap_idx = np.argsort(mean_abs_shap)[-20:][::-1]
        
        stats['top_20_shap_features'] = [
            {'name': self.feature_names[idx], 'mean_abs_shap': float(mean_abs_shap[idx])}
            for idx in top_20_shap_idx
        ]
        
        print("✓ Global statistics computed")
        return stats
    
    def explain_event(self, position: int) -> dict:
        """Get SHAP explanation for single event."""
        shap_values = self.explainer.shap_values(self.X[position:position+1])[0]
        abs_shap = np.abs(shap_values)
        top_20_idx = np.argsort(abs_shap)[-20:][::-1]
        
        explanation = {
            'event_id': self.get_event_label(position),
            'position': position,
            'prediction': float(self.predictions[position]),
            'top_20_features': [
                {
                    'name': self.feature_names[idx],
                    'value': float(self.X[position, idx]),
                    'shap_value': float(shap_values[idx]),
                    'abs_shap': float(abs_shap[idx])
                }
                for idx in top_20_idx
            ]
        }
        
        if self.pred_proba is not None:
            explanation['pred_proba'] = float(self.pred_proba[position].max())
        
        if self.y is not None:
            explanation['true_label'] = int(self.y[position])
            explanation['correct'] = bool(self.predictions[position] == self.y[position])
        
        # Anomalous features (95th percentile)
        threshold = np.percentile(abs_shap, 95)
        anomalous_idx = np.where(abs_shap > threshold)[0]
        explanation['anomalous_features'] = [
            {
                'name': self.feature_names[idx],
                'value': float(self.X[position, idx]),
                'shap_value': float(shap_values[idx])
            }
            for idx in anomalous_idx
        ]
        
        return explanation
    
    def explain_events(
        self,
        event_ids: List,
        events_are_labels: bool = True,
        output_dir: str = "./xgb_explanations",
        create_reports: bool = True
    ) -> dict:
        """Explain multiple events efficiently."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "=" * 80)
        print(f"EXPLAINING {len(event_ids)} EVENTS")
        print("=" * 80)
        
        # Convert labels
        if events_are_labels:
            if not self.id_to_position:
                raise ValueError("event_ids required when events_are_labels=True")
            positions = self.convert_labels_to_positions(event_ids)
            print(f"✓ Converted {len(positions)}/{len(event_ids)} events")
        else:
            positions = event_ids
        
        # Global stats
        global_stats = self.get_global_stats()
        
        # Explain each event
        print(f"\nExplaining {len(positions)} events...")
        explanations = []
        import time
        start = time.time()
        
        for i, pos in enumerate(positions):
            if i > 0 and i % 100 == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                remaining = (len(positions) - i) / rate if rate > 0 else 0
                print(f"  Progress: {i}/{len(positions)} ({rate:.1f}/sec, ~{remaining/60:.1f} min left)")
            
            if pos < len(self.X):
                explanations.append(self.explain_event(pos))
        
        elapsed = time.time() - start
        print(f"\n✓ Explained {len(explanations)} in {elapsed:.1f}s ({len(explanations)/elapsed:.1f}/sec)")
        
        results = {
            'global_stats': global_stats,
            'explanations': explanations,
            'metadata': {'n_events': len(explanations), 'timestamp': datetime.now().isoformat()}
        }
        
        # Save JSON
        with open(output_dir / "explanations.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Reports
        if create_reports:
            print("\nCreating reports...")
            self._create_reports(results, output_dir)
        
        # CSV
        self._create_csv(results, output_dir)
        
        print("\n" + "=" * 80)
        print("✅ COMPLETE!")
        print("=" * 80)
        print(f"\nOutput: {output_dir}")
        print(f"  • global_stats.html")
        print(f"  • explanations/index.html (master)")
        print(f"  • explanations/*.html ({len(explanations)} reports)")
        print(f"  • *.csv (exports)")
        
        return results
    
    def _create_reports(self, results, output_dir):
        """Create HTML reports."""
        reports_dir = output_dir / "explanations"
        reports_dir.mkdir(exist_ok=True)
        
        # Global report
        self._create_global_html(results['global_stats'], output_dir)
        
        # Master summary
        self._create_summary_html(results['explanations'], reports_dir)
        
        # Individual reports
        for exp in results['explanations']:
            self._create_event_html(exp, reports_dir)
    
    def _create_global_html(self, stats, output_dir):
        """Global statistics HTML."""
        html = self._html_header("Model Statistics")
        html.append('<h1>XGBoost Model Statistics</h1>')
        
        # Stats cards
        html.append('<div class="stats">')
        html.append(f'<div class="stat-card"><div class="stat-value">{stats["n_samples"]:,}</div><div class="stat-label">Samples</div></div>')
        html.append(f'<div class="stat-card"><div class="stat-value">{stats["n_features"]:,}</div><div class="stat-label">Features</div></div>')
        if 'performance' in stats:
            html.append(f'<div class="stat-card"><div class="stat-value">{stats["performance"]["accuracy"]:.3f}</div><div class="stat-label">Accuracy</div></div>')
        html.append('</div>')
        
        # Top features
        html.append('<h2>Top 20 Features (Importance)</h2><table>')
        html.append('<tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>')
        for i, f in enumerate(stats['top_20_features'], 1):
            html.append(f'<tr><td>{i}</td><td>{f["name"]}</td><td>{f["importance"]:.6f}</td></tr>')
        html.append('</table>')
        
        html.append('<h2>Top 20 Features (SHAP)</h2><table>')
        html.append('<tr><th>Rank</th><th>Feature</th><th>Mean |SHAP|</th></tr>')
        for i, f in enumerate(stats['top_20_shap_features'], 1):
            html.append(f'<tr><td>{i}</td><td>{f["name"]}</td><td>{f["mean_abs_shap"]:.6f}</td></tr>')
        html.append('</table>')
        
        html.append('</div></body></html>')
        with open(output_dir / "global_stats.html", 'w') as f:
            f.write('\n'.join(html))
    
    def _create_summary_html(self, explanations, reports_dir):
        """Master summary HTML."""
        html = self._html_header(f"Explanations ({len(explanations)} events)")
        html.append(f'<h1>Event Explanations ({len(explanations)})</h1>')
        
        html.append('<table>')
        html.append('<tr><th>Event</th><th>Pred</th><th>Prob</th><th>Top Feature</th><th>SHAP</th><th>Anomalous</th><th>Link</th></tr>')
        
        for exp in explanations:
            top = exp['top_20_features'][0]
            prob = exp.get('pred_proba', '-')
            prob_str = f'{prob:.3f}' if isinstance(prob, float) else '-'
            
            html.append(f'<tr>')
            html.append(f'<td><strong>{exp["event_id"]}</strong></td>')
            html.append(f'<td>{exp["prediction"]:.3f}</td>')
            html.append(f'<td>{prob_str}</td>')
            html.append(f'<td>{top["name"][:40]}</td>')
            html.append(f'<td>{top["shap_value"]:+.4f}</td>')
            html.append(f'<td>{len(exp["anomalous_features"])}</td>')
            html.append(f'<td><a href="event_{exp["event_id"]}.html">View</a></td>')
            html.append('</tr>')
        
        html.append('</table></div></body></html>')
        with open(reports_dir / "index.html", 'w') as f:
            f.write('\n'.join(html))
    
    def _create_event_html(self, exp, reports_dir):
        """Individual event HTML."""
        html = self._html_header(f"Event {exp['event_id']}")
        html.append(f'<a href="index.html">← Back</a>')
        html.append(f'<h1>Event {exp["event_id"]}</h1>')
        
        # Prediction
        html.append(f'<div class="metric">Prediction: {exp["prediction"]:.4f}')
        if 'pred_proba' in exp:
            html.append(f' | Prob: {exp["pred_proba"]:.4f}')
        if 'correct' in exp:
            html.append(f' | {"✓" if exp["correct"] else "✗"}')
        html.append('</div>')
        
        # Anomalous alert
        if exp['anomalous_features']:
            html.append(f'<div class="alert">⚠️ {len(exp["anomalous_features"])} anomalous features</div>')
        
        # Top 20
        html.append('<h2>Top 20 Features</h2><table>')
        html.append('<tr><th>Rank</th><th>Feature</th><th>Value</th><th>SHAP</th></tr>')
        for i, f in enumerate(exp['top_20_features'], 1):
            cls = 'shap-pos' if f['shap_value'] > 0 else 'shap-neg'
            html.append(f'<tr><td>{i}</td><td>{f["name"]}</td><td>{f["value"]:.6f}</td>')
            html.append(f'<td class="{cls}">{f["shap_value"]:+.6f}</td></tr>')
        html.append('</table>')
        
        html.append('</div></body></html>')
        filename = f"event_{exp['event_id']}.html".replace('/', '_')
        with open(reports_dir / filename, 'w') as f:
            f.write('\n'.join(html))
    
    def _html_header(self, title):
        """Common HTML header."""
        return [
            '<!DOCTYPE html><html><head><meta charset="UTF-8">',
            f'<title>{title}</title>',
            '<style>',
            'body{font-family:Arial;margin:20px;background:#f5f5f5}',
            '.container,.stat-card{background:#fff}',
            'h1{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}',
            'h2{color:#34495e;margin-top:30px;background:#ecf0f1;padding:10px}',
            'table{border-collapse:collapse;width:100%;margin:20px 0}',
            'th{background:#34495e;color:#fff;padding:12px;text-align:left}',
            'td{padding:10px;border-bottom:1px solid #ddd}',
            'tr:hover{background:#f8f9fa}',
            '.shap-pos{color:#27ae60;font-weight:bold}',
            '.shap-neg{color:#e74c3c;font-weight:bold}',
            '.metric{background:#e8f4f8;padding:15px;margin:10px 0;border-radius:5px}',
            '.alert{background:#fff3cd;border:1px solid #ffc107;padding:15px;margin:20px 0}',
            '.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:20px 0}',
            '.stat-card{padding:20px;text-align:center;border-radius:8px}',
            '.stat-value{font-size:32px;font-weight:bold;color:#2c3e50}',
            '.stat-label{color:#7f8c8d;margin-top:5px}',
            'a{color:#3498db;text-decoration:none}',
            'a:hover{text-decoration:underline}',
            '</style></head><body><div class="container">'
        ]
    
    def _create_csv(self, results, output_dir):
        """Export to CSV."""
        import csv
        
        with open(output_dir / "explanations_summary.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Event_ID', 'Prediction', 'Prob', 'Top_Feature', 'Top_SHAP', 'Anomalous'])
            
            for exp in results['explanations']:
                top = exp['top_20_features'][0]
                writer.writerow([
                    exp['event_id'], exp['prediction'], exp.get('pred_proba', ''),
                    top['name'], top['shap_value'], len(exp['anomalous_features'])
                ])
        
        with open(output_dir / "explanations_detailed.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Event_ID', 'Rank', 'Feature', 'Value', 'SHAP'])
            
            for exp in results['explanations']:
                for i, f in enumerate(exp['top_20_features'], 1):
                    writer.writerow([exp['event_id'], i, f['name'], f['value'], f['shap_value']])
        
        print("✓ CSV: explanations_summary.csv, explanations_detailed.csv")


def explain_xgb_events(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: List[str],
    event_ids: List,
    events_are_labels: bool = True,
    event_index: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    output_dir: str = "./xgb_explanations"
) -> dict:
    """
    Quick interface for XGBoost SHAP explanations.
    
    Example:
        results = explain_xgb_events(
            model=xgb_model,
            X=X_data,
            feature_names=feature_names,
            event_ids=[3947355, ...],  # 3000 events
            event_index=df.index.values,
            y=y_true,
            output_dir="./explanations"
        )
    """
    explainer = XGBExplainer(model, X, feature_names, event_index, y)
    return explainer.explain_events(event_ids, events_are_labels, output_dir)


if __name__ == "__main__":
    print("XGBoost + SHAP Explainer")
    print("Efficient for 5M rows × 1000 features")
    print("\nSee module docstring for usage")
