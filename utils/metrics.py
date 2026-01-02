"""
Metrics and evaluation utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from typing import Dict, List


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def compute_per_class_metrics(
    y_true: List[int], 
    y_pred: List[int],
    class_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary mapping class names to metrics
    """
    # Compute per-class precision, recall, f1
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Compute support (number of samples per class)
    unique, counts = np.unique(y_true, return_counts=True)
    support_dict = dict(zip(unique, counts))
    
    # Organize by class
    num_classes = len(precision)
    per_class = {}
    
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class_{i}"
        per_class[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support_dict.get(i, 0))
        }
    
    return per_class


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def print_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = None
):
    """
    Print detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)


class MetricTracker:
    """
    Utility class for tracking metrics during training.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metric_name: str, value: float):
        """Add a value to a metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name: str) -> float:
        """Get average of a metric."""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 0.0
        return np.mean(self.metrics[metric_name])
    
    def get_all(self, metric_name: str) -> List[float]:
        """Get all values of a metric."""
        return self.metrics.get(metric_name, [])
    
    def reset(self, metric_name: str = None):
        """Reset metrics."""
        if metric_name:
            self.metrics[metric_name] = []
        else:
            self.metrics = {}
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get full history of all metrics."""
        return self.metrics.copy()


# ============================================================================
# Ablation Comparison Utilities
# ============================================================================

def create_ablation_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Create ablation comparison table for paper.
    
    Args:
        results: Dictionary mapping experiment names to metrics
            Example: {
                'Single Frame (Baseline)': {'accuracy': 0.869, 'f1_macro': 0.860},
                '8-Frame Temporal': {'accuracy': 0.885, 'f1_macro': 0.880},
                ...
            }
    
    Returns:
        Formatted table string (LaTeX format)
    """
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\caption{Ablation Study: Temporal Visual Modeling}\n"
    table += "\\begin{tabular}{l|cc}\n"
    table += "\\hline\n"
    table += "Configuration & Accuracy (\\%) & F1-Score (Macro) \\\\\n"
    table += "\\hline\n"
    
    for exp_name, metrics in results.items():
        acc = metrics.get('accuracy', 0) * 100
        f1 = metrics.get('f1_macro', 0)
        table += f"{exp_name} & {acc:.1f} & {f1:.3f} \\\\\n"
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}"
    
    return table


def compare_models(
    baseline_results: Dict[str, float],
    temporal_results: Dict[str, float]
) -> Dict[str, float]:
    """
    Compare baseline and temporal model performance.
    
    Args:
        baseline_results: Baseline model metrics
        temporal_results: Temporal model metrics
    
    Returns:
        Dictionary of improvements
    """
    improvements = {}
    
    for metric_name in baseline_results.keys():
        if metric_name in temporal_results:
            baseline_val = baseline_results[metric_name]
            temporal_val = temporal_results[metric_name]
            
            # Absolute improvement
            abs_improvement = temporal_val - baseline_val
            
            # Relative improvement (%)
            rel_improvement = (abs_improvement / baseline_val) * 100 if baseline_val != 0 else 0
            
            improvements[metric_name] = {
                'baseline': baseline_val,
                'temporal': temporal_val,
                'absolute_improvement': abs_improvement,
                'relative_improvement': rel_improvement
            }
    
    return improvements


if __name__ == "__main__":
    print("Metrics module loaded successfully!")
    
    # Example usage
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1, 0, 2, 2]
    
    print("\n=== Example Metrics ===")
    metrics = compute_metrics(y_true, y_pred)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    print("\n=== Per-Class Metrics ===")
    per_class = compute_per_class_metrics(y_true, y_pred, class_names=['A', 'B', 'C'])
    for class_name, class_metrics in per_class.items():
        print(f"{class_name}: {class_metrics}")
