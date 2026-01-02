"""
Visualization utilities for results and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str = None,
    normalize: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize by row (true labels)
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Configure axes
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    metric_tracker,
    save_path: str = None
):
    """
    Plot training and validation curves.
    
    Args:
        metric_tracker: MetricTracker object with training history
        save_path: Path to save figure
    """
    history = metric_tracker.get_history()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
    if 'val_loss' in history:
        # Val loss is per-epoch, need to align with train loss
        val_loss = history['val_loss']
        val_epochs = np.linspace(0, len(history.get('train_loss', val_loss)), len(val_loss))
        axes[0].plot(val_epochs, val_loss, label='Val Loss', linewidth=2)
    
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy', alpha=0.7)
    if 'val_acc' in history:
        val_acc = history['val_acc']
        val_epochs = np.linspace(0, len(history.get('train_acc', val_acc)), len(val_acc))
        axes[1].plot(val_epochs, val_acc, label='Val Accuracy', linewidth=2)
    
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_performance(
    per_class_metrics: Dict[str, Dict[str, float]],
    save_path: str = None
):
    """
    Plot per-class performance metrics.
    
    Args:
        per_class_metrics: Dictionary of per-class metrics
        save_path: Path to save figure
    """
    classes = list(per_class_metrics.keys())
    precision = [per_class_metrics[c]['precision'] for c in classes]
    recall = [per_class_metrics[c]['recall'] for c in classes]
    f1 = [per_class_metrics[c]['f1'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Emotion Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class performance plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ablation_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    save_path: str = None
):
    """
    Plot ablation study comparison.
    
    Args:
        results: Dictionary mapping experiment names to metrics
        metric: Metric to plot ('accuracy' or 'f1_macro')
        save_path: Path to save figure
    """
    experiments = list(results.keys())
    values = [results[exp].get(metric, 0) for exp in experiments]
    
    # Convert to percentage if accuracy
    if metric == 'accuracy':
        values = [v * 100 for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(experiments, values, alpha=0.8, color='steelblue')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{value:.2f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)' if metric == 'accuracy' else 'F1-Score (Macro)')
    ax.set_title(f'Ablation Study: {metric.replace("_", " ").title()}')
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits
    if metric == 'accuracy':
        ax.set_ylim([min(values) - 5, 100])
    else:
        ax.set_ylim([min(values) - 0.05, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ablation comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_dataset_comparison(
    dataset_results: Dict[str, Dict[str, float]],
    save_path: str = None
):
    """
    Plot comparison across datasets.
    
    Args:
        dataset_results: Dictionary mapping dataset names to metrics
        save_path: Path to save figure
    """
    datasets = list(dataset_results.keys())
    accuracy = [dataset_results[d]['accuracy'] * 100 for d in datasets]
    f1_macro = [dataset_results[d]['f1_macro'] for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', alpha=0.8)
    
    # Create second y-axis for F1
    ax2 = ax.twinx()
    ax2.bar(x + width/2, f1_macro, width, label='F1-Score', alpha=0.8, color='orange')
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy (%)', color='steelblue')
    ax2.set_ylabel('F1-Score (Macro)', color='orange')
    ax.set_title('Model Performance Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    
    # Example usage
    print("\n=== Example Plots ===")
    
    # Example confusion matrix
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2] * 10
    y_pred = [0, 1, 2, 0, 1, 1, 0, 2, 2] * 10
    plot_confusion_matrix(y_true, y_pred, class_names=['Happy', 'Sad', 'Angry'])
    
    print("Example plots generated!")
