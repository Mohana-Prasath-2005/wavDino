"""
Evaluation script for Temporal wavDINO-Emotion.

Evaluates trained models and generates:
- Accuracy, F1, Precision, Recall
- Confusion matrices
- Per-class performance
- Comparison tables for paper
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json

from models import create_model
from data import create_dataset, collate_fn
from utils.metrics import compute_metrics, compute_per_class_metrics
from utils.visualization import plot_confusion_matrix, plot_per_class_performance


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Temporal wavDINO-Emotion')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, choices=['CREMA-D', 'RAVDESS', 'AFEW'])
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()


def evaluate(model, dataloader, device):
    """
    Evaluate model on dataset.
    
    Returns:
        all_preds: List of predictions
        all_labels: List of ground truth labels
        all_logits: List of logits (for further analysis)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    progress_bar = tqdm(dataloader, desc='Evaluating')
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move to device
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(audio, video)
            
            # Predictions
            preds = torch.argmax(logits, dim=1)
            
            # Store
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            
            # Update progress bar
            acc = (preds == labels).float().mean()
            progress_bar.set_postfix({'acc': f'{acc.item():.4f}'})
    
    return all_preds, all_labels, all_logits


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluation Results will be saved to: {output_dir}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"\nLoading {args.dataset} dataset ({args.split} split)...")
    dataset = create_dataset(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split=args.split,
        num_frames=config.get('model', {}).get('num_frames', 8),
        augment=False,
        use_single_frame=config.get('model', {}).get('model_type') == 'baseline'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Total samples: {len(dataset)}")
    print(f"Emotion classes: {dataset.emotion_labels}")
    
    # Create model
    model_config = config.get('model', {}).copy()
    model_config['num_classes'] = len(dataset.emotion_labels)
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"\nModel type: {model_config.get('model_type', 'temporal')}")
    print(f"Number of frames: {model_config.get('num_frames', 1)}")
    
    # Evaluate
    print("\nEvaluating...")
    all_preds, all_labels, all_logits = evaluate(model, dataloader, device)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(all_labels, all_preds)
    per_class_metrics = compute_per_class_metrics(
        all_labels, all_preds,
        class_names=dataset.emotion_labels
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"{'='*60}")
    
    print("\nPer-Class Performance:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print(f"{'-'*60}")
    for emotion in dataset.emotion_labels:
        p = per_class_metrics[emotion]['precision']
        r = per_class_metrics[emotion]['recall']
        f1 = per_class_metrics[emotion]['f1']
        s = per_class_metrics[emotion]['support']
        print(f"{emotion:<15} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {s}")
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'split': args.split,
        'model_config': model_config,
        'metrics': metrics,
        'per_class_metrics': per_class_metrics
    }
    
    results_path = output_dir / f'results_{args.dataset}_{args.split}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Plot confusion matrix
    cm_path = output_dir / f'confusion_matrix_{args.dataset}_{args.split}.png'
    plot_confusion_matrix(
        all_labels, all_preds,
        class_names=dataset.emotion_labels,
        save_path=str(cm_path)
    )
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    # Plot per-class performance
    perf_path = output_dir / f'per_class_performance_{args.dataset}_{args.split}.png'
    plot_per_class_performance(
        per_class_metrics,
        save_path=str(perf_path)
    )
    print(f"✓ Per-class performance plot saved to: {perf_path}")
    
    # Save predictions for further analysis
    pred_path = output_dir / f'predictions_{args.dataset}_{args.split}.npz'
    np.savez(
        pred_path,
        predictions=np.array(all_preds),
        labels=np.array(all_labels),
        logits=np.array(all_logits)
    )
    print(f"✓ Predictions saved to: {pred_path}")
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
