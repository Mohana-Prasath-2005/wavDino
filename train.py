"""
Training script for Temporal wavDINO-Emotion.

Supports:
- Baseline (single frame) and temporal (multi-frame) models
- Multiple datasets (CREMA-D, RAVDESS, AFEW)
- Ablation experiments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os

from models import create_model
from data import create_dataset, collate_fn
from utils.metrics import compute_metrics, MetricTracker
from utils.visualization import plot_confusion_matrix, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description='Train Temporal wavDINO-Emotion')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--dataset', type=str, required=True, choices=['CREMA-D', 'RAVDESS', 'AFEW'])
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--afew_percentage', type=int, default=10, help='Percentage of AFEW to use (default: 10)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config, args):
    """Create train, val, and test dataloaders."""
    # Determine AFEW percentage
    afew_percentage = args.afew_percentage if args.dataset == 'AFEW' else 100
    
    # Training data (with augmentation)
    train_dataset = create_dataset(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split='train',
        num_frames=config['model']['num_frames'],
        augment=True,
        use_single_frame=config['model']['model_type'] == 'baseline',
        afew_percentage=afew_percentage
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Validation data (no augmentation)
    val_dataset = create_dataset(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split='val',
        num_frames=config['model']['num_frames'],
        augment=False,
        use_single_frame=config['model']['model_type'] == 'baseline',
        afew_percentage=afew_percentage
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    if args.dataset == 'AFEW':
        print(f"Using {afew_percentage}% of AFEW dataset")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, metric_tracker):
    """Train for one epoch."""
    model.train()
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(audio, video)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Update metrics
        metric_tracker.update('train_loss', loss.item())
        metric_tracker.update('train_acc', acc.item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc.item():.4f}'
        })
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        writer.add_scalar('Train/Accuracy', acc.item(), global_step)
    
    # Get epoch averages
    avg_loss = metric_tracker.get_average('train_loss')
    avg_acc = metric_tracker.get_average('train_acc')
    
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device, epoch, writer, metric_tracker):
    """Validate the model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move to device
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(audio, video)
            loss = criterion(logits, labels)
            
            # Predictions
            preds = torch.argmax(logits, dim=1)
            
            # Store for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update metrics
            acc = (preds == labels).float().mean()
            metric_tracker.update('val_loss', loss.item())
            metric_tracker.update('val_acc', acc.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.4f}'
            })
    
    # Compute detailed metrics
    metrics = compute_metrics(all_labels, all_preds)
    
    # Get averages
    avg_loss = metric_tracker.get_average('val_loss')
    avg_acc = metric_tracker.get_average('val_acc')
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Accuracy', avg_acc, epoch)
    writer.add_scalar('Val/F1', metrics['f1_macro'], epoch)
    
    return avg_loss, avg_acc, metrics, all_preds, all_labels


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{config['experiment_name']}_{args.dataset}_{timestamp}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"Experiment: {exp_name}")
    print(f"Output directory: {output_dir}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, args)
    
    # Create model
    model_config = config['model'].copy()
    model_config['num_classes'] = train_loader.dataset.label_to_idx.__len__()
    model = create_model(model_config)
    model = model.to(device)
    
    print("\nModel Architecture:")
    print(f"Parameters: {model.get_num_params()}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Tensorboard
    writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
    
    # Metric tracker
    metric_tracker = MetricTracker()
    
    # Training loop
    best_val_acc = 0.0
    best_val_f1 = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    num_epochs = config['training']['num_epochs']
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, metric_tracker
        )
        
        # Validate
        val_loss, val_acc, val_metrics, val_preds, val_labels = validate(
            model, val_loader, criterion, device,
            epoch, writer, metric_tracker
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        print(f"  Val F1 (weighted): {val_metrics['f1_weighted']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_metrics['f1_macro'],
            'best_val_acc': best_val_acc,
            'config': config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pth')
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_metrics['f1_macro']
            torch.save(checkpoint, output_dir / 'checkpoint_best.pth')
            print(f"  ✓ New best model saved! (Acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f})")
        
        # Plot confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0:
            cm_path = output_dir / f'confusion_matrix_epoch_{epoch+1}.png'
            plot_confusion_matrix(
                val_labels, val_preds,
                class_names=train_loader.dataset.emotion_labels,
                save_path=str(cm_path)
            )
    
    # Training complete
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Plot training curves
    plot_training_curves(
        metric_tracker,
        save_path=str(output_dir / 'training_curves.png')
    )
    
    writer.close()


if __name__ == '__main__':
    main()
