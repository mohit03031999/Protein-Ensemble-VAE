#!/usr/bin/env python3
"""
Plot training curves from saved checkpoint to visualize early stopping.

Usage:
    python plot_training_curves.py --checkpoint checkpoints/hier_cvae.pt
    python plot_training_curves.py --checkpoint checkpoints/hier_cvae.pt --metric rmsd
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves(checkpoint_path, output_dir='train_viz', metric='rec'):
    """
    Plot training and validation curves from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        output_dir: Directory to save plots
        metric: Which metric to plot ('rec', 'loss', 'rmsd')
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    loss_history = checkpoint.get('loss_history', None)
    
    if loss_history is None:
        print("❌ No loss history found in checkpoint!")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get early stopping info
    early_stop = loss_history.get('early_stopping', {})
    best_epoch = early_stop.get('best_epoch', None)
    stopped_at = early_stop.get('stopped_at_epoch', None)
    best_metric = early_stop.get('best_val_metric', None)
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    if best_epoch:
        print(f"✅ Best epoch: {best_epoch}")
        if best_metric:
            print(f"✅ Best validation metric: {best_metric:.6f}")
        if stopped_at:
            print(f"🛑 Training stopped at epoch: {stopped_at}")
            print(f"⏱️  Epochs saved: {stopped_at - best_epoch}")
    else:
        print("ℹ️  Training completed all epochs (no early stopping)")
    print("="*80 + "\n")
    
    # Extract data
    train_losses = loss_history['train']
    val_losses = loss_history['val']
    epochs = np.arange(1, len(train_losses['loss']) + 1)
    
    # Convert reconstruction MSE to RMSD if requested
    if metric == 'rmsd':
        train_metric = np.sqrt(np.array(train_losses['rec']))
        val_metric = np.sqrt(np.array(val_losses['rec']))
        metric_name = 'RMSD (Å)'
    elif metric == 'rec':
        train_metric = np.array(train_losses['rec'])
        val_metric = np.array(val_losses['rec'])
        metric_name = 'Reconstruction MSE (Ų)'
    elif metric == 'loss':
        train_metric = np.array(train_losses['loss'])
        val_metric = np.array(val_losses['loss'])
        metric_name = 'Total Loss'
    else:
        train_metric = np.array(train_losses.get(metric, train_losses['loss']))
        val_metric = np.array(val_losses.get(metric, val_losses['loss']))
        metric_name = metric.capitalize()
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Progress - {Path(checkpoint_path).name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Main metric (with early stopping marker)
    ax = axes[0, 0]
    ax.plot(epochs, train_metric, 'b-', label='Train', alpha=0.7, linewidth=2)
    ax.plot(epochs, val_metric, 'r-', label='Validation', alpha=0.7, linewidth=2)
    
    # Mark best epoch
    if best_epoch and best_epoch <= len(val_metric):
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, 
                   label=f'Best Epoch ({best_epoch})')
        ax.scatter([best_epoch], [val_metric[best_epoch-1]], 
                  color='green', s=200, marker='*', zorder=5, label='Best Model')
    
    # Mark stopped epoch
    if stopped_at and stopped_at <= len(val_metric):
        ax.axvline(x=stopped_at, color='orange', linestyle=':', linewidth=2, 
                   label=f'Stopped ({stopped_at})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Total Loss
    ax = axes[0, 1]
    ax.plot(epochs, train_losses['loss'], 'b-', label='Train', alpha=0.7)
    ax.plot(epochs, val_losses['loss'], 'r-', label='Validation', alpha=0.7)
    if best_epoch:
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pair Distance Loss
    ax = axes[0, 2]
    ax.plot(epochs, train_losses['pair'], 'b-', label='Train', alpha=0.7)
    ax.plot(epochs, val_losses['pair'], 'r-', label='Validation', alpha=0.7)
    if best_epoch:
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pair Distance Loss', fontsize=12)
    ax.set_title('Pair Distance Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: KL Divergence (Global)
    ax = axes[1, 0]
    ax.plot(epochs, train_losses['klg'], 'b-', label='Train', alpha=0.7)
    ax.plot(epochs, val_losses['klg'], 'r-', label='Validation', alpha=0.7)
    if best_epoch:
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Warning if KL is too low
    avg_val_kl = np.mean(val_losses['klg'][-10:])  # Last 10 epochs
    if avg_val_kl < 0.01:
        ax.text(0.5, 0.95, '⚠️ LOW KL - Possible Posterior Collapse', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('KL Divergence (Global)', fontsize=12)
    ax.set_title('KL Divergence (Global)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: KL Divergence (Local)
    ax = axes[1, 1]
    ax.plot(epochs, train_losses['kll'], 'b-', label='Train', alpha=0.7)
    ax.plot(epochs, val_losses['kll'], 'r-', label='Validation', alpha=0.7)
    if best_epoch:
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Warning if KL is too low
    avg_val_kl_local = np.mean(val_losses['kll'][-10:])
    if avg_val_kl_local < 0.01:
        ax.text(0.5, 0.95, '⚠️ LOW KL - Check Ensemble Diversity', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('KL Divergence (Local)', fontsize=12)
    ax.set_title('KL Divergence (Local)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Physics Losses (Rama + Bond)
    ax = axes[1, 2]
    ax.plot(epochs, val_losses['rama'], 'r-', label='Ramachandran', alpha=0.7, linewidth=2)
    ax.plot(epochs, val_losses['bond'], 'b-', label='Bond Length', alpha=0.7, linewidth=2)
    if best_epoch:
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_title('Physics-Based Losses (Validation)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'training_curves_{metric}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot: {output_path}")
    
    # Create a summary statistics plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Compute improvement over epochs
    improvement = (val_metric[0] - val_metric) / val_metric[0] * 100  # % improvement
    
    ax2.plot(epochs, improvement, 'g-', linewidth=2.5, label='% Improvement from Epoch 1')
    ax2.fill_between(epochs, 0, improvement, alpha=0.3, color='green')
    
    if best_epoch:
        ax2.axvline(x=best_epoch, color='orange', linestyle='--', linewidth=2, 
                   label=f'Best Model (Epoch {best_epoch})')
        best_improvement = improvement[best_epoch-1]
        ax2.scatter([best_epoch], [best_improvement], 
                  color='orange', s=300, marker='*', zorder=5, edgecolor='black', linewidth=2)
        ax2.text(best_epoch, best_improvement, 
                f'\n{best_improvement:.1f}%', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Improvement from Baseline (%)', fontsize=14)
    ax2.set_title(f'Validation {metric_name} - Improvement Over Training', 
                 fontsize=16, fontweight='bold')
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path2 = Path(output_dir) / f'improvement_curve_{metric}.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot: {output_path2}")
    
    # Print statistics
    print("\n" + "="*80)
    print("TRAINING STATISTICS")
    print("="*80)
    print(f"Initial validation {metric_name}: {val_metric[0]:.4f}")
    if best_epoch:
        print(f"Best validation {metric_name}: {val_metric[best_epoch-1]:.4f}")
        print(f"Improvement: {improvement[best_epoch-1]:.2f}%")
    print(f"Final validation {metric_name}: {val_metric[-1]:.4f}")
    print(f"\nAverage validation KL (global): {np.mean(val_losses['klg'][-10:]):.6f}")
    print(f"Average validation KL (local): {np.mean(val_losses['kll'][-10:]):.6f}")
    
    if np.mean(val_losses['klg'][-10:]) < 0.01 or np.mean(val_losses['kll'][-10:]) < 0.01:
        print("\n⚠️  WARNING: KL divergence is very low!")
        print("   This suggests posterior collapse - model may ignore latent space.")
        print("   Recommendation: Increase --klw_global and --klw_local in next training.")
    
    print("="*80 + "\n")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_1.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--output_dir", type=str, default="train_viz",
                       help="Directory to save plots")
    parser.add_argument("--metric", type=str, default="rmsd", 
                       choices=["rmsd", "rec", "loss"],
                       help="Metric to highlight in main plot")
    
    args = parser.parse_args()
    
    print("\n🎨 Plotting training curves...")
    print(f"📂 Checkpoint: {args.checkpoint}")
    print(f"📊 Output directory: {args.output_dir}\n")
    
    plot_training_curves(args.checkpoint, args.output_dir, args.metric)
    
    print("\n✅ Done! Check the plots in the output directory.")
    print(f"   Main plot: {args.output_dir}/training_curves_{args.metric}.png")
    print(f"   Improvement plot: {args.output_dir}/improvement_curve_{args.metric}.png\n")


if __name__ == "__main__":
    main()

