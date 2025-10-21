#!/usr/bin/env python3
"""
Visualize VAE model parameters (weights and biases).

This script:
1. Loads a trained VAE checkpoint
2. Extracts weights and biases from key layers
3. Creates visualizations:
   - Weight distribution histograms
   - Weight magnitude heatmaps
   - Bias distributions
   - Layer-wise parameter statistics
   - Gradient flow visualization (if available)
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Add models to path
sys.path.insert(0, 'models')
from model import HierCVAE
from training import load_checkpoint


def get_parameter_stats(model):
    """
    Extract parameter statistics from the model.
    
    Returns:
        Dictionary with layer names and their parameter statistics
    """
    stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_np = param.detach().cpu().numpy()
            stats[name] = {
                'shape': param.shape,
                'mean': float(param_np.mean()),
                'std': float(param_np.std()),
                'min': float(param_np.min()),
                'max': float(param_np.max()),
                'num_params': param.numel(),
                'data': param_np
            }
    
    return stats


def plot_parameter_distributions(stats, output_dir, max_plots=20):
    """Plot histograms of parameter distributions."""
    print(f"\n[1/6] Plotting parameter distributions...")
    
    # Filter to most important layers (weights only, not biases)
    weight_stats = {k: v for k, v in stats.items() if 'weight' in k.lower()}
    
    # Sort by number of parameters
    sorted_layers = sorted(weight_stats.items(), key=lambda x: x[1]['num_params'], reverse=True)
    
    # Plot top layers
    num_plots = min(max_plots, len(sorted_layers))
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (name, stat) in enumerate(sorted_layers[:num_plots]):
        ax = axes[idx]
        data = stat['data'].flatten()
        
        # Plot histogram
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f"{name}\n({stat['num_params']:,} params)", fontsize=8)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Count', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"μ={stat['mean']:.3f}\nσ={stat['std']:.3f}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=6, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Weight Distributions (Top 20 Layers by Parameter Count)', fontsize=14, y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'weight_distributions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def plot_bias_distributions(stats, output_dir):
    """Plot bias distributions."""
    print(f"[2/6] Plotting bias distributions...")
    
    # Filter biases
    bias_stats = {k: v for k, v in stats.items() if 'bias' in k.lower()}
    
    if not bias_stats:
        print("  → No biases found in model")
        return
    
    # Sort by size
    sorted_biases = sorted(bias_stats.items(), key=lambda x: x[1]['num_params'], reverse=True)
    
    # Create figure
    num_plots = min(20, len(sorted_biases))
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (name, stat) in enumerate(sorted_biases[:num_plots]):
        ax = axes[idx]
        data = stat['data'].flatten()
        
        # Plot histogram
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f"{name}\n({stat['num_params']} params)", fontsize=8)
        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Count', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"μ={stat['mean']:.3f}\nσ={stat['std']:.3f}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=6, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Bias Distributions (Top 20 Layers)', fontsize=14, y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'bias_distributions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def plot_weight_heatmaps(stats, output_dir, max_plots=6):
    """Plot heatmaps of 2D weight matrices."""
    print(f"[3/6] Plotting weight heatmaps...")
    
    # Find 2D weight matrices
    weight_2d = {}
    for name, stat in stats.items():
        if 'weight' in name.lower() and len(stat['shape']) == 2:
            weight_2d[name] = stat
    
    if not weight_2d:
        print("  → No 2D weight matrices found")
        return
    
    # Sort by size and take largest
    sorted_weights = sorted(weight_2d.items(), key=lambda x: x[1]['num_params'], reverse=True)
    
    num_plots = min(max_plots, len(sorted_weights))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, stat) in enumerate(sorted_weights[:num_plots]):
        ax = axes[idx]
        data = stat['data']
        
        # Limit size for visualization
        if data.shape[0] > 500 or data.shape[1] > 500:
            # Sample the matrix
            row_step = max(1, data.shape[0] // 500)
            col_step = max(1, data.shape[1] // 500)
            data = data[::row_step, ::col_step]
        
        # Plot heatmap
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', 
                       vmin=-max(abs(data.min()), abs(data.max())),
                       vmax=max(abs(data.min()), abs(data.max())))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_title(f"{name}\nShape: {stat['shape']}", fontsize=9)
        ax.set_xlabel('Input dim', fontsize=8)
        ax.set_ylabel('Output dim', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Weight Matrix Heatmaps (Top 6 Largest)', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'weight_heatmaps.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def plot_layer_statistics(stats, output_dir):
    """Plot layer-wise parameter statistics."""
    print(f"[4/6] Plotting layer-wise statistics...")
    
    # Group by module
    modules = defaultdict(lambda: {'params': 0, 'mean_abs': [], 'std': []})
    
    for name, stat in stats.items():
        # Extract module name (e.g., 'encoder', 'decoder')
        module = name.split('.')[0] if '.' in name else name
        modules[module]['params'] += stat['num_params']
        modules[module]['mean_abs'].append(abs(stat['mean']))
        modules[module]['std'].append(stat['std'])
    
    # Convert to arrays
    module_names = list(modules.keys())
    param_counts = [modules[m]['params'] for m in module_names]
    avg_mean_abs = [np.mean(modules[m]['mean_abs']) for m in module_names]
    avg_std = [np.mean(modules[m]['std']) for m in module_names]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Parameter counts
    ax = axes[0]
    bars = ax.bar(range(len(module_names)), param_counts, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(module_names)))
    ax.set_xticklabels(module_names, rotation=45, ha='right')
    ax.set_ylabel('Number of Parameters', fontsize=10)
    ax.set_title('Parameters per Module', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, param_counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:,}', ha='center', va='bottom', fontsize=8)
    
    # 2. Average absolute mean
    ax = axes[1]
    ax.bar(range(len(module_names)), avg_mean_abs, color='coral', edgecolor='black')
    ax.set_xticks(range(len(module_names)))
    ax.set_xticklabels(module_names, rotation=45, ha='right')
    ax.set_ylabel('Average |Mean|', fontsize=10)
    ax.set_title('Average Absolute Mean per Module', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Average std
    ax = axes[2]
    ax.bar(range(len(module_names)), avg_std, color='mediumseagreen', edgecolor='black')
    ax.set_xticks(range(len(module_names)))
    ax.set_xticklabels(module_names, rotation=45, ha='right')
    ax.set_ylabel('Average Std Dev', fontsize=10)
    ax.set_title('Average Standard Deviation per Module', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'layer_statistics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def plot_parameter_magnitude_comparison(stats, output_dir):
    """Compare parameter magnitudes across all layers."""
    print(f"[5/6] Plotting parameter magnitude comparison...")
    
    # Collect data
    names = []
    means = []
    stds = []
    ranges = []
    param_counts = []
    
    for name, stat in sorted(stats.items()):
        names.append(name)
        means.append(abs(stat['mean']))
        stds.append(stat['std'])
        ranges.append(stat['max'] - stat['min'])
        param_counts.append(stat['num_params'])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mean absolute values
    ax = axes[0, 0]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.split('.')[-1] for n in names], fontsize=6)
    ax.set_xlabel('|Mean|', fontsize=10)
    ax.set_title('Absolute Mean Values', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Standard deviations
    ax = axes[0, 1]
    ax.barh(y_pos, stds, color='coral', edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.split('.')[-1] for n in names], fontsize=6)
    ax.set_xlabel('Std Dev', fontsize=10)
    ax.set_title('Standard Deviations', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Value ranges
    ax = axes[1, 0]
    ax.barh(y_pos, ranges, color='mediumseagreen', edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.split('.')[-1] for n in names], fontsize=6)
    ax.set_xlabel('Range (Max - Min)', fontsize=10)
    ax.set_title('Value Ranges', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Parameter counts (log scale)
    ax = axes[1, 1]
    ax.barh(y_pos, param_counts, color='mediumpurple', edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.split('.')[-1] for n in names], fontsize=6)
    ax.set_xlabel('Number of Parameters', fontsize=10)
    ax.set_title('Parameter Counts (log scale)', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Parameter Statistics Across All Layers', fontsize=14, y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'parameter_magnitudes.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {output_path}")


def save_parameter_summary(stats, output_dir, epoch):
    """Save detailed parameter summary to text file."""
    print(f"[6/6] Saving parameter summary...")
    
    output_path = os.path.join(output_dir, 'parameter_summary.txt')
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL PARAMETER SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint epoch: {epoch}\n\n")
        
        # Overall statistics
        total_params = sum(stat['num_params'] for stat in stats.values())
        trainable_params = sum(stat['num_params'] for stat in stats.values())
        
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n\n")
        
        # Group by module
        modules = defaultdict(lambda: {'params': 0, 'layers': 0})
        for name, stat in stats.items():
            module = name.split('.')[0] if '.' in name else name
            modules[module]['params'] += stat['num_params']
            modules[module]['layers'] += 1
        
        f.write("Parameters by module:\n")
        f.write("-" * 80 + "\n")
        for module, info in sorted(modules.items()):
            f.write(f"  {module:20s}: {info['params']:12,} params ({info['layers']:3d} layers)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED LAYER STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by parameter count
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['num_params'], reverse=True)
        
        for name, stat in sorted_stats:
            f.write(f"{name}\n")
            f.write(f"  Shape:      {stat['shape']}\n")
            f.write(f"  Parameters: {stat['num_params']:,}\n")
            f.write(f"  Mean:       {stat['mean']:+.6f}\n")
            f.write(f"  Std:        {stat['std']:.6f}\n")
            f.write(f"  Min:        {stat['min']:+.6f}\n")
            f.write(f"  Max:        {stat['max']:+.6f}\n")
            f.write(f"  Range:      {stat['max'] - stat['min']:.6f}\n")
            f.write("\n")
    
    print(f"  → Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize VAE model parameters")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="parameter_viz", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    # Model architecture (must match training)
    parser.add_argument("--use_seqemb", action="store_true", help="Use sequence embeddings")
    parser.add_argument("--seqemb_dim", type=int, default=None, help="Sequence embedding dimension")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ff", type=int, default=1024)
    parser.add_argument("--nlayers", type=int, default=6)
    parser.add_argument("--z_global", type=int, default=512)
    parser.add_argument("--z_local", type=int, default=256)
    parser.add_argument("--decoder_hidden", type=int, default=512)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("VAE PARAMETER VISUALIZATION")
    print("=" * 80)
    print()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    
    # Determine sequence embedding dimension from checkpoint if needed
    seqemb_dim = args.seqemb_dim
    
    # Always try to infer from checkpoint first
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Check if checkpoint has sequence projection layer
        if 'encoder.enc.seq_proj.weight' in state_dict:
            # Extract input dimension from seq_proj: [out_features, in_features]
            detected_seqemb_dim = state_dict['encoder.enc.seq_proj.weight'].shape[1]
            print(f"Detected sequence embedding dim from checkpoint: {detected_seqemb_dim}")
            
            if args.use_seqemb:
                if seqemb_dim is not None and seqemb_dim != detected_seqemb_dim:
                    print(f"Warning: Provided --seqemb_dim {seqemb_dim} != detected {detected_seqemb_dim}")
                    print(f"Using detected value: {detected_seqemb_dim}")
                seqemb_dim = detected_seqemb_dim
            else:
                print("Warning: Checkpoint was trained WITH sequence embeddings, but --use_seqemb not set")
                print("Enabling sequence embeddings automatically...")
                args.use_seqemb = True
                seqemb_dim = detected_seqemb_dim
        else:
            # Checkpoint has no sequence embeddings
            if args.use_seqemb:
                print("Warning: --use_seqemb set but checkpoint has no sequence embeddings")
                print("Disabling sequence embeddings...")
                args.use_seqemb = False
                seqemb_dim = None
    
    # Load checkpoint to get saved hyperparameters
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu')
    saved_hyperparams = checkpoint_data.get('hyperparameters', None)
    
    # Use hyperparameters from checkpoint if available
    if saved_hyperparams:
        print("✅ Using hyperparameters from checkpoint")
        d_model = saved_hyperparams.get('d_model', args.d_model)
        nhead = saved_hyperparams.get('nhead', args.nhead)
        ff = saved_hyperparams.get('ff', args.ff)
        nlayers = saved_hyperparams.get('nlayers', args.nlayers)
        z_global = saved_hyperparams.get('z_global', args.z_global)
        z_local = saved_hyperparams.get('z_local', args.z_local)
        decoder_hidden = saved_hyperparams.get('decoder_hidden', args.decoder_hidden)
        saved_seqemb_dim = saved_hyperparams.get('seqemb_dim', seqemb_dim)
        
        if saved_seqemb_dim is not None:
            seqemb_dim = saved_seqemb_dim
            
        print(f"  d_model={d_model}, nhead={nhead}, ff={ff}, nlayers={nlayers}")
        print(f"  z_global={z_global}, z_local={z_local}, decoder_hidden={decoder_hidden}")
        print(f"  seqemb_dim={seqemb_dim}")
    else:
        print("⚠️  Using command-line hyperparameters (old checkpoint format)")
        d_model = args.d_model
        nhead = args.nhead
        ff = args.ff
        nlayers = args.nlayers
        z_global = args.z_global
        z_local = args.z_local
        decoder_hidden = args.decoder_hidden
    print()
    
    model = HierCVAE(
        seqemb_dim=seqemb_dim,
        d_model=d_model,
        nhead=nhead,
        ff=ff,
        nlayers=nlayers,
        z_g=z_global,
        z_l=z_local,
        equivariant=True,
        decoder_hidden=decoder_hidden,
        use_dihedrals=True
    ).to(args.device)
    
    model, epoch, loss_history, _ = load_checkpoint(model, args.checkpoint, device=args.device)
    print(f"Loaded checkpoint from epoch: {epoch}")
    print()
    
    # Extract parameter statistics
    print("Extracting parameter statistics...")
    stats = get_parameter_stats(model)
    print(f"Found {len(stats)} parameter tensors")
    print(f"Total parameters: {sum(s['num_params'] for s in stats.values()):,}")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    print("=" * 80)
    
    plot_parameter_distributions(stats, args.output_dir)
    plot_bias_distributions(stats, args.output_dir)
    plot_weight_heatmaps(stats, args.output_dir)
    plot_layer_statistics(stats, args.output_dir)
    plot_parameter_magnitude_comparison(stats, args.output_dir)
    save_parameter_summary(stats, args.output_dir, epoch)
    
    print()
    print("=" * 80)
    print("✅ Parameter visualization complete!")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

