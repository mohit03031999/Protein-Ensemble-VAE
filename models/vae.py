#!/usr/bin/env python3
# train_hier_cvae.py - Updated to use modular components
import os, argparse

import torch
import torch.nn as nn
import wandb

# Import modular components
from model import HierCVAE
from data import create_data_loaders
from training import train_model, save_checkpoint

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Train a hierarchical, state-conditioned CVAE on NMR + cross-PDB ensembles.")
    ap.add_argument("--manifest_train", required=True)
    ap.add_argument("--manifest_val",   required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=13)

    # Sampling & inputs
    ap.add_argument("--use_seqemb", action="store_true", help="Use ESM per-residue embeddings from H5 if present")

    # Model sizes
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--nlayers", type=int, default=6)
    ap.add_argument("--z_global", type=int, default=512)
    ap.add_argument("--z_local", type=int, default=256)
    ap.add_argument("--decoder_hidden", type=int, default=512, help="Hidden dimension for decoder")

    # Loss weights
    ap.add_argument("--pair_stride", type=int, default=8)
    ap.add_argument("--w_rec", type=float, default=10.0, help="Weight for reconstruction - MAXIMUM priority for sub-angstrom")
    ap.add_argument("--w_pair", type=float, default=10.0, help="Weight for pair distance - increased for fine structure")
    ap.add_argument("--kl_warmup_epochs", type=int, default=20)
    ap.add_argument("--klw_global", type=float, default=1.0, help="KL weight for global - reduced for better diversity")
    ap.add_argument("--klw_local", type=float, default=0.5, help="KL weight for local - reduced for better diversity")
    ap.add_argument("--w_dihedral", type=float, default=20.0, help="Weight for dihedral - minimal for sub-angstrom")
    ap.add_argument("--w_rama", type=float, default=400, help="Weight for Ramachandran - minimal for sub-angstrom")
    ap.add_argument("--w_bond", type=float, default=500.0, help="Weight for bond length - CRITICAL for C-N peptide bonds")
    ap.add_argument("--w_angle", type=float, default=500.0, help="Weight for bond angle - important for backbone geometry")
    ap.add_argument("--w_seq", type=float, default=50.0, help="Weight for sequence prediction loss")
    ap.add_argument("--w_clash", type=float, default=300.0, help="Weight for clash loss")

    # KL Annealing (Expert-level posterior collapse prevention)
    ap.add_argument("--kl_schedule", type=str, default="cyclical", 
                    choices=["cyclical", "monotonic", "adaptive", "exponential"],
                    help="KL annealing schedule type (cyclical recommended for ensembles)")
    ap.add_argument("--kl_cycles", type=int, default=4, 
                    help="Number of cycles for cyclical annealing")
    ap.add_argument("--kl_ratio", type=float, default=0.4,
                    help="Ratio of cycle spent increasing (cyclical only, 0.5=linear sawtooth)")

    # Runtime
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="checkpoints/hier_cvae.pt")
    
    # Early Stopping
    ap.add_argument("--early_stopping_patience", type=int, default=20, 
                    help="Number of epochs without improvement before stopping")
    ap.add_argument("--early_stopping_metric", type=str, default="rec", 
                    choices=["rec", "loss", "rmsd"], 
                    help="Metric to monitor for early stopping (rec=reconstruction MSE, rmsd=RMSD, loss=total loss)")
    ap.add_argument("--early_stopping_delta", type=float, default=1e-4, 
                    help="Minimum change in monitored metric to qualify as improvement")
    
    # Weights & Biases logging
    ap.add_argument("--wandb_project", type=str, default="Protein-VAE", help="W&B project name")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: auto-generated)")
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], 
                    help="W&B mode: online (sync to cloud), offline (save locally), disabled (no logging)")

    args = ap.parse_args()

    # Create data loaders
    tr_dl, va_dl, seqemb_dim = create_data_loaders(
        args.manifest_train, args.manifest_val, 
        batch_size=args.batch_size, use_seqemb=args.use_seqemb, seed=args.seed
    )
    
    # Store seqemb_dim in args for later use
    args.seqemb_dim = seqemb_dim
    
    # Initialize Weights & Biases
    if args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                # Data config
                'manifest_train': args.manifest_train,
                'manifest_val': args.manifest_val,
                'batch_size': args.batch_size,
                'use_seqemb': args.use_seqemb,
                'seqemb_dim': seqemb_dim,
                'seed': args.seed,
                # Model architecture
                'd_model': args.d_model,
                'nhead': args.nhead,
                'ff': args.ff,
                'nlayers': args.nlayers,
                'z_global': args.z_global,
                'z_local': args.z_local,
                'decoder_hidden': args.decoder_hidden,
                # Training configL
                'epochs': args.epochs,
                'lr': args.lr,
                'kl_warmup_epochs': args.kl_warmup_epochs,
                # Loss weights
                'w_rec': args.w_rec,
                'w_pair': args.w_pair,
                'pair_stride': args.pair_stride,
                'klw_global': args.klw_global,
                'klw_local': args.klw_local,
                'w_dihedral': args.w_dihedral,
                'w_rama': args.w_rama,
                'w_bond': args.w_bond,
                'w_angle': args.w_angle,
                'w_seq': args.w_seq,
                'w_clash': args.w_clash,
                # Runtime
                'device': args.device,
            },
            tags=['protein-vae', 'hierarchical-cvae', 'structure-generation']
        )
        print(f"🔗 W&B run initialized: {wandb.run.name}")
        print(f"📊 View at: {wandb.run.get_url()}")

    # Model & optimizer
    model = HierCVAE(seqemb_dim,
                     d_model=args.d_model, nhead=args.nhead, ff=args.ff, nlayers=args.nlayers,
                     z_g=args.z_global, z_l=args.z_local, equivariant=True,
                     decoder_hidden=args.decoder_hidden, use_dihedrals=True).to(args.device)
    
    # Train the model
    model, loss_history = train_model(model, tr_dl, va_dl, args)

    # Save checkpoint with hyperparameters
    hyperparameters = {
        'seqemb_dim': seqemb_dim,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'ff': args.ff,
        'nlayers': args.nlayers,
        'z_global': args.z_global,
        'z_local': args.z_local,
        'decoder_hidden': args.decoder_hidden,
        'use_seqemb': args.use_seqemb,
    }
    print (hyperparameters)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    save_checkpoint(model, args.save, loss_history=loss_history, hyperparameters=hyperparameters)
    
    # Finish W&B run
    if args.wandb_mode != "disabled":
        wandb.finish()
        print(f"✅ W&B run finished and synced")
    
    # Final checkpoint is automatically saved with timestamp
    print(f"Training complete! Check checkpoints/ directory for saved models.")

if __name__ == "__main__":
    main()