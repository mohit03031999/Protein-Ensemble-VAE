#!/usr/bin/env python3
"""
Training Module
Contains training utilities and functions for the protein VAE model.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import wandb
import math

from losses import compute_total_loss
from kl_schedulers import (
    create_kl_scheduler, 
    FreeBitsKLLoss,
    CyclicalKLScheduler,
    MonotonicKLScheduler
)


def run_epoch(model, loader, opt, device, klw_g, klw_l, w_pair, pair_stride,
              train, w_dihedral, w_rama, w_bond, w_angle, w_rec, w_seq, w_clash, epoch):
    """
    Run one training or validation epoch.
    
    Args:
        model: HierCVAE model
        loader: DataLoader
        opt: Optimizer (only used if train=True)
        device: torch device
        klw_g, klw_l: KL loss weights
        w_pair: pair distance loss weight
        pair_stride: stride for pair distance loss
        train: whether to train or validate
        w_dihedral: dihedral loss weight
        w_rama: ramachandran loss weight
        w_bond: bond length constraint weight
        w_angle: bond angle constraint weight
        w_rec: reconstruction loss weight
        epoch: current epoch number
        
    Returns:
        dict with loss statistics
    """
    if train: 
        model.train()
    else:     
        model.eval()
    
    tot = tot_rec = tot_pair = tot_kg = tot_kl = tot_dih = tot_rama = tot_bond = tot_angle = tot_seq = tot_clash = 0.0
    tot_seq_acc = 0.0  # NEW: track sequence accuracy
    n = 0
    batch_idx = 0
    
    for batch_data in loader:
        # Pair-wise training: batch_data is (input_data, target_data)
        input_data, target_data = batch_data
        
        # Unpack input conformer (for encoding)
        n_in, ca_in, c_in, mask_in, seqemb_in, dih_in, seq_lbl_in = input_data
        # Unpack target conformer (for reconstruction)
        n_tgt, ca_tgt, c_tgt, mask_tgt, seqemb_tgt, dih_tgt, seq_lbl_tgt = target_data
        
        # Move input data to device
        n_in = n_in.to(device)
        ca_in = ca_in.to(device)
        c_in = c_in.to(device)
        mask_in = mask_in.to(device)
        dih_in = dih_in.to(device)
        if seqemb_in is not None:
            seqemb_in = seqemb_in.to(device)
        
        # Move target data to device
        n_tgt = n_tgt.to(device)
        ca_tgt = ca_tgt.to(device)
        c_tgt = c_tgt.to(device)
        mask_tgt = mask_tgt.to(device)
        dih_tgt = dih_tgt.to(device)
        seq_lbl_tgt = seq_lbl_tgt.to(device)
        if seqemb_tgt is not None:
            seqemb_tgt = seqemb_tgt.to(device)
        
        # Use mask from target for reconstruction (both should be same anyway)
        mask = mask_tgt

        with torch.set_grad_enabled(train):
            # PAIR-WISE TRAINING: Encode input conformer, decode to reconstruct target
            pred_N, pred_CA, pred_C, pred_seq, mu_g, lv_g, mu_l, lv_l = model(
                seqemb_in, n_in, ca_in, c_in, dih_in, mask
            )

            # Compute loss against TARGET conformer (not input!)
            loss_dict = compute_total_loss(
                pred_N=pred_N, pred_CA=pred_CA, pred_C=pred_C, pred_seq=pred_seq,
                target_N=n_tgt, target_CA=ca_tgt, target_C=c_tgt, target_seq_labels=seq_lbl_tgt,
                mask=mask, mu_g=mu_g, lv_g=lv_g, mu_l=mu_l, lv_l=lv_l,
                target_dihedrals=dih_tgt,
                klw_g=klw_g, klw_l=klw_l, w_pair=w_pair, pair_stride=pair_stride,
                w_dihedral=w_dihedral, w_rama=w_rama, w_bond=w_bond, w_angle=w_angle,
                w_rec=w_rec, w_seq=w_seq, w_clash=w_clash,
            )
            
            loss = loss_dict['total']
            
            # NEW: Compute sequence accuracy
            with torch.no_grad():
                pred_seq_labels = torch.argmax(pred_seq, dim=-1)  # [B, L]
                correct = (pred_seq_labels == seq_lbl_tgt) & mask.bool()
                seq_accuracy = correct.sum().float() / mask.sum().float()

            # Print detailed loss info
            if train and batch_idx == 0:  # First batch of epoch
                with torch.no_grad():
                    valid_coords = pred_CA[mask.bool()]
                    if len(valid_coords) > 0:
                        coord_std = valid_coords.std(dim=0).mean().item()
                        coord_range = (valid_coords.max() - valid_coords.min()).item()
                        # Convert MSE to RMSD for readability
                        rmsd_ca = torch.sqrt(torch.tensor(loss_dict['reconstruction_ca'])).item()
                        rmsd_n = torch.sqrt(torch.tensor(loss_dict['reconstruction_n'])).item()
                        rmsd_c = torch.sqrt(torch.tensor(loss_dict['reconstruction_c'])).item()
                        print(f"  [Batch 1] RMSD_CA: {rmsd_ca:.2f}Å | "
                              f"RMSD_N: {rmsd_n:.2f}Å | "
                              f"RMSD_C: {rmsd_c:.2f}Å | "
                              f"Seq_Acc: {seq_accuracy:.3f} | "
                              f"Seq_Loss: {loss_dict['sequence']:.3f} | "
                              f"PairDist: {loss_dict['pair_distance']:.3f} | "
                              f"Rama: {loss_dict['ramachandran']:.3f}")

            if train:
                opt.zero_grad()
                loss.backward()
                
                # In training.py, after line 114 (loss.backward())
                if not torch.isfinite(loss):
                    print(f"❌ NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}")
                    print(f"   Loss components: {loss_dict}")
                    print(f"   Coordinates - CA mean: {pred_CA.mean():.2f}, std: {pred_CA.std():.2f}")
                    print(f"   Latents - mu_g: {mu_g.mean():.2f}, lv_g: {lv_g.mean():.2f}")
                    raise ValueError("Training collapsed - NaN detected")

                # Check individual loss components
                for key, val in loss_dict.items():
                    if not torch.isfinite(val):
                        print(f"❌ NaN in loss component: {key} = {val}")
                
                # Gradient clipping to prevent explosion (more aggressive)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                opt.step()
                
                # Log gradient norm to W&B (per batch)
                if wandb.run is not None:
                    wandb.log({
                        'train/batch_grad_norm': grad_norm.item(),
                        'train/batch_loss': loss.item(),
                    })

        # Accumulate statistics
        bs = ca_tgt.size(0)
        tot      += loss.item() * bs
        tot_rec  += loss_dict['reconstruction'].item() * bs
        tot_pair += loss_dict['pair_distance'].item() * bs
        tot_kg   += loss_dict['kl_global'].item() * bs
        tot_kl   += loss_dict['kl_local'].item() * bs
        tot_dih  += loss_dict['dihedral_total'].item() * bs
        tot_rama += loss_dict['ramachandran'].item() * bs
        tot_bond += loss_dict['bond_length'].item() * bs
        tot_angle += loss_dict['bond_angle'].item() * bs
        tot_seq  += loss_dict['sequence'].item() * bs  # NEW
        tot_seq_acc += seq_accuracy.item() * bs  # NEW
        tot_clash += loss_dict['clash'].item() * bs  # NEW
        n += bs
        batch_idx += 1
    
    return dict(
        loss=tot/n, 
        rec=tot_rec/n, 
        pair=tot_pair/n, 
        klg=tot_kg/n, 
        kll=tot_kl/n,
        dihedral=tot_dih/n, 
        rama=tot_rama/n,
        bond=tot_bond/n,
        angle=tot_angle/n,
        seq=tot_seq/n,  # NEW
        seq_acc=tot_seq_acc/n,  # NEW
        clash=tot_clash/n  # NEW
    )


def train_model(model, train_loader, val_loader, args):
    """
    Train the model for multiple epochs with early stopping and cyclical KL annealing.
    
    Args:
        model: HierCVAE model
        train_loader: training DataLoader
        val_loader: validation DataLoader
        args: training arguments
        
    Returns:
        tuple: (model, loss_history)
    """
    device = args.device
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Watch model with W&B to log gradients and parameters
    if wandb.run is not None:
        wandb.watch(model, log='all', log_freq=100)  # Log gradients and parameters every 100 batches
    
    # Learning rate scheduler - reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
    )
    
    # ========== KL ANNEALING SCHEDULERS ==========
    # Create KL schedulers for global and local latents
    kl_schedule_type = getattr(args, 'kl_schedule', 'cyclical')  # cyclical, monotonic, adaptive, exponential
    
    print(f"\n{'='*80}")
    print(f"KL ANNEALING CONFIGURATION:")
    print(f"  Schedule type: {kl_schedule_type}")
    print(f"  Global max weight: {args.klw_global}")
    print(f"  Local max weight: {args.klw_local}")
    
    n_cycles = getattr(args, 'kl_cycles', 4)
    ratio = getattr(args, 'kl_ratio', 0.5)
    print(f"  Cycles: {n_cycles}")
    print(f"  Ratio: {ratio}")
    kl_scheduler_global = CyclicalKLScheduler(
            n_cycles=n_cycles, ratio=ratio, max_weight=args.klw_global
    )
    kl_scheduler_local = CyclicalKLScheduler(
            n_cycles=n_cycles, ratio=ratio, max_weight=args.klw_local
    )
    
    # Early stopping configuration
    early_stopping_patience = getattr(args, 'early_stopping_patience', 20)
    early_stopping_metric = getattr(args, 'early_stopping_metric', 'rec')  # 'rec', 'loss', or 'rmsd'
    early_stopping_delta = getattr(args, 'early_stopping_delta', 1e-4)
    
    best_val_metric = float('inf')
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    
    print(f"\n{'='*80}")
    print(f"EARLY STOPPING CONFIGURATION:")
    print(f"  Metric: {early_stopping_metric}")
    print(f"  Patience: {early_stopping_patience} epochs")
    print(f"  Min improvement delta: {early_stopping_delta}")
    print(f"{'='*80}\n")
    
    loss_history = {
        'train': {'loss': [], 'rec': [], 'pair': [], 'klg': [], 'kll': [], 'dihedral': [], 'rama': [], 'bond': [], 'angle': [], 'seq': [], 'seq_acc': []},
        'val': {'loss': [], 'rec': [], 'pair': [], 'klg': [], 'kll': [], 'dihedral': [], 'rama': [], 'bond': [], 'angle': [], 'seq': [], 'seq_acc': []},
        'early_stopping': {
            'best_epoch': 0,
            'best_val_metric': float('inf'),
            'metric_name': early_stopping_metric
        }
    }
    
    for epoch in range(1, args.epochs + 1):
        # ========== DYNAMIC KL WEIGHT SCHEDULING ==========
        # Get KL weights from schedulers (this is the key improvement!)
        klw_g = kl_scheduler_global.step(epoch, args.epochs)
        klw_l = kl_scheduler_local.step(epoch, args.epochs)
        
        # Training epoch
        tr = run_epoch(
            model, train_loader, opt, device, klw_g, klw_l, 
            args.w_pair, args.pair_stride, train=True, 
            w_dihedral=args.w_dihedral, w_rama=args.w_rama,
            w_bond=args.w_bond, w_angle=args.w_angle, w_rec=args.w_rec, w_seq=args.w_seq, w_clash=args.w_clash, epoch=epoch
        )
        
        # Validation epoch (use same KL weights as training)
        va = run_epoch(
            model, val_loader, opt, device, 
            klw_g=klw_g,  # Use same scheduled weights
            klw_l=klw_l,
            w_pair=args.w_pair, pair_stride=args.pair_stride, train=False,
            w_dihedral=args.w_dihedral, w_rama=args.w_rama,
            w_bond=args.w_bond, w_angle=args.w_angle, w_rec=args.w_rec, w_seq=args.w_seq, w_clash=args.w_clash, epoch=epoch
        )

        # Store loss history
        for key in loss_history['train']:
            loss_history['train'][key].append(tr[key])
            loss_history['val'][key].append(va[key])

        # Update learning rate based on validation RMSD
        scheduler.step(va['rec'])
        
        # Get current learning rate
        current_lr = opt.param_groups[0]['lr']

        # Log to W&B
        if wandb.run is not None:
            # Convert reconstruction MSE to RMSD for better interpretability
            train_rmsd = math.sqrt(tr['rec']) if tr['rec'] >= 0 else 0
            val_rmsd = math.sqrt(va['rec']) if va['rec'] >= 0 else 0
            
            wandb.log({
                'epoch': epoch,
                'learning_rate': current_lr,
                'kl_weight_global': klw_g,
                'kl_weight_local': klw_l,
                'kl_schedule_info/cycle_position': (epoch % (args.epochs / getattr(args, 'kl_cycles', 4))) if kl_schedule_type == 'cyclical' else 0,
                # Training metrics
                'train/loss': tr['loss'],
                'train/reconstruction': tr['rec'],
                'train/rmsd': train_rmsd,
                'train/pair_distance': tr['pair'],
                'train/kl_global': tr['klg'],
                'train/kl_local': tr['kll'],
                'train/dihedral': tr.get('dihedral', 0),
                'train/ramachandran': tr.get('rama', 0),
                'train/bond_length': tr.get('bond', 0),
                'train/bond_angle': tr.get('angle', 0),
                'train/sequence_loss': tr.get('seq', 0),  # NEW
                'train/sequence_accuracy': tr.get('seq_acc', 0),  # NEW
                # Validation metrics
                'val/loss': va['loss'],
                'val/reconstruction': va['rec'],
                'val/rmsd': val_rmsd,
                'val/pair_distance': va['pair'],
                'val/kl_global': va['klg'],
                'val/kl_local': va['kll'],
                'val/dihedral': va.get('dihedral', 0),
                'val/ramachandran': va.get('rama', 0),
                'val/bond_length': va.get('bond', 0),
                'val/bond_angle': va.get('angle', 0),
                'val/sequence_loss': va.get('seq', 0),  # NEW
                'val/sequence_accuracy': va.get('seq_acc', 0),  # NEW
            })

        # Print epoch summary
        print(f"Epoch {epoch:03d} | "
              f"train: loss {tr['loss']:.4f} rec {tr['rec']:.4f} seq_acc {tr.get('seq_acc', 0):.3f} | "
              f"val:   loss {va['loss']:.4f} rec {va['rec']:.4f} seq_acc {va.get('seq_acc', 0):.3f}")

        # ========== EARLY STOPPING LOGIC ==========
        # Get the metric to monitor
        if early_stopping_metric == 'rmsd':
            current_val_metric = math.sqrt(va['rec']) if va['rec'] >= 0 else 0
        else:
            current_val_metric = va.get(early_stopping_metric, va['loss'])
        
        # Check for improvement
        improvement = best_val_metric - current_val_metric
        
        if improvement > early_stopping_delta:
            # Significant improvement found
            best_val_metric = current_val_metric
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            
            # Save best model checkpoint
            import os
            checkpoint_dir = os.path.dirname(args.save)
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            
            hyperparameters = {
                'seqemb_dim': getattr(args, 'seqemb_dim', None),
                'd_model': args.d_model,
                'nhead': args.nhead,
                'ff': args.ff,
                'nlayers': args.nlayers,
                'z_global': args.z_global,
                'z_local': args.z_local,
                'decoder_hidden': args.decoder_hidden,
                'use_seqemb': args.use_seqemb,
            }
            
            save_checkpoint(
                model, best_checkpoint_path, 
                epoch=epoch, 
                loss_history=loss_history,
                hyperparameters=hyperparameters
            )
            
            print(f"  ✅ New best model! {early_stopping_metric}: {current_val_metric:.6f} (improved by {improvement:.6f})")
            
            # Log to W&B
            if wandb.run is not None:
                wandb.log({
                    'best_epoch': best_epoch,
                    f'best_val_{early_stopping_metric}': best_val_metric,
                })
        else:
            epochs_without_improvement += 1
            print(f"  ⏳ No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs "
                  f"(best {early_stopping_metric}: {best_val_metric:.6f} at epoch {best_epoch})")
        
        # Check if we should stop
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n{'='*80}")
            print(f"🛑 EARLY STOPPING TRIGGERED")
            print(f"  No improvement for {early_stopping_patience} consecutive epochs")
            print(f"  Best epoch: {best_epoch}")
            print(f"  Best validation {early_stopping_metric}: {best_val_metric:.6f}")
            print(f"{'='*80}\n")
            
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"✅ Restored model weights from epoch {best_epoch}")
            
            # Update loss history with early stopping info
            loss_history['early_stopping']['best_epoch'] = best_epoch
            loss_history['early_stopping']['best_val_metric'] = best_val_metric
            loss_history['early_stopping']['stopped_at_epoch'] = epoch
            
            break
    
    # If training completed without early stopping
    if epochs_without_improvement < early_stopping_patience:
        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETED - All {args.epochs} epochs finished")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Best validation {early_stopping_metric}: {best_val_metric:.6f}")
        print(f"{'='*80}\n")
        
        # Restore best model if it exists
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"✅ Restored best model weights from epoch {best_epoch}")
    
    # Update final loss history
    loss_history['early_stopping']['best_epoch'] = best_epoch
    loss_history['early_stopping']['best_val_metric'] = best_val_metric

    return model, loss_history


def save_checkpoint(model, path, epoch=None, loss_history=None, hyperparameters=None):
    """
    Save model checkpoint with hyperparameters and log to W&B.
    
    Args:
        model: The model to save
        path: Path to save checkpoint
        epoch: Current epoch number
        loss_history: Training loss history
        hyperparameters: Dict of model hyperparameters (d_model, z_g, z_l, etc.)
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss_history': loss_history,
        'hyperparameters': hyperparameters
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")
    
    # Log checkpoint as W&B artifact
    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f'model-checkpoint',
            type='model',
            description='Protein VAE model checkpoint',
            metadata={
                'epoch': epoch,
                'hyperparameters': hyperparameters
            }
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        print(f"Logged checkpoint to W&B")


def load_checkpoint(model, path, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: Model instance to load weights into
        path: Path to checkpoint
        device: Device to load model on
        
    Returns:
        tuple: (model, epoch, loss_history, hyperparameters)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', None)
    loss_history = checkpoint.get('loss_history', None)
    hyperparameters = checkpoint.get('hyperparameters', None)
    
    if hyperparameters:
        print(f"  Loaded hyperparameters: {hyperparameters}")
    else:
        print("  Warning: No hyperparameters found in checkpoint (old format)")
    
    return model, epoch, loss_history, hyperparameters
