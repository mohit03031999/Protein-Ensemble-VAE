#!/usr/bin/env python3
"""
Data Module
Contains dataset classes and data loading utilities for the protein VAE model.
"""

import os
import csv
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EnsembleDataset(Dataset):
    """
    Each item = one conformer:
      n [L,3], ca [L,3], c [L,3], mask [L], seq_emb [L,D]
    Drawn from union of NMR pool (label 'nmr')
    """
    def __init__(self, manifest_csv: str, use_seqemb: bool = True):
        self.use_seqemb = use_seqemb
        self.conformers = []  # List of (n, ca, c, mask, seq_emb, h5_path)
        
        # Load all conformers from manifest
        with open(manifest_csv, "r") as f:
            for row in csv.DictReader(f):
                h5_path = row["h5_path"].strip()
                if not os.path.exists(h5_path):
                    print(f"⚠️  H5 not found: {h5_path}")
                    continue
                
                print(f"📂 Loading: {h5_path}")
                self._load_h5(h5_path)
        
        if len(self.conformers) == 0:
            raise RuntimeError(f"No data loaded from {manifest_csv}")
        
        print(f"✅ Loaded {len(self.conformers)} conformers")
    
    def _load_h5(self, h5_path: str):
        """Load all conformers from one H5 file."""
        with h5py.File(h5_path, "r") as fh:
            # Load coordinates
            n_coords = fh["coords_N"][:]      # [K, L, 3]
            ca_coords = fh["coords_ca"][:]    # [K, L, 3]
            c_coords = fh["coords_C"][:]      # [K, L, 3]
            mask = fh["mask_ca"][:]           # [K, L]
            
            K, L, _ = ca_coords.shape
            print(f"   → {K} conformers, {L} residues")
            
            # Load sequence embeddings if needed
            seq_emb = None
            if self.use_seqemb:
                if "seq_embed" in fh and "esm2_t33_650M_UR50D" in fh["seq_embed"]:
                    if "layer_33" in fh["seq_embed/esm2_t33_650M_UR50D"]:
                        seq_emb = fh["seq_embed/esm2_t33_650M_UR50D/layer_33"][:]  # [L, D]
                        print(f"   → ESM embeddings: {seq_emb.shape}")
            
            # Load dihedrals (sin/cos of phi, psi, omega)
            dihedrals = None
            if all(k in fh for k in ["torsion_phi_sincos", "torsion_psi_sincos", "torsion_omega_sincos"]):
                phi = fh["torsion_phi_sincos"][:]      # [K, L, 2]
                psi = fh["torsion_psi_sincos"][:]      # [K, L, 2]
                omega = fh["torsion_omega_sincos"][:]  # [K, L, 2]
                dihedrals = np.concatenate([phi, psi, omega], axis=-1)  # [K, L, 6]
                print(f"   → Dihedrals: {dihedrals.shape}")
            
            # Extract sequence for sequence prediction training
            sequence = None
            if 'sequence' in fh:
                seq_raw = fh['sequence'][()]
                sequence = seq_raw.decode('utf-8') if isinstance(seq_raw, (bytes, bytearray)) else str(seq_raw)
                print(f"   → Sequence for training: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
            
            # Store each conformer
            for k in range(K):
                if mask[k].sum() > 0:  # Only keep non-empty
                    dih_k = dihedrals[k] if dihedrals is not None else np.zeros((L, 6), dtype=np.float32)
                    self.conformers.append({
                        'n': n_coords[k],      # [L, 3]
                        'ca': ca_coords[k],    # [L, 3]
                        'c': c_coords[k],      # [L, 3]
                        'mask': mask[k],       # [L]
                        'seq_emb': seq_emb,    # [L, D] or None
                        'dihedrals': dih_k,    # [L, 6]
                        'sequence': sequence,  # [L] string or None
                        'h5': h5_path
                    })
    
    def __len__(self):
        return len(self.conformers)
    
    def __getitem__(self, idx):
        """Return one conformer."""
        conf = self.conformers[idx]
        
        # Convert to tensors
        n = torch.from_numpy(conf['n']).float()          # [L, 3]
        ca = torch.from_numpy(conf['ca']).float()        # [L, 3]
        c = torch.from_numpy(conf['c']).float()          # [L, 3]
        mask = torch.from_numpy(conf['mask']).float()    # [L]
        dih = torch.from_numpy(conf['dihedrals']).float()  # [L, 6]
        
        # Center on CA centroid (CRITICAL for EGNN - keeps centering!)
        # NOTE: Centering preserves all bond lengths - it's just a translation
        valid_ca = ca[mask.bool()]
        if len(valid_ca) > 0:
            centroid = valid_ca.mean(dim=0)
            n = n - centroid
            ca = ca - centroid
            c = c - centroid
        
        # Sequence embeddings
        seq_emb = None
        if conf['seq_emb'] is not None:
            seq_emb = torch.from_numpy(conf['seq_emb']).float()
        
        # NEW: Convert sequence string to integer labels for classification
        # Standard 20 amino acids in alphabetical order
        aa_to_idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        
        sequence = conf.get('sequence', None)
        if sequence:
            # Convert sequence string to integer indices
            L = len(mask)
            seq_labels = torch.zeros(L, dtype=torch.long)  # [L]
            for i, aa in enumerate(sequence[:L]):  # Handle truncation
                seq_labels[i] = aa_to_idx.get(aa, 0)  # Default to 'A' (alanine) for unknown
        else:
            # No sequence available - use dummy labels (will be masked out in loss)
            seq_labels = torch.zeros(len(mask), dtype=torch.long)
        
        return n, ca, c, mask, seq_emb, dih, seq_labels


def collate_pad(batch):
    """
    Collate batch with padding.
    
    Returns:
        n_coords: [B, Lmax, 3]
        ca_coords: [B, Lmax, 3]
        c_coords: [B, Lmax, 3]
        mask: [B, Lmax]
        seq_emb: [B, Lmax, D] or None
        dihedrals: [B, Lmax, 6]
        seq_labels: [B, Lmax] - amino acid class labels (0-19)
    """
    B = len(batch)
    Lmax = max(b[0].shape[0] for b in batch)
    
    # Allocate tensors
    n_coords = torch.zeros(B, Lmax, 3)
    ca_coords = torch.zeros(B, Lmax, 3)
    c_coords = torch.zeros(B, Lmax, 3)
    mask = torch.zeros(B, Lmax)
    dihedrals = torch.zeros(B, Lmax, 6)
    seq_labels = torch.zeros(B, Lmax, dtype=torch.long)  # NEW: sequence labels
    
    # Handle sequence embeddings
    seq_emb = None
    any_seq = any(b[4] is not None for b in batch)  # Index 4 is seq_emb
    if any_seq:
        D = None
        for b in batch:
            if b[4] is not None:
                D = b[4].shape[-1]
                break
        seq_emb = torch.zeros(B, Lmax, D)
    
    # Fill batch
    for i, (n, ca, c, m, emb, dih, seq_lbl) in enumerate(batch):
        L = n.shape[0]
        n_coords[i, :L] = n
        ca_coords[i, :L] = ca
        c_coords[i, :L] = c
        mask[i, :L] = m
        dihedrals[i, :L] = dih
        seq_labels[i, :L] = seq_lbl
        if seq_emb is not None and emb is not None:
            seq_emb[i, :L] = emb
    
    return n_coords, ca_coords, c_coords, mask, seq_emb, dihedrals, seq_labels


def create_data_loaders(manifest_train: str, manifest_val: str, 
                          batch_size: int, use_seqemb: bool, seed: int):
    """
    Create simple data loaders for single protein training.
    
    Returns:
        (train_loader, val_loader, seqemb_dim)
    """
    print("=" * 60)
    print("SIMPLIFIED DATA LOADING")
    print("=" * 60)
    
    # Create datasets
    train_ds = EnsembleDataset(manifest_train, use_seqemb=use_seqemb)
    val_ds = EnsembleDataset(manifest_val, use_seqemb=use_seqemb)
    
    # Create loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_pad,
        num_workers=0  # Single worker for debugging
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_pad,
        num_workers=0
    )
    
    # Get embedding dimension
    seqemb_dim = None
    if use_seqemb:
        for i in range(len(train_ds)):
            _, _, _, _, emb, _, _ = train_ds[i]  # Updated: n, ca, c, mask, seq_emb, dih, seq_labels
            if emb is not None:
                seqemb_dim = emb.shape[-1]
                print(f"✅ Sequence embedding dimension: {seqemb_dim}")
                break
        if seqemb_dim is None:
            print("⚠️  No sequence embeddings found!")
    
    print(f"✅ Train: {len(train_ds)} conformers")
    print(f"✅ Val: {len(val_ds)} conformers")
    print("=" * 60)
    print()
    
    return train_loader, val_loader, seqemb_dim