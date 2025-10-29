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
    Pair-wise ensemble dataset for learning conformational distributions.
    
    Each item = pair of conformers from the same protein:
      (conf1, conf2) where model encodes conf1 and decodes to reconstruct conf2
    
    This forces the VAE to learn ensemble distributions rather than just copying.
    """
    def __init__(self, manifest_csv: str, use_seqemb: bool = True):
        self.use_seqemb = use_seqemb
        self.conformers = []  # List of all conformers
        self.proteins = {}  # {protein_id: [conf_idx1, conf_idx2, ...]}
        
        print(f"🔄 Loading conformers from {manifest_csv}")
        print(f"   Mode: PAIR-WISE ensemble training")
        
        # Load all conformers from manifest
        with open(manifest_csv, "r") as f:
            for row in csv.DictReader(f):
                h5_path = row["h5_path"].strip()
                if not os.path.exists(h5_path):
                    print(f"⚠️  H5 not found: {h5_path}")
                    continue
                
                # Extract protein ID from filename (e.g., "1ubq_nmr.h5" -> "1ubq")
                protein_id = os.path.basename(h5_path).replace('.h5', '').split('_')[0]
                
                if protein_id not in self.proteins:
                    self.proteins[protein_id] = []
                
                print(f"📂 Loading: {h5_path} (protein: {protein_id})")
                
                # Load conformers and track their indices
                start_idx = len(self.conformers)
                new_conformers = self._load_h5(h5_path, protein_id)
                self.conformers.extend(new_conformers)
                end_idx = len(self.conformers)
                
                # Store indices of conformers for this protein
                self.proteins[protein_id].extend(range(start_idx, end_idx))
        
        if len(self.conformers) == 0:
            raise RuntimeError(f"No data loaded from {manifest_csv}")
        
        # Create pairs of conformers from same protein
        self.pairs = []
        for protein_id, conf_indices in self.proteins.items():
            if len(conf_indices) >= 2:
                # Create all unique pairs for this protein
                for i in range(len(conf_indices)):
                    for j in range(i+1, len(conf_indices)):
                        self.pairs.append((conf_indices[i], conf_indices[j]))
        
        if len(self.pairs) == 0:
            raise RuntimeError(f"No pairs could be created! Each protein needs at least 2 conformers.")
        
        print(f"✅ Created {len(self.pairs)} conformer pairs from {len(self.proteins)} proteins")
        print(f"   Total conformers: {len(self.conformers)}")
        print(f"   Average conformers per protein: {len(self.conformers) / len(self.proteins):.1f}")
        print(f"   Training samples (pairs): {len(self.pairs)}")
    
    def _load_h5(self, h5_path: str, protein_id: str):
        """Load all conformers from one H5 file and return as list."""
        conformers_list = []
        
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
                    conformers_list.append({
                        'n': n_coords[k],      # [L, 3]
                        'ca': ca_coords[k],    # [L, 3]
                        'c': c_coords[k],      # [L, 3]
                        'mask': mask[k],       # [L]
                        'seq_emb': seq_emb,    # [L, D] or None
                        'dihedrals': dih_k,    # [L, 6]
                        'sequence': sequence,  # string or None
                        'protein_id': protein_id,  # NEW: track protein ID
                        'h5': h5_path
                    })
        
        return conformers_list
    
    def __len__(self):
        """Return number of training samples (conformer pairs)."""
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Return pair of conformers for pair-wise ensemble training.
        
        Returns:
            tuple of two 7-tuples: (input_data, target_data)
            Each is: (n, ca, c, mask, seq_emb, dih, seq_labels)
        """
        # Pair-wise training: encode conf1, reconstruct conf2
        conf1_idx, conf2_idx = self.pairs[idx]
        conf1 = self.conformers[conf1_idx]
        conf2 = self.conformers[conf2_idx]
        
        # Process both conformers
        data1 = self._process_conformer(conf1)
        data2 = self._process_conformer(conf2)
        
        return data1, data2
    
    def _process_conformer(self, conf):
        """Convert conformer dict to tensors with centering and preprocessing."""
        # Convert to tensors
        n = torch.from_numpy(conf['n']).float()          # [L, 3]
        ca = torch.from_numpy(conf['ca']).float()        # [L, 3]
        c = torch.from_numpy(conf['c']).float()          # [L, 3]
        mask = torch.from_numpy(conf['mask']).float()    # [L]
        dih = torch.from_numpy(conf['dihedrals']).float()  # [L, 6]
        
        # Center on CA centroid (CRITICAL for EGNN - preserves bond lengths)
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
        
        # Convert sequence string to integer labels for classification
        aa_to_idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        
        sequence = conf.get('sequence', None)
        if sequence:
            L = len(mask)
            seq_labels = torch.zeros(L, dtype=torch.long)
            for i, aa in enumerate(sequence[:L]):
                seq_labels[i] = aa_to_idx.get(aa, 0)
        else:
            seq_labels = torch.zeros(len(mask), dtype=torch.long)
        
        return n, ca, c, mask, seq_emb, dih, seq_labels


def collate_pad(batch):
    """
    Collate batch with padding for pair-wise ensemble training.
    
    Args:
        batch: List of pairs from dataset: [(data1_0, data2_0), (data1_1, data2_1), ...]
    
    Returns:
        Two sets of data: (input_batch, target_batch)
        Each is: (n_coords, ca_coords, c_coords, mask, seq_emb, dihedrals, seq_labels)
    """
    # Separate into input and target batches
    input_batch = [item[0] for item in batch]
    target_batch = [item[1] for item in batch]
    
    # Collate both separately
    input_data = _collate_single_batch(input_batch)
    target_data = _collate_single_batch(target_batch)
    
    return input_data, target_data


def _collate_single_batch(batch):
    """
    Helper function to collate a single batch (not pairs).
    
    Returns:
        n_coords: [B, Lmax, 3]
        ca_coords: [B, Lmax, 3]
        c_coords: [B, Lmax, 3]
        mask: [B, Lmax]
        seq_emb: [B, Lmax, D] or None
        dihedrals: [B, Lmax, 6]
        seq_labels: [B, Lmax]
    """
    B = len(batch)
    Lmax = max(b[0].shape[0] for b in batch)
    
    # Allocate tensors
    n_coords = torch.zeros(B, Lmax, 3)
    ca_coords = torch.zeros(B, Lmax, 3)
    c_coords = torch.zeros(B, Lmax, 3)
    mask = torch.zeros(B, Lmax)
    dihedrals = torch.zeros(B, Lmax, 6)
    seq_labels = torch.zeros(B, Lmax, dtype=torch.long)
    
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
    Create data loaders for pair-wise ensemble training.
    
    Args:
        manifest_train: Path to training manifest CSV
        manifest_val: Path to validation manifest CSV
        batch_size: Batch size for training
        use_seqemb: Whether to use sequence embeddings
        seed: Random seed
    
    Returns:
        (train_loader, val_loader, seqemb_dim)
    """
    print("=" * 80)
    print("🔬 PROTEIN ENSEMBLE DATA LOADING - PAIR-WISE MODE")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Sequence embeddings: {'Yes' if use_seqemb else 'No'}")
    print("=" * 80)
    print()
    
    # Create datasets (always pair-wise)
    train_ds = EnsembleDataset(manifest_train, use_seqemb=use_seqemb)
    print()
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
        # Get a sample to determine embedding dimension
        sample = train_ds[0]
        # In pair mode, sample is (data1, data2)
        data = sample[0]
        
        # data is (n, ca, c, mask, seq_emb, dih, seq_labels)
        if data[4] is not None:
            seqemb_dim = data[4].shape[-1]
            print(f"✅ Sequence embedding dimension: {seqemb_dim}")
        else:
            print("⚠️  No sequence embeddings found!")
    
    print()
    print("=" * 80)
    print(f"✅ Train: {len(train_ds)} conformer pairs")
    print(f"✅ Val: {len(val_ds)} conformer pairs")
    print("=" * 80)
    print()
    
    return train_loader, val_loader, seqemb_dim