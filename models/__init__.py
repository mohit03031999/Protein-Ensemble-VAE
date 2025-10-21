#!/usr/bin/env python3
"""
Protein VAE Models Package
Modular implementation of hierarchical CVAE for protein structure generation.
"""

from .model import HierCVAE
from .encoder import ProteinEncoder, DihedralAwareEncoder, HierLatent
from .en_gnn_decoder import ResidueDecoder
from .losses import (
    compute_total_loss, recon_loss_kabsch, pair_distance_loss,
    kl_global, kl_local, dihedral_consistency_loss, ramachandran_loss,
    omega_trans_loss, compute_dihedrals_from_coords
)
from .training import train_model, run_epoch, save_checkpoint, load_checkpoint
from .data import EnsembleDataset, create_data_loaders, collate_pad, set_seed

__all__ = [
    'HierCVAE',
    'ProteinEncoder', 
    'DihedralAwareEncoder',
    'HierLatent',
    'ResidueDecoder',
    'compute_total_loss',
    'recon_loss_kabsch',
    'pair_distance_loss',
    'kl_global',
    'kl_local', 
    'dihedral_consistency_loss',
    'ramachandran_loss',
    'omega_trans_loss',
    'compute_dihedrals_from_coords',
    'train_model',
    'run_epoch',
    'save_checkpoint',
    'load_checkpoint',
    'EnsembleDataset',
    'create_data_loaders',
    'collate_pad',
    'set_seed'
]
