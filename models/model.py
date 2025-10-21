#!/usr/bin/env python3
"""
Protein VAE Model Module
Main model class that combines encoder and decoder components.
"""

import torch
import torch.nn as nn
from typing import Optional

from encoder import ProteinEncoder
from en_gnn_decoder import ResidueDecoder


class HierCVAE(nn.Module):
    """
    Hierarchical Conditional VAE for protein structure generation.
    Combines encoder and decoder components.
    """
    def __init__(self, seqemb_dim: Optional[int],
                 d_model: int = 512, nhead: int = 8, ff: int = 1024, nlayers: int = 6,
                 z_g: int = 512, z_l: int = 256, dropout: float = 0.1,
                 equivariant: bool = True, decoder_hidden: int = 256, use_dihedrals: bool = True):
        super().__init__()
        self.seqemb_dim = seqemb_dim
        self.equivariant = equivariant
        self.use_dihedrals = use_dihedrals
        
        # Initialize simplified encoder
        self.encoder = ProteinEncoder(
            seqemb_dim=seqemb_dim,
            d_model=d_model, nhead=nhead, ff=ff, nlayers=nlayers,
            z_g=z_g, z_l=z_l, dropout=dropout,
            use_dihedrals=use_dihedrals
        )
        
        self.decoder = ResidueDecoder(
            z_g=z_g, z_l=z_l,
            hidden=decoder_hidden, dropout=dropout, equivariant=equivariant
        )

    def forward(self, seqemb_or_none, n_coords, ca_coords, c_coords, dihedrals, mask):
        """
        Forward pass with dihedral-aware encoding using full backbone.
        
        Args:
            seqemb_or_none: [B,L,seq_dim] or None - sequence embeddings
            n_coords: [B,L,3] - N atom coordinates
            ca_coords: [B,L,3] - Cα coordinates
            c_coords: [B,L,3] - C atom coordinates
            dihedrals: [B,L,6] - backbone dihedral sin/cos values
            mask: [B,L] - valid residue mask
            
        Returns:
            pred_N, pred_CA, pred_C: [B,L,3] predicted N, CA, C coordinates
            pred_seq: [B,L,20] predicted amino acid logits
            mu_g, lv_g, mu_l, lv_l: latent parameters for KL loss
        """
        # Encode to latents using full backbone
        z_g, z_l, mu_g, lv_g, mu_l, lv_l = self.encoder(
            seqemb_or_none, n_coords, ca_coords, c_coords, dihedrals, mask
        )

        # Decode to N, CA, C coordinates and sequence logits
        pred_N, pred_CA, pred_C, pred_seq = self.decoder(z_g, z_l, mask=mask)

        return pred_N, pred_CA, pred_C, pred_seq, mu_g, lv_g, mu_l, lv_l

    def encode(self, seqemb_or_none, n_coords, ca_coords, c_coords, dihedrals, mask):
        """Encode inputs to latent space using full backbone."""
        return self.encoder(seqemb_or_none, n_coords, ca_coords, c_coords, dihedrals, mask)
    
    def decode(self, z_g, z_l, mask=None):
        """Decode latents to N, CA, C coordinates and sequence logits."""
        return self.decoder(z_g, z_l, mask=mask)  # Returns (N, CA, C, seq_logits)
    
    def sample(self, mask, seqemb_or_none=None, num_samples=1):
        """
        Sample new conformations from the model.
        
        Args:
            mask: [B,L] valid residue mask
            seqemb_or_none: [B,L,seq_dim] or None - sequence embeddings
            num_samples: number of samples to generate
            
        Returns:
            (N_coords, CA_coords, C_coords, seq_logits): tuple of [B*num_samples,L,3] sampled coordinates
                                                          and [B*num_samples,L,20] sequence logits
        """
        B = mask.size(0)
        device = mask.device
        
        # Sample from prior
        z_g = torch.randn(B * num_samples, self.encoder.latent.global_head[-1].out_features // 2, device=device)
        z_l = torch.randn(B * num_samples, mask.size(1), self.encoder.latent.local_head[-1].out_features // 2, device=device)
        
        # Expand mask
        mask_expanded = mask.repeat_interleave(num_samples, dim=0)
        
        # Decode - returns (N, CA, C, seq_logits)
        N_coords, CA_coords, C_coords, seq_logits = self.decode(z_g, z_l, mask=mask_expanded)
        
        return N_coords, CA_coords, C_coords, seq_logits
