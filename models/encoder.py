#!/usr/bin/env python3
"""
Protein VAE Encoder Module
Contains the encoder components for the hierarchical CVAE model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding for transformer layers."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]
    
    def forward(self, x):
        # x: [B,L,d_model]
        return x + self.pe[:x.size(1)]


class DihedralAwareEncoder(nn.Module):
    """
    Enhanced encoder that combines sequence embeddings, dihedral angles, and coordinates
    into a unified geometric representation for protein structures.
    """
    def __init__(self, seq_dim: int, dihedral_dim: int, d_model: int, 
                 nhead: int, ff: int, nlayers: int, dropout: float = 0.1,
                 d_pair: int = 128):
        super().__init__()
        self.seq_dim = seq_dim
        self.dihedral_dim = dihedral_dim
        self.d_model = d_model
        
        # Always use sequence embeddings - remove conditional logic
        self.seq_proj = nn.Linear(seq_dim, d_model // 2)
        
        self.dihedral_proj = nn.Linear(dihedral_dim, d_model // 4)
        # full backbone (N, CA, C) = 9D
        self.coord_proj = nn.Linear(9, d_model // 4)
        # Add LayerNorm for stable feature learning instead of manual scaling
        self.coord_norm = nn.LayerNorm(d_model // 4)
        self.dihedral_norm = nn.LayerNorm(d_model // 4)
        
        # Since we always use sequence embeddings, final dimension is fixed
        final_dim = d_model  # seq(d_model/2) + coord(d_model/4) + dihedral(d_model/4)
        
        # Geometric feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(final_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder with positional encoding
        self.pe = SinusoidalPE(d_model)
        self.nhead = nhead
        self.nlayers = nlayers
        
        # Simple transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=ff,
                dropout=dropout, norm_first=True, batch_first=True
            ) for _ in range(nlayers)
        ])
        self.ln = nn.LayerNorm(d_model)
        
        # Geometric attention for local interactions
        self.geom_res_scale = nn.Parameter(torch.tensor(0.1))
        self.geometric_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead//2, dropout=dropout, batch_first=True
        )
        
    def forward(self, sequence_emb: torch.Tensor, 
                n_coords: torch.Tensor, ca_coords: torch.Tensor, c_coords: torch.Tensor,
                dihedrals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_emb: [B, L, seq_dim] sequence embeddings (always provided)
            n_coords: [B, L, 3] N atom coordinates
            ca_coords: [B, L, 3] Cα atom coordinates  
            c_coords: [B, L, 3] C atom coordinates
            dihedrals: [B, L, 6] sin/cos of phi/psi/omega angles
            mask: [B, L] valid residue mask
            
        Returns:
            [B, L, d_model] encoded features
        """
        B, L = ca_coords.shape[:2]
        
        # ========== STEP 1: Per-Residue Feature Encoding ==========
        # Concatenate full backbone (N, CA, C) for complete geometry
        backbone_coords = torch.cat([n_coords, ca_coords, c_coords], dim=-1)  # [B, L, 9]
        
        # Project and normalize features (not coordinates directly)
        coord_feat = self.coord_proj(backbone_coords)    # [B, L, d_model//4]
        coord_feat = self.coord_norm(coord_feat)         # Normalize features, not coords!
        
        dihedral_feat = self.dihedral_proj(dihedrals)    # [B, L, d_model//4]
        dihedral_feat = self.dihedral_norm(dihedral_feat)  # Normalize features
        
        seq_feat = self.seq_proj(sequence_emb)  # [B, L, d_model//2]
        combined = torch.cat([seq_feat, coord_feat, dihedral_feat], dim=-1)
            
        # Fuse features
        features = self.feature_fusion(combined)  # [B, L, d_model]
        
        # Add positional encoding
        features = self.pe(features)
        
        # ========== STEP 2: Geometric Attention (Local) ==========
        # Apply geometric attention for local structure awareness
        attn_mask = ~mask.bool() if mask is not None else None
        attn_out, _ = self.geometric_attention(
                features, features, features, 
                key_padding_mask=attn_mask
            )
        # Scale down attention output to prevent explosion
        features = features + self.geom_res_scale * attn_out
        
        # ========== STEP 3: Transformer Layers (Global) ==========
        # Apply transformer layers for global context
        padding_mask = ~mask.bool() if mask is not None else None
        
        for layer in self.transformer_layers:
            features = layer(features, src_key_padding_mask=padding_mask)
        
        # Final layer norm
        encoded = self.ln(features)
        
        return encoded


class HierLatent(nn.Module):
    """Hierarchical latent space with global and local components using attention-based pooling."""
    def __init__(self, d_model: int, z_g: int = 64, z_l: int = 32):
        super().__init__()
        self.d_model = d_model
        self.z_g = z_g
        self.z_l = z_l
        
        # Attention-based global pooling instead of simple mean
        self.global_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, dropout=0.1, batch_first=True
        )
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Global head with attention-weighted features
        self.global_head = nn.Sequential(
            nn.Linear(d_model, 256), 
            nn.ReLU(), 
            nn.Linear(256, 2*z_g)
        )
        
        # Local head (per residue)
        self.local_head = nn.Sequential(
            nn.Linear(d_model, 256), 
            nn.ReLU(), 
            nn.Linear(256, 2*z_l)
        )
        
        with torch.no_grad():
            # Initialize biases for stable training
            self.global_head[-1].bias[z_g:] = -2.0
            self.local_head[-1].bias[z_l:] = -2.0
            # Initialize global query with small values
            self.global_query.data.normal_(0, 0.02)

    def forward(self, H, mask):
        """
        H: [B,L,d_model]; mask: [B,L]
        Returns: mu_g, lv_g, mu_l, lv_l
        """
        B, L, _ = H.shape
        
        # ========== GLOBAL LATENT (Attention-based pooling) ==========
        # Create global query and compute attention weights
        global_query = self.global_query.expand(B, -1, -1)  # [B, 1, d_model]
        
        # Apply attention with mask
        attn_mask = ~mask.bool() if mask is not None else None
        global_features, attn_weights = self.global_attention(
            global_query, H, H, 
            key_padding_mask=attn_mask
        )
        
        # Global features from attention-weighted aggregation
        global_features = global_features.squeeze(1)  # [B, d_model]
        
        # Pass through global head
        g = self.global_head(global_features)  # [B, 2*z_g]
        mu_g, lv_g = torch.chunk(g, 2, dim=-1)
        
        # ========== LOCAL LATENT (Per residue) ==========
        l = self.local_head(H)  # [B, L, 2*z_l]
        mu_l, lv_l = torch.chunk(l, 2, dim=-1)
        
        return mu_g, lv_g, mu_l, lv_l


class ProteinEncoder(nn.Module):
    """
    Main encoder class that combines all encoding components.
    """
    def __init__(self, seqemb_dim: int,
                 d_model: int = 512, nhead: int = 8, ff: int = 1024, nlayers: int = 6,
                 z_g: int = 512, z_l: int = 256, dropout: float = 0.1,
                 use_dihedrals: bool = True):
        super().__init__()
        self.seqemb_dim = seqemb_dim
        self.use_dihedrals = use_dihedrals
        
        # Use simplified dihedral-aware encoder
        self.enc = DihedralAwareEncoder(
            seqemb_dim, dihedral_dim=6, d_model=d_model, 
            nhead=nhead, ff=ff, nlayers=nlayers, dropout=dropout
        )
            
        self.latent = HierLatent(d_model, z_g, z_l)

    def reparam(self, mu, lv):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, seqemb, n_coords, ca_coords, c_coords, dihedrals, mask):
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
            z_g: [B,zg] global latents
            z_l: [B,L,zl] local latents  
            mu_g, lv_g, mu_l, lv_l: latent parameters for KL loss
        """
        # Use dihedral-aware encoder with full backbone
        H = self.enc(seqemb, n_coords, ca_coords, c_coords, dihedrals, mask)  # [B,L,d_model]

        mu_g, lv_g, mu_l, lv_l = self.latent(H, mask)
        z_g = self.reparam(mu_g, lv_g)           # [B,zg]
        z_l = self.reparam(mu_l, lv_l)           # [B,L,zl]

        return z_g, z_l, mu_g, lv_g, mu_l, lv_l
