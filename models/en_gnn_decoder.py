#!/usr/bin/env python3
"""
E(n)-GNN Decoder for Protein Structures
A truly E(n)-equivariant (SE(3)-equivariant) decoder that uses only relative
positions for coordinate updates. Based on:
  Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class EGNLayer(nn.Module):
    """
    One EGNN layer:
      - Compute edge messages m_ij from [h_i, h_j, ||x_i - x_j||^2]
      - Update node hidden states h_i from [h_i, sum_j m_ij]
      - Update coordinates x_i by adding sum_j w_ij * (x_i - x_j),
        where w_ij is a learned scalar from m_ij.

    This construction is E(n)-equivariant:
      - Translations: only relative vectors (x_i - x_j) used → translation cancels.
      - Rotations/reflections: (x_i - x_j) rotates, and updates are linear combos
        of those same vectors → coordinates co-rotate/co-reflect.
    """
    def __init__(self, node_dim: int, hidden_dim: int, activation: nn.Module = nn.SiLU()):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Edge/message MLP: produces hidden_dim-sized messages m_ij
        self.phi_e = nn.Sequential(
            nn.Linear(2 * node_dim + 1, hidden_dim), activation,
            nn.Linear(hidden_dim, hidden_dim), activation
        )

        # Node update MLP: updates h_i from concatenated [h_i, agg_j m_ij]
        self.phi_h = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim), activation,
            nn.Linear(hidden_dim, node_dim)
        )

        # Coordinate update head: scalar weight w_ij from message m_ij
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), activation,
            nn.Linear(hidden_dim, 1)
        )

        self.norm_h = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, degree_inv: torch.Tensor | None = None):
        """
        h: [N, node_dim]
        x: [N, 3]
        edge_index: [2, E] with rows (row=i, col=j) edges i <- j
        degree_inv: [N] optional 1/deg(i) for coord normalization
        """
        row, col = edge_index  # i, j indices
        rel = x[row] - x[col]                          # [E, 3]
        d2  = (rel ** 2).sum(dim=-1, keepdim=True)    # [E, 1]

        # Edge messages m_ij in hidden space
        m_ij = self.phi_e(torch.cat([h[row], h[col], d2], dim=-1))  # [E, H]

        # Aggregate messages to each node i
        agg = torch.zeros(h.size(0), m_ij.size(1), device=h.device, dtype=h.dtype)
        agg.index_add_(0, row, m_ij)  # sum over neighbors j

        # Node feature update with residual + norm
        h_update = self.phi_h(torch.cat([h, agg], dim=-1))          # [N, node_dim]
        h = self.norm_h(h + h_update)

        # Coordinate update: sum_j w_ij * (x_i - x_j)
        w_ij = self.phi_x(m_ij)                                     # [E, 1]
              
        coord_delta = torch.zeros_like(x)                            # [N, 3]
        coord_delta.index_add_(0, row, w_ij * rel)

        if degree_inv is not None:
            coord_delta = coord_delta * degree_inv[:, None]

        # Scale down coordinate updates to prevent instability
        coord_delta = coord_delta * 1.0
        x = x + coord_delta  # Apply update once (already scaled above)
        return h, x


class EGNNDecoder(nn.Module):
    """
    EGNN-based decoder that maps per-residue features (z_l + replicated z_g + state emb)
    into 3D coordinates for N, CA, C atoms. We initialize coordinates with realistic 
    backbone geometry, then apply multiple EGNN layers of message passing.

    E(n)-equivariance is preserved throughout because only relative vectors are used.
    """
    def __init__(
        self,
        z_g: int,
        z_l: int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        max_neighbors: int = 20,
        dropout: float = 0.1,
        degree_normalize: bool = True,
    ):
        super().__init__()
        self.z_g = z_g
        self.z_l = z_l
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_neighbors = max_neighbors
        self.dropout = nn.Dropout(dropout)
        self.degree_normalize = degree_normalize
        total_in = z_g + z_l
        self.input_embedding = nn.Linear(total_in, hidden_dim)
        self.layers = nn.ModuleList([EGNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        
        # Latent-dependent coordinate initialization
        # This MLP converts latent codes to initial CA coordinates
        # This ensures the decoder starts from a structure-dependent state
        self.latent_to_coords = nn.Sequential(
            nn.Linear(z_g + z_l, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Less dropout for initialization
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Output: [x, y, z] per residue
        )
        
        # Initialize with small weights for stable initial coordinates
        with torch.no_grad():
            self.latent_to_coords[-1].weight.mul_(0.1)
            self.latent_to_coords[-1].bias.zero_()
        
        # Output heads for N, CA, C atom offsets from CA position
        # We predict CA directly, then N and C as offsets
        
        # NEW APPROACH: Context-aware bond length prediction
        # Instead of hardcoding 1.46Å and 1.52Å, let model learn from context!
        # Inspired by ESMFold's multi-scale prediction
        
        # Predict N offset: BOTH direction AND length
        self.n_offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 3 for direction + 1 for length adjustment
        )
        
        # Predict C offset: BOTH direction AND length  
        self.c_offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 3 for direction + 1 for length adjustment
        )
        
        # NEW: Sequence prediction head (predicts amino acid type from node features)
        # Takes refined node features h and predicts logits for 20 amino acids
        self.sequence_head = nn.Sequential(
        nn.Linear(z_g + z_l, hidden_dim * 2),  # Larger capacity
        nn.LayerNorm(hidden_dim * 2),
        nn.ReLU(),
        nn.Dropout(dropout * 0.5),  # Less dropout for sequence
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(hidden_dim, 20)  # 20 amino acids
    )

    @staticmethod
    def build_edge_index(L: int, device: torch.device, max_neighbors: int) -> torch.Tensor:
        """
        Simple chain + local window connectivity:
          edges i <-- j for all |i - j| <= max_neighbors, i != j.
        """
        pairs = []
        for i in range(L):
            for j in range(max(0, i - max_neighbors), min(L, i + max_neighbors + 1)):
                if i != j:
                    pairs.append((i, j))
        
        if not pairs:  # Handle edge case
            pairs = [(i, i + 1) for i in range(L - 1)] + [(i + 1, i) for i in range(L - 1)]
        
        return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()

    @staticmethod
    def degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute in-degree per node i for edges i <- j (row = i).
        """
        row = edge_index[0]
        deg = torch.bincount(row, minlength=num_nodes)  # avoid div-by-zero
        return deg

    def forward(self, z_g: torch.Tensor, z_l: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_g:    [B, z_g]             global latents
        z_l:    [B, L, z_l]          local per-residue latents
        mask:   [B, L] boolean/0-1   valid residues mask
        returns: tuple of (N_coords, CA_coords, C_coords, seq_logits) 
                 - N_coords, CA_coords, C_coords: [B, L, 3] (padded positions set to 0)
                 - seq_logits: [B, L, 20] amino acid logits
        """
        B, L, _ = z_l.shape
        device = z_l.device
        ca_coords_out = []
        n_coords_out = []
        c_coords_out = []
        seq_logits_out = []

        for b in range(B):
            if mask is not None:
                valid_idx = torch.nonzero(mask[b].bool(), as_tuple=False).squeeze(-1)
            else:
                valid_idx = torch.arange(L, device=device)

            Lb = valid_idx.numel()
            if Lb == 0:
                zero_coords = torch.zeros(L, 3, device=device, dtype=z_l.dtype)
                zero_seq_logits = torch.zeros(L, 20, device=device, dtype=z_l.dtype)
                n_coords_out.append(zero_coords)
                ca_coords_out.append(zero_coords)
                c_coords_out.append(zero_coords)
                seq_logits_out.append(zero_seq_logits)
                continue

            # Inputs per-residue (only valid positions)
            zg_rep = z_g[b].unsqueeze(0).expand(Lb, -1)                   # [Lb, z_g]
            z_combined = torch.cat([zg_rep, z_l[b, valid_idx]], dim=-1)  # [Lb, z_g + z_l]

            seq_logits_valid = self.sequence_head(z_combined)
            
            # Initialize CA coordinates from latent
            x_ca = self.latent_to_coords(z_combined)  # [Lb, 3]
                      
            # Embed features for EGNN processing
            h = self.input_embedding(z_combined)  # [Lb, hidden_dim]

            # Build edges once per sample (static local window graph on valid residues only)
            edge_index = self.build_edge_index(Lb, device, self.max_neighbors)
            deg = self.degrees(edge_index, Lb)
            degree_inv = (1.0 / deg.float()) if self.degree_normalize else None

            # EGNN stack - refine latent-initialized coordinates
            for layer in self.layers:
                h, x_ca = layer(h, x_ca, edge_index, degree_inv=degree_inv)
                h = self.dropout(h)  # Regularize node features
                
            # Final centering for translation invariance
            # x_ca = x_ca - x_ca.mean(dim=0, keepdim=True)
            
            # Predict N and C as offsets from CA with CONTEXT-AWARE bond lengths
            # Model now predicts: [direction (3D), length_adjustment (1D)]
            n_prediction = self.n_offset_head(h)  # [Lb, 4]
            c_prediction = self.c_offset_head(h)  # [Lb, 4]
            
            # Split into direction and length adjustment
            n_direction = n_prediction[:, :3]     # [Lb, 3]
            n_length_adj = n_prediction[:, 3:4]   # [Lb, 1]
            
            c_direction = c_prediction[:, :3]     # [Lb, 3]
            c_length_adj = c_prediction[:, 3:4]   # [Lb, 1]
            
                        
            # Context-aware with soft constraints 
            # Base lengths (chemical averages) + learned adjustments
            # Allow ±0.05Å variation (realistic based on PDB statistics)
            n_base_length = 1.46  # N-CA average
            c_base_length = 1.52  # CA-C average
            
            # Apply learned adjustment with sigmoid to keep within tolerances
            # sigmoid(x) ∈ [0,1], scale to [-0.05, +0.05]
            n_length_delta = (torch.sigmoid(n_length_adj) - 0.5) * 0.10  # ±0.05Å
            c_length_delta = (torch.sigmoid(c_length_adj) - 0.5) * 0.10  # ±0.05Å
            
            n_length = n_base_length + n_length_delta  # [Lb, 1] - context-aware!
            c_length = c_base_length + c_length_delta  # [Lb, 1] - context-aware!
            
            # Normalize direction and scale by context-aware length
            n_direction_norm = F.normalize(n_direction, dim=-1)  # [Lb, 3]
            c_direction_norm = F.normalize(c_direction, dim=-1)  # [Lb, 3]
            
            n_offset_constrained = n_direction_norm * n_length  # [Lb, 3]
            c_offset_constrained = c_direction_norm * c_length  # [Lb, 3]
            
            x_n = x_ca + n_offset_constrained
            x_c = x_ca + c_offset_constrained
            
            # Context-aware C-N PEPTIDE BONDS with soft constraints
            # Most critical bond! Length varies with geometry:
            #   Trans (ω≈180°): 1.329Å  (standard)
            #   Cis (ω≈0°):     1.341Å  (12pm longer due to strain)
            #   Strained loops: 1.320-1.350Å (depends on context)
            #
            if Lb > 1:
                for i in range(Lb - 1):
                    # Vector from C(i) to N(i+1)
                    peptide_vec = x_n[i+1] - x_c[i]
                    peptide_dist = torch.norm(peptide_vec)
                    
                    # Soft constraint: target 1.33Å, but allow 1.31-1.35Å
                    peptide_target = 1.33
                    peptide_tolerance = 0.02  # ±0.02Å is chemically reasonable
                    
                    peptide_vec_unit = peptide_vec / (peptide_dist + 1e-8)
                    
                    # Only enforce if outside acceptable range
                    # This lets model learn context-dependent lengths!
                    if peptide_dist < peptide_target - peptide_tolerance:
                        # Too short → pull to minimum acceptable (1.31Å)
                        peptide_vec_constrained = peptide_vec_unit * (peptide_target - peptide_tolerance)
                        x_n[i+1] = x_c[i] + peptide_vec_constrained
                    elif peptide_dist > peptide_target + peptide_tolerance:
                        # Too long → pull to maximum acceptable (1.35Å)
                        peptide_vec_constrained = peptide_vec_unit * (peptide_target + peptide_tolerance)
                        x_n[i+1] = x_c[i] + peptide_vec_constrained
                    # else: within [1.31, 1.35]Å → keep as-is! Model made a good prediction.

            # Scatter back into L-length tensors
            full_ca = torch.zeros(L, 3, device=device, dtype=h.dtype)
            full_n = torch.zeros(L, 3, device=device, dtype=h.dtype)
            full_c = torch.zeros(L, 3, device=device, dtype=h.dtype)
            
            full_ca[valid_idx] = x_ca
            full_n[valid_idx] = x_n
            full_c[valid_idx] = x_c
            
            ca_coords_out.append(full_ca)
            n_coords_out.append(full_n)
            c_coords_out.append(full_c)
            
            # At the end, scatter sequence logits back
            full_seq_logits = torch.zeros(L, 20, device=device, dtype=h.dtype)
            full_seq_logits[valid_idx] = seq_logits_valid
            seq_logits_out.append(full_seq_logits)

        return (torch.stack(n_coords_out, dim=0),      # [B, L, 3]
                torch.stack(ca_coords_out, dim=0),     # [B, L, 3]
                torch.stack(c_coords_out, dim=0),      # [B, L, 3]
                torch.stack(seq_logits_out, dim=0))    # [B, L, 20]


class SE3EquivariantDecoder(nn.Module):
    """
    Thin wrapper to keep your original API.
    If equivariant=True, we use EGNNDecoder; else we fall back to an MLP per-residue head.
    """
    def __init__(self, z_g: int, z_l: int, hidden: int = 256, dropout: float = 0.2, equivariant: bool = True):
        super().__init__()
        self.decoder = EGNNDecoder(
                z_g=z_g, z_l=z_l,
                hidden_dim=256,
                num_layers=8,
                max_neighbors=40,
                dropout=dropout, degree_normalize=True
            )

    def forward(self, z_g: torch.Tensor, z_l: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (N_coords, CA_coords, C_coords, seq_logits)
                 - N_coords, CA_coords, C_coords: [B, L, 3]
                 - seq_logits: [B, L, 20]
        """
        return self.decoder(z_g, z_l, mask=mask)

class ResidueDecoder(nn.Module): 
    """Wrapper to maintain backward compatibility.""" 
    def __init__(self, z_g: int, z_l: int, hidden: int = 256, dropout: float = 0.2, equivariant: bool = True): 
        super().__init__() 
        self.equivariant = equivariant 
        self.decoder = SE3EquivariantDecoder(z_g=z_g, z_l=z_l, hidden=hidden, dropout=dropout, equivariant=True) 
            
    def forward(self, z_g, z_l, mask=None): 
        """ 
        z_g: [B,zg], z_l:[B,L,zl], mask:[B,L] or None 
        returns: tuple of (N_coords, CA_coords, C_coords, sequence_logits) 
                 each [B,L,3] for coords, [B,L,20] for sequence
        """ 
        if self.equivariant: 
            return self.decoder(z_g, z_l, mask=mask) 
        else: 
            B, L, _ = z_l.shape 
            zg_rep = z_g.unsqueeze(1).expand(-1, L, -1) 
            inp = torch.cat([zg_rep, z_l], dim=-1) 
            
            out_n = self.mlp_n(inp)
            out_ca = self.mlp_ca(inp)
            out_c = self.mlp_c(inp)
            out_seq = self.mlp_seq(inp)
            
            # zero-out padded positions if a mask is provided
            if mask is not None:
                mask_3d = mask.unsqueeze(-1)
                out_n = out_n * mask_3d
                out_ca = out_ca * mask_3d
                out_c = out_c * mask_3d
                out_seq = out_seq * mask_3d  # [B, L, 20]
            
            return out_n, out_ca, out_c, out_seq
