#!/usr/bin/env python3
"""
Loss Functions Module
Contains all loss functions for the protein VAE model.
"""

import torch
import torch.nn.functional as F
import math


def rmsd_loss(pred, target, mask):
    """
    MSE-based coordinate loss.
    Returns MSE (not RMSD) for better gradient stability.
    """
    #pred_aln = kabsch_align(pred, target, mask)
    diff = (pred - target).pow(2).sum(-1)  # [B,L]
    mse = (diff * mask).sum(dim=1) / mask.sum(dim=1)  # [B]
    
    return mse.mean()  # Returns mean squared error in Ų


def pair_distance_loss(pred, target, mask, stride=4, min_sep=2):
    """
    Subsample residues every 'stride' to stabilize long-range geometry.
    Uses squared error for consistency with reconstruction loss.
    """
    idx = torch.arange(0, pred.size(1), stride, device=pred.device)
    P = pred[:, idx, :]     # [B,M,3]
    T = target[:, idx, :]
    m = mask[:, idx]
    # pair masks
    M = (m[:, :, None] * m[:, None, :]).float()  # [B,M,M]
    dP = torch.cdist(P, P)                       # [B,M,M]
    dT = torch.cdist(T, T)
    return ((dP - dT).abs() * M).sum() / M.sum()


def _kl_unit_gauss(mu, lv, reduce_dims=None):
    """KL divergence with unit Gaussian prior."""
    # KL(q||p) for diag Gaussians with p=N(0,I): 
    kl = 0.5 * (lv.exp() + mu.pow(2) - 1.0 - lv)
    if reduce_dims is None:
        return kl
    return kl.sum(dim=reduce_dims)


def kl_global(mu, lv):
    """Global KL divergence loss."""
    return _kl_unit_gauss(mu, lv, reduce_dims=1).mean()


def kl_local(mu, lv, mask):
    """Local KL divergence loss."""
    kl = _kl_unit_gauss(mu, lv, reduce_dims=-1)      # [B,L]
    return (kl * mask).sum() / mask.sum()


def dihedral_consistency_loss(pred_dihedrals, target_dihedrals, mask):
    """Dihedral consistency loss between predicted and target dihedrals."""   
    if pred_dihedrals is None or target_dihedrals is None: 
        return torch.tensor(0.0, device=mask.device)
    valid = (mask.unsqueeze(-1).bool() &
             torch.isfinite(pred_dihedrals) &
             torch.isfinite(target_dihedrals))
    diff  = torch.where(valid, pred_dihedrals - target_dihedrals, torch.zeros_like(pred_dihedrals))
    den   = valid.float().sum()
    return (diff.pow(2).sum() / den)


def ramachandran_loss(dihedrals, mask, aa_types=None):
    """
    Simplified Ramachandran loss focusing on forbidden regions only.
    More computationally efficient and training-stable for VAE.
    
    Args:
        dihedrals: [B, L, 6] sin/cos values for φ, ψ, ω
        mask: [B, L] validity mask
        aa_types: [B, L] amino acid types (optional, for future use)
        
    Returns:
        Ramachandran penalty
    """
    if dihedrals.numel() == 0:
        return torch.tensor(0.0, device=mask.device)
    
    phi = torch.atan2(dihedrals[..., 0], dihedrals[..., 1])
    psi = torch.atan2(dihedrals[..., 2], dihedrals[..., 3])
    
    # Alpha helix: φ ≈ -60° ± 20°, ψ ≈ -45° ± 20° (wider allowance)
    alpha_helix = torch.exp(-(
        (phi + 1.05)**2 / 0.6 +   # Width increased: 0.25 → 0.6
        (psi + 0.79)**2 / 0.6
    ))
    
    # Beta sheet: φ ≈ -120° ± 30°, ψ ≈ +120° ± 30°
    beta_sheet = torch.exp(-(
        (phi + 2.09)**2 / 0.9 +    # Width increased: 0.25 → 0.9
        (psi - 2.09)**2 / 0.9
    ))
    
    # Left-handed alpha (allowed for Gly): φ ≈ +60°, ψ ≈ +45°
    left_alpha = torch.exp(-(
        (phi - 1.05)**2 / 0.6 +
        (psi - 0.79)**2 / 0.6
    ))
    
    # Polyproline II (common in coils): φ ≈ -75°, ψ ≈ +145°
    ppII = torch.exp(-(
        (phi + 1.31)**2 / 0.5 +
        (psi - 2.53)**2 / 0.5
    ))
    
    # Combine all allowed regions
    in_allowed = torch.maximum(
        torch.maximum(alpha_helix, beta_sheet),
        torch.maximum(left_alpha, ppII)
    )
    
    # Penalty = 1 - (how much in allowed)
    penalty = 1.0 - in_allowed
    
    # Add STRONG penalty for highly forbidden regions
    # (e.g., positive phi in non-Gly residues)
    forbidden_mask = (phi > 0) & (psi < 0)  # Forbidden quadrant
    forbidden_penalty = 5.0 * forbidden_mask.float()  # Heavy penalty
    
    total_penalty = penalty + forbidden_penalty
    
    return (total_penalty * mask).sum() / mask.sum()

def ang_wrap(x):  # maps to (-π, π]
    return torch.atan2(torch.sin(x), torch.cos(x))

def omega_trans_loss(dihedrals, mask):
    """
    Encourage trans peptide bonds (omega ~ 180°).
    """
    if dihedrals.numel() == 0:
        return torch.tensor(0.0, device=mask.device)
        
    # Extract omega angles (peptide bond dihedrals)
    sin_omega, cos_omega = dihedrals[..., 4], dihedrals[..., 5]
    omega = torch.atan2(sin_omega, cos_omega)
    
    trans_penalty = 1.0 - torch.cos(omega - torch.pi)       # zero at 180°, smooth
    cis_indicator = (torch.abs(ang_wrap(omega)) < 0.5).float()
    cis_penalty = 3.0 * cis_indicator                        # keep if you like the extra push
    total_penalty = 2.0 * trans_penalty + cis_penalty
    
    # Apply mask and return mean
    masked_penalty = (total_penalty * mask).sum() / mask.sum()
    
    return masked_penalty


def _dihedral_from_four(p0, p1, p2, p3, eps=1e-8):
    """
    Compute dihedral angles from four points using proper torsion angle formula.
    
    This implements the standard protein backbone dihedral angle computation:
    - Properly handles the torsion angle between two planes
    - Uses robust numerical methods for edge cases
    - Returns sin/cos representation for stability
    
    Args:
        p0, p1, p2, p3: [B, M, 3] coordinates defining the dihedral angle
        eps: numerical stability constant
        
    Returns:
        sin_angles, cos_angles: [B, M] sin and cos of dihedral angles
    """
    # Compute bond vectors
    b1 = p1 - p0  # Vector from p0 to p1
    b2 = p2 - p1  # Vector from p1 to p2  
    b3 = p3 - p2  # Vector from p2 to p3
    
    # Compute normal vectors to the two planes
    # Plane 1: defined by b1 and b2
    n1 = torch.cross(b1, b2, dim=-1)
    # Plane 2: defined by b2 and b3  
    n2 = torch.cross(b2, b3, dim=-1)
    
    # Normalize normal vectors
    n1_norm = torch.norm(n1, dim=-1, keepdim=True)
    n2_norm = torch.norm(n2, dim=-1, keepdim=True)
    
    # Check for collinear vectors (degenerate cases)
    valid_mask = (n1_norm.squeeze(-1) > eps) & (n2_norm.squeeze(-1) > eps)
    
    # Initialize output tensors
    sin_angles = torch.zeros_like(b1[..., 0])
    cos_angles = torch.ones_like(b1[..., 0])
    
    if valid_mask.any():
        # Normalize only valid vectors
        n1_unit = torch.where(
            valid_mask.unsqueeze(-1), 
            n1 / (n1_norm + eps), 
            torch.zeros_like(n1)
        )
        n2_unit = torch.where(
            valid_mask.unsqueeze(-1),
            n2 / (n2_norm + eps), 
            torch.zeros_like(n2)
        )
        
        # Normalize b2 for proper sign determination
        b2_norm = torch.norm(b2, dim=-1, keepdim=True)
        b2_unit = torch.where(
            valid_mask.unsqueeze(-1),
            b2 / (b2_norm + eps),
            torch.zeros_like(b2)
        )
        
        # Compute cos(angle) = n1 · n2
        cos_angle = (n1_unit * n2_unit).sum(dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
        
        # Compute sin(angle) using the sign of the mixed product
        # sin(angle) = sign((n1 × n2) · b2) * sqrt(1 - cos²(angle))
        cross_product = torch.cross(n1_unit, n2_unit, dim=-1)
        mixed_product = (cross_product * b2_unit).sum(dim=-1)
        
        sin_angle = torch.sign(mixed_product) * torch.sqrt(1.0 - cos_angle**2 + eps)
        
        # Update valid positions
        sin_angles = torch.where(valid_mask, sin_angle, sin_angles)
        cos_angles = torch.where(valid_mask, cos_angle, cos_angles)
    
    return sin_angles, cos_angles


def compute_dihedrals_from_coords(N, CA, C, mask):
    """
    Compute TRUE backbone dihedral angles from N, CA, C coordinates.
    
    This computes the actual Ramachandran φ/ψ angles used in protein structure:
    - φ (phi): C(i-1) - N(i) - CA(i) - C(i)
    - ψ (psi): N(i) - CA(i) - C(i) - N(i+1)
    - ω (omega): CA(i-1) - C(i-1) - N(i) - CA(i)
    
    Args:
        N: [B, L, 3] N atom coordinates
        CA: [B, L, 3] CA atom coordinates  
        C: [B, L, 3] C atom coordinates
        mask: [B, L] validity mask
        
    Returns:
        [B, L, 6] tensor of sin/cos values: [sin_phi, cos_phi, sin_psi, cos_psi, sin_omega, cos_omega]
    """
    B, L, _ = CA.shape
    device, dtype = CA.device, CA.dtype
    out = torch.zeros(B, L, 6, device=device, dtype=dtype)
    if L < 2: return out

    m = mask.bool()

    # Compute φ angles: C(i-1) - N(i) - CA(i) - C(i)
    # φ is defined for residues 1 to L-1 (need previous C)
    if L >= 2:
        phi_p0 = C[:, :-1]   # C(i-1)
        phi_p1 = N[:, 1:]    # N(i)
        phi_p2 = CA[:, 1:]   # CA(i)
        phi_p3 = C[:, 1:]    # C(i)
        
        # Mask: need both residue i and i-1 to be valid
        phi_mask = m[:, :-1] & m[:, 1:]
        phi_sin, phi_cos = _dihedral_from_four(phi_p0, phi_p1, phi_p2, phi_p3)
        
        # Store φ angles at position i (starting from i=1)
        out[:, 1:, 0] = torch.where(phi_mask, phi_sin, torch.zeros_like(phi_sin))
        out[:, 1:, 1] = torch.where(phi_mask, phi_cos, torch.zeros_like(phi_cos))

    # Compute ψ angles: N(i) - CA(i) - C(i) - N(i+1)
    # ψ is defined for residues 0 to L-2 (need next N)
    if L >= 2:
        psi_p0 = N[:, :-1]   # N(i)
        psi_p1 = CA[:, :-1]  # CA(i)
        psi_p2 = C[:, :-1]   # C(i)
        psi_p3 = N[:, 1:]    # N(i+1)
        
        # Mask: need both residue i and i+1 to be valid
        psi_mask = m[:, :-1] & m[:, 1:]
        psi_sin, psi_cos = _dihedral_from_four(psi_p0, psi_p1, psi_p2, psi_p3)
        
        # Store ψ angles at position i (ending at i=L-2)
        out[:, :-1, 2] = torch.where(psi_mask, psi_sin, torch.zeros_like(psi_sin))
        out[:, :-1, 3] = torch.where(psi_mask, psi_cos, torch.zeros_like(psi_cos))

    # Compute ω (omega) angles: CA(i-1) - C(i-1) - N(i) - CA(i)
    # ω is the peptide bond dihedral, defined for residues 1 to L-1
    if L >= 2:
        omega_p0 = CA[:, :-1]  # CA(i-1)
        omega_p1 = C[:, :-1]   # C(i-1)
        omega_p2 = N[:, 1:]    # N(i)
        omega_p3 = CA[:, 1:]   # CA(i)
        
        # Mask: need both residue i and i-1 to be valid
        omega_mask = m[:, :-1] & m[:, 1:]
        omega_sin, omega_cos = _dihedral_from_four(omega_p0, omega_p1, omega_p2, omega_p3)
        
        # Store ω angles at position i (starting from i=1)
        out[:, 1:, 4] = torch.where(omega_mask, omega_sin, torch.zeros_like(omega_sin))
        out[:, 1:, 5] = torch.where(omega_mask, omega_cos, torch.zeros_like(omega_cos))

    return out


def huber_loss(x, delta=0.2):
    """Huber loss for robust optimization."""
    abs_x = torch.abs(x)
    return torch.where(abs_x < delta, 
                      0.5 * x**2, 
                      delta * (abs_x - 0.5 * delta))

def bond_length_loss(pred_N, pred_CA, pred_C, mask):
    """
    Enforce realistic bond lengths in the predicted backbone.
    
    Typical protein backbone bond lengths:
    - N-CA: 1.46 Å
    - CA-C: 1.52 Å
    - C-N (peptide bond): 1.33 Å
    
    Args:
        pred_N, pred_CA, pred_C: [B, L, 3] predicted backbone coordinates
        mask: [B, L] validity mask
        
    Returns:
        Bond length penalty
    """
    # N-CA bond lengths - use Huber loss for stability
    n_ca_dists = torch.norm(pred_CA - pred_N, dim=-1)  # [B, L]
    n_ca_error = n_ca_dists - 1.46
    n_ca_penalty = (huber_loss(n_ca_error, delta=0.02) * mask).sum() / mask.sum()
    
    # CA-C bond lengths
    ca_c_dists = torch.norm(pred_C - pred_CA, dim=-1)  # [B, L]
    ca_c_error = ca_c_dists - 1.52
    ca_c_penalty = (huber_loss(ca_c_error, delta=0.02) * mask).sum() / mask.sum()
    
    # C-N peptide bond (between residues) - most critical
    if pred_N.shape[1] > 1:
        c_n_dists = torch.norm(pred_N[:, 1:] - pred_C[:, :-1], dim=-1)  # [B, L-1]
        c_n_error = c_n_dists - 1.33
        mask_peptide = mask[:, :-1] * mask[:, 1:]  # Both residues must be valid
        
        # Use Huber loss with tighter delta for peptide bonds
        c_n_penalty = (huber_loss(c_n_error, delta=0.01) * mask_peptide).sum() / mask_peptide.sum()
    else:
        c_n_penalty = torch.tensor(0.0, device=pred_N.device)
    
    return n_ca_penalty + ca_c_penalty + 2*c_n_penalty


def _angle_cos(A, B, C, eps=1e-8):
    """
    Cosine of angle ABC (at vertex B).
    A,B,C: [..., 3]
    Returns: [...], clipped to [-1, 1]
    """
    BA = A - B
    BC = C - B
    BA = BA / (BA.norm(dim=-1, keepdim=True) + eps)
    BC = BC / (BC.norm(dim=-1, keepdim=True) + eps)
    return torch.clamp((BA * BC).sum(dim=-1), -1.0, 1.0)


def bond_angle_loss(pred_N, pred_CA, pred_C, mask):
    """Enforce backbone angles using Huber loss in angle space."""
    device = pred_CA.device
    mask = mask.float()
    
    # Target angles in RADIANS
    TARGET_NCAC = 110.0 * torch.pi / 180.0  # 1.919 rad
    TARGET_CNNCA = 121.0 * torch.pi / 180.0  # 2.111 rad
    TARGET_CACN = 116.0 * torch.pi / 180.0  # 2.024 rad
    
    # 1) N–Cα–C angle
    cos_ncac = _angle_cos(pred_N, pred_CA, pred_C)
    angle_ncac = torch.acos(torch.clamp(cos_ncac, -1.0, 1.0))  # [B, L]
    error_ncac = angle_ncac - TARGET_NCAC
    loss_ncac = (huber_loss(error_ncac, delta=0.1) * mask).sum() / mask.sum()
    
    # 2) C–N–Cα angle (inter-residue)
    if pred_N.shape[1] > 1:
        cos_cnnca = _angle_cos(pred_C[:, :-1], pred_N[:, 1:], pred_CA[:, 1:])
        angle_cnnca = torch.acos(torch.clamp(cos_cnnca, -1.0, 1.0))
        error_cnnca = angle_cnnca - TARGET_CNNCA
        mask_cnnca = mask[:, :-1] * mask[:, 1:]
        loss_cnnca = (huber_loss(error_cnnca, delta=0.1) * mask_cnnca).sum() / mask_cnnca.sum()
    else:
        loss_cnnca = torch.tensor(0.0, device=device)
    
    # 3) Cα–C–N angle (inter-residue)
    if pred_N.shape[1] > 1:
        cos_cacn = _angle_cos(pred_CA[:, :-1], pred_C[:, :-1], pred_N[:, 1:])
        angle_cacn = torch.acos(torch.clamp(cos_cacn, -1.0, 1.0))
        error_cacn = angle_cacn - TARGET_CACN
        mask_cacn = mask[:, :-1] * mask[:, 1:]
        loss_cacn = (huber_loss(error_cacn, delta=0.1) * mask_cacn).sum() / mask_cacn.sum()
    else:
        loss_cacn = torch.tensor(0.0, device=device)
    
    # Weight inter-residue angles 3x more (they're more important)
    return loss_ncac + 2.0 * (loss_cnnca + loss_cacn)


def sequence_classification_loss(pred_seq_logits, target_seq_labels, mask):
    """
    Compute sequence classification loss (cross-entropy) for amino acid prediction.
    
    Args:
        pred_seq_logits: [B, L, 20] predicted amino acid logits
        target_seq_labels: [B, L] target amino acid labels (0-19)
        mask: [B, L] valid residue mask
        
    Returns:
        loss: scalar cross-entropy loss
    """
    B, L, num_classes = pred_seq_logits.shape
    
    # Reshape for cross-entropy: [B*L, 20] and [B*L]
    pred_flat = pred_seq_logits.reshape(B * L, num_classes)
    target_flat = target_seq_labels.reshape(B * L)
    mask_flat = mask.reshape(B * L)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(pred_flat, target_flat, reduction='none')  # [B*L]
    
    # Apply mask and average
    loss_masked = loss * mask_flat
    num_valid = mask_flat.sum() + 1e-8
    
    return loss_masked.sum() / num_valid

def clash_loss(pred_N, pred_CA, pred_C, mask, clash_dist=3.2, soft_margin=0.5):
    """
    FAST VECTORIZED clash loss - penalizes steric clashes without Python loops.
    
    Van der Waals radii:
    - N: 1.55Å, C: 1.70Å, O: 1.52Å
    - Min separation: ~3.0Å between heavy atoms
    - We use 2.5Å as hard threshold for clashes
    
    Args:
        pred_N, pred_CA, pred_C: [B, L, 3] backbone coordinates
        mask: [B, L] validity mask
        clash_dist: minimum allowed distance (Å) - default 2.5Å
        soft_margin: soft margin for differentiability
        
    Returns:
        Scalar clash penalty (mean over batch)
    """
    B, L = pred_CA.shape[:2]
    device = pred_CA.device
    
    # Stack all atoms: [B, L, 3, 3] where dim=2 is [N, CA, C]
    all_atoms = torch.stack([pred_N, pred_CA, pred_C], dim=2)  # [B, L, 3, 3]
    all_atoms = all_atoms.reshape(B, L * 3, 3)  # [B, 3*L, 3]
    
    # Create atom mask: [B, 3*L]
    atom_mask = torch.stack([mask, mask, mask], dim=2).reshape(B, L * 3)
    
    # Compute pairwise distances for all atoms in batch
    # [B, 3*L, 3*L]
    dists = torch.cdist(all_atoms, all_atoms)  # Euclidean distance
    
    # Create sequence separation mask
    # We need to ignore atoms that are close in sequence (bonded neighbors)
    # Atoms within same residue or adjacent residues are allowed to be close
    atom_indices = torch.arange(L * 3, device=device)  # [3*L]
    residue_indices = atom_indices // 3  # Which residue each atom belongs to
    
    # Compute residue separation: [3*L, 3*L]
    res_sep = torch.abs(residue_indices[:, None] - residue_indices[None, :])
    
    # Atoms must be at least 2 residues apart (|i-j| >= 2) to be checked for clashes
    # This allows bonded and next-neighbor atoms to be close
    separation_mask = (res_sep >= 2).float()  # [3*L, 3*L]
    
    # Upper triangle mask to avoid double counting
    triu_mask = torch.triu(torch.ones(L * 3, L * 3, device=device), diagonal=1)
    
    # Combine masks: [B, 3*L, 3*L]
    # Valid pairs: both atoms valid, sufficient separation, upper triangle
    pair_mask = (atom_mask[:, :, None] * atom_mask[:, None, :])  # [B, 3*L, 3*L]
    pair_mask = pair_mask * separation_mask[None, :, :]  # Apply sequence separation
    pair_mask = pair_mask * triu_mask[None, :, :]  # Apply upper triangle
    
    # Compute clash violations: [B, 3*L, 3*L]
    violations = clash_dist - dists  # Positive = clash
    violations = torch.relu(violations)  # Only keep distances < clash_dist
    
    # Apply soft margin for smooth gradients (Huber-like loss)
    # For small violations: quadratic
    # For large violations: linear
    clash_penalty = torch.where(
        violations < soft_margin,
        0.5 * violations ** 2,
        violations**2 # Quadratic for all! (was linear)
    )
    
    # Apply pair mask and compute mean
    masked_penalty = clash_penalty * pair_mask
    
    # Sum over pairs and normalize
    total_clash = masked_penalty.sum(dim=[1, 2])  # [B]
    num_pairs = pair_mask.sum(dim=[1, 2])  # [B]
    
    # Average over batch (only count samples with valid pairs)
    loss = total_clash / (num_pairs + 1e-8)
    loss = loss.mean()
    
    return loss


def compute_total_loss(pred_N, pred_CA, pred_C, pred_seq, target_N, target_CA, target_C, target_seq_labels,
                      mask, mu_g, lv_g, mu_l, lv_l,
                      target_dihedrals, klw_g, klw_l, w_pair, pair_stride,
                      w_dihedral, w_rama, w_bond, w_angle, w_rec, w_seq, w_clash):
    """
    Compute total loss with improved gradient flow and stability.
    
    Args:
        pred_N, pred_CA, pred_C: [B,L,3] predicted N, CA, C coordinates
        pred_seq: [B,L,20] predicted amino acid logits
        target_N, target_CA, target_C: [B,L,3] target N, CA, C coordinates
        target_seq_labels: [B,L] target amino acid labels (0-19)
        mask: [B,L] valid residue mask
        mu_g, lv_g, mu_l, lv_l: latent parameters
        target_dihedrals: [B,L,6] target dihedral angles
        klw_g, klw_l: KL loss weights
        w_pair: pair distance loss weight
        pair_stride: stride for pair distance loss
        w_dihedral: dihedral loss weight
        w_rama: ramachandran loss weight
        w_bond: bond length constraint weight
        w_angle: bond angle constraint weight
        w_rec: reconstruction loss weight
        w_seq: sequence prediction loss weight (default: 1.0)
        
    Returns:
        dict with individual loss components and total loss
    """
    # Primary reconstruction loss (focus on CA atoms for ensemble VAE)
    loss_rec_ca = rmsd_loss(pred_CA, target_CA, mask)
    
    # Secondary reconstruction losses (lighter weight)
    loss_rec_n = rmsd_loss(pred_N, target_N, mask)
    loss_rec_c = rmsd_loss(pred_C, target_C, mask)
    loss_rec = loss_rec_ca + 0.5 * (loss_rec_n + loss_rec_c)
    
    # Long-range structure preservation
    loss_pair = pair_distance_loss(pred_CA, target_CA, mask, stride=pair_stride)
    
    # KL losses
    loss_kg = kl_global(mu_g, lv_g)
    loss_kl = kl_local(mu_l, lv_l, mask)

    # Predicted dihedrals from predicted coords (TRUE BACKBONE DIHEDRALS!)
    pred_dih = compute_dihedrals_from_coords(pred_N, pred_CA, pred_C, mask)  # [B,L,6]

    # Consistency w.r.t. target dihedrals
    loss_dih_cons = dihedral_consistency_loss(pred_dih, target_dihedrals, mask)

    # Physics priors on predicted torsions
    loss_rama_pred = ramachandran_loss(pred_dih, mask)
    loss_omega_pred = omega_trans_loss(pred_dih, mask)
    loss_dihedral = loss_dih_cons + loss_omega_pred

    # Geometric constraints
    loss_bond = bond_length_loss(pred_N, pred_CA, pred_C, mask)
    loss_angle = bond_angle_loss(pred_N, pred_CA, pred_C, mask)
    
    # Sequence prediction loss
    loss_seq = sequence_classification_loss(pred_seq, target_seq_labels, mask)

    # Clash loss
    loss_clash = clash_loss(pred_N, pred_CA, pred_C, mask)

    # Weighted total (w_rec gives explicit weight to reconstruction)
    loss = w_rec * loss_rec \
         + w_pair * loss_pair \
         + klw_g  * loss_kg \
         + klw_l  * loss_kl \
         + w_dihedral * loss_dihedral \
         + w_rama     * loss_rama_pred \
         + w_bond     * loss_bond \
         + w_angle    * loss_angle \
         + w_seq      * loss_seq \
         + w_clash    * loss_clash

    return {
        'total': loss,
        'reconstruction': loss_rec,
        'reconstruction_ca': loss_rec_ca,
        'reconstruction_n': loss_rec_n,
        'reconstruction_c': loss_rec_c,
        'pair_distance': loss_pair,
        'kl_global': loss_kg,
        'kl_local': loss_kl,
        'dihedral_consistency': loss_dih_cons,
        'omega_trans': loss_omega_pred,
        'ramachandran': loss_rama_pred,
        'dihedral_total': loss_dihedral,
        'bond_length': loss_bond,
        'bond_angle': loss_angle,
        'sequence': loss_seq,
        'clash': loss_clash
    }
