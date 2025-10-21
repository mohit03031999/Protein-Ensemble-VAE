#!/usr/bin/env python3
"""
Comprehensive validation metrics for protein ensemble generation.
Includes TM-score, lDDT, GDT, RMSF, and physical plausibility checks.

Usage:
    python validation_metrics.py --pred structure_pred.pdb --true structure_true.pdb
    python validation_metrics.py --ensemble ensemble_dir/ --output validation_report.txt
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import argparse
import os
from pathlib import Path


# ============================================================================
# TM-SCORE (Topology Validation)
# ============================================================================

def compute_tm_score_python(coords_pred, coords_true):
    """
    Python implementation of TM-score (approximation).
    
    TM-score formula:
        TM = (1/L_target) * Σ [ 1 / (1 + (d_i/d_0)²) ]
    
    where d_0 = 1.24 * ∛(L-15) - 1.8
    
    Args:
        coords_pred: [L, 3] predicted CA coordinates
        coords_true: [L, 3] true CA coordinates
    
    Returns:
        tm_score: float in [0, 1]
            > 0.5: Same fold
            > 0.7: Good model
            > 0.9: Excellent model
    """
    L = len(coords_true)
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    
    # Kabsch alignment
    coords_pred_aligned = kabsch_align(coords_pred, coords_true)
    
    # Compute distances after alignment
    distances = np.linalg.norm(coords_pred_aligned - coords_true, axis=1)
    
    # TM-score
    tm_score = np.mean(1.0 / (1.0 + (distances / d0) ** 2))
    
    return tm_score


def kabsch_align(coords_mobile, coords_target):
    """
    Kabsch algorithm for optimal rigid-body alignment.
    
    Returns:
        aligned_coords: [L, 3] optimally aligned coordinates
    """
    # Center both coordinate sets
    mobile_centered = coords_mobile - coords_mobile.mean(axis=0)
    target_centered = coords_target - coords_target.mean(axis=0)
    
    # Compute covariance matrix
    H = mobile_centered.T @ target_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Optimal rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1, not -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation and translation
    aligned = mobile_centered @ R.T + coords_target.mean(axis=0)
    
    return aligned


# ============================================================================
# lDDT (Local Distance Difference Test)
# ============================================================================

def compute_lddt(coords_pred, coords_true, mask=None, cutoff=15.0):
    """
    Compute lDDT (local Distance Difference Test).
    
    Used by AlphaFold2 for quality assessment. Better than RMSD because:
    - Per-residue scores (identifies problem regions)
    - Local metric (robust to domain movements)
    - Standardized threshold (0.5, 1.0, 2.0, 4.0 Å)
    
    Args:
        coords_pred: [L, 3] predicted coordinates
        coords_true: [L, 3] true coordinates
        mask: [L] boolean mask for valid residues
        cutoff: float, distance threshold (default 15Å)
    
    Returns:
        lddt_global: float in [0, 1] (higher is better)
        lddt_per_residue: [L] per-residue scores
    """
    if mask is None:
        mask = np.ones(len(coords_true), dtype=bool)
    
    L = len(coords_true)
    lddt_scores = np.zeros(L)
    
    # Compute distance matrices
    dist_true = cdist(coords_true, coords_true)
    dist_pred = cdist(coords_pred, coords_pred)
    
    # For each residue
    for i in range(L):
        if not mask[i]:
            continue
        
        # Find neighbors within cutoff
        neighbors = (dist_true[i] < cutoff) & (dist_true[i] > 0) & mask
        n_neighbors = neighbors.sum()
        
        if n_neighbors == 0:
            continue
        
        # Compute distance differences
        dist_diff = np.abs(dist_true[i, neighbors] - dist_pred[i, neighbors])
        
        # Count preserved distances at different thresholds
        preserved_0_5 = (dist_diff < 0.5).sum()
        preserved_1_0 = (dist_diff < 1.0).sum()
        preserved_2_0 = (dist_diff < 2.0).sum()
        preserved_4_0 = (dist_diff < 4.0).sum()
        
        # lDDT = average of preserved fractions
        lddt_scores[i] = (preserved_0_5 + preserved_1_0 + 
                          preserved_2_0 + preserved_4_0) / (4 * n_neighbors)
    
    # Global lDDT (average over valid residues)
    lddt_global = lddt_scores[mask].mean() if mask.sum() > 0 else 0.0
    
    return lddt_global, lddt_scores


# ============================================================================
# GDT (Global Distance Test)
# ============================================================================

def compute_gdt(coords_pred, coords_true, mask=None):
    """
    Compute GDT-TS and GDT-HA scores (CASP evaluation metrics).
    
    GDT-TS (Total Score): % residues under 1, 2, 4, 8Å
    GDT-HA (High Accuracy): % residues under 0.5, 1, 2, 4Å
    
    Args:
        coords_pred: [L, 3] predicted coordinates
        coords_true: [L, 3] true coordinates
        mask: [L] boolean mask
    
    Returns:
        gdt_ts: float in [0, 100]
        gdt_ha: float in [0, 100]
    """
    if mask is None:
        mask = np.ones(len(coords_true), dtype=bool)
    
    # Align structures
    coords_pred_aligned = kabsch_align(coords_pred, coords_true)
    
    # Compute per-residue distances
    distances = np.linalg.norm(coords_pred_aligned - coords_true, axis=1)
    distances = distances[mask]
    
    if len(distances) == 0:
        return 0.0, 0.0
    
    # GDT-TS thresholds: 1, 2, 4, 8Å
    p1 = (distances < 1.0).mean() * 100
    p2 = (distances < 2.0).mean() * 100
    p4 = (distances < 4.0).mean() * 100
    p8 = (distances < 8.0).mean() * 100
    gdt_ts = (p1 + p2 + p4 + p8) / 4
    
    # GDT-HA thresholds: 0.5, 1, 2, 4Å (higher accuracy)
    p0_5 = (distances < 0.5).mean() * 100
    p1_ha = (distances < 1.0).mean() * 100
    p2_ha = (distances < 2.0).mean() * 100
    p4_ha = (distances < 4.0).mean() * 100
    gdt_ha = (p0_5 + p1_ha + p2_ha + p4_ha) / 4
    
    return gdt_ts, gdt_ha


# ============================================================================
# RMSF (Ensemble Flexibility)
# ============================================================================

def compute_rmsf(ensemble_coords, mask=None):
    """
    Compute RMSF (Root Mean Square Fluctuation) per residue.
    
    Measures flexibility:
        High RMSF: Flexible region (loops, termini)
        Low RMSF: Rigid region (secondary structure core)
    
    Args:
        ensemble_coords: [N_models, L, 3] ensemble CA coordinates
        mask: [L] valid residues
    
    Returns:
        rmsf: [L] per-residue fluctuation in Å
    """
    if mask is None:
        mask = np.ones(ensemble_coords.shape[1], dtype=bool)
    
    N, L, _ = ensemble_coords.shape
    
    if N == 1:
        return np.zeros(L)
    
    # Align all structures to the first one
    aligned_ensemble = np.zeros_like(ensemble_coords)
    for i in range(N):
        aligned_ensemble[i] = kabsch_align(ensemble_coords[i], ensemble_coords[0])
    
    # Compute mean structure
    mean_structure = aligned_ensemble.mean(axis=0)
    
    # RMSF = RMS deviation from mean
    deviations = aligned_ensemble - mean_structure
    rmsf = np.sqrt((deviations ** 2).sum(axis=-1).mean(axis=0))
    
    return rmsf


# ============================================================================
# PHYSICAL PLAUSIBILITY METRICS
# ============================================================================

def compute_radius_of_gyration(coords, mask=None):
    """
    Radius of gyration (Rg) - measure of compactness.
    
    Empirical formula for globular proteins:
        Rg ≈ 2.2 × L^0.38 Å
    
    Args:
        coords: [L, 3] CA coordinates
        mask: [L] valid residues
    
    Returns:
        rg: float (Angstroms)
    """
    if mask is not None:
        coords = coords[mask]
    
    if len(coords) == 0:
        return 0.0
    
    center = coords.mean(axis=0)
    rg = np.sqrt(((coords - center) ** 2).sum() / len(coords))
    
    return rg


def expected_rg(length):
    """Expected Rg for globular protein of given length"""
    return 2.2 * (length ** 0.38)


def compute_contact_map(coords, cutoff=8.0):
    """
    Compute residue-residue contact map.
    
    Args:
        coords: [L, 3] CA coordinates
        cutoff: float, distance threshold (default 8Å)
    
    Returns:
        contact_map: [L, L] boolean array
    """
    dist_matrix = cdist(coords, coords)
    
    # Exclude self and nearest neighbors
    for i in range(len(coords)):
        dist_matrix[i, max(0, i-1):min(len(coords), i+2)] = np.inf
    
    contact_map = dist_matrix < cutoff
    return contact_map


def contact_map_overlap(contact_pred, contact_true):
    """
    Compute precision, recall, F1 for contact map prediction.
    
    Returns:
        precision, recall, f1_score
    """
    # Ignore diagonal and immediate neighbors
    mask = ~np.eye(len(contact_true), dtype=bool)
    
    pred_contacts = contact_pred[mask]
    true_contacts = contact_true[mask]
    
    tp = (pred_contacts & true_contacts).sum()
    fp = (pred_contacts & ~true_contacts).sum()
    fn = (~pred_contacts & true_contacts).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_ensemble_diversity(ensemble_coords):
    """
    Compute pairwise RMSD between all ensemble members.
    
    Returns:
        mean_pairwise_rmsd: float
        rmsd_matrix: [N, N] pairwise RMSDs
    """
    N = len(ensemble_coords)
    rmsd_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i+1, N):
            coords_i = ensemble_coords[i]
            coords_j = ensemble_coords[j]
            
            # Align j to i
            coords_j_aligned = kabsch_align(coords_j, coords_i)
            
            # RMSD
            rmsd = np.sqrt(((coords_i - coords_j_aligned) ** 2).mean())
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd
    
    mean_pairwise_rmsd = rmsd_matrix[np.triu_indices(N, k=1)].mean() if N > 1 else 0.0
    return mean_pairwise_rmsd, rmsd_matrix


# ============================================================================
# PDB PARSING
# ============================================================================

def load_ca_coords_from_pdb(pdb_file, model_num=None):
    """
    Load CA coordinates from PDB file.
    
    Args:
        pdb_file: path to PDB file
        model_num: int, specific model number (for NMR/ensemble PDFs)
                   If None, loads first model
    
    Returns:
        coords: [L, 3] numpy array of CA coordinates
    """
    coords = []
    current_model = 0
    in_model = (model_num is None)
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                current_model = int(line.split()[1])
                in_model = (model_num is None or current_model == model_num)
            
            if line.startswith('ENDMDL'):
                if model_num is not None and current_model == model_num:
                    break
                in_model = False
            
            if line.startswith('ATOM') and in_model:
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    
    return np.array(coords)


def load_ensemble_from_pdb(pdb_file):
    """
    Load all models from a multi-model PDB file (e.g., NMR ensemble).
    
    Returns:
        ensemble_coords: [N_models, L, 3] numpy array
    """
    ensemble = []
    current_model = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                current_model = []
            
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    current_model.append([x, y, z])
            
            if line.startswith('ENDMDL'):
                if current_model:
                    ensemble.append(np.array(current_model))
                current_model = []
    
    # Handle single-model PDB (no MODEL/ENDMDL records)
    if len(ensemble) == 0 and len(current_model) > 0:
        ensemble.append(np.array(current_model))
    
    return np.array(ensemble)


# ============================================================================
# VALIDATION REPORT GENERATION
# ============================================================================

def validate_single_structure(pred_pdb, true_pdb):
    """
    Validate a single predicted structure against ground truth.
    
    Returns:
        results: dict with all metrics
    """
    print("Loading structures...")
    coords_pred = load_ca_coords_from_pdb(pred_pdb)
    coords_true = load_ca_coords_from_pdb(true_pdb)
    
    L = len(coords_true)
    print(f"  Protein length: {L} residues")
    
    results = {}
    
    # 1. TM-score
    print("\n[1/6] Computing TM-score...")
    tm_score = compute_tm_score_python(coords_pred, coords_true)
    results['tm_score'] = tm_score
    print(f"  TM-score: {tm_score:.3f}")
    if tm_score > 0.9:
        print(f"  → Excellent model (>0.9) ✓✓")
    elif tm_score > 0.7:
        print(f"  → Good model (>0.7) ✓")
    elif tm_score > 0.5:
        print(f"  → Same fold (>0.5) ✓")
    else:
        print(f"  → Different fold (<0.5) ✗")
    
    # 2. lDDT
    print("\n[2/6] Computing lDDT...")
    lddt_global, lddt_per_res = compute_lddt(coords_pred, coords_true)
    results['lddt'] = lddt_global
    results['lddt_per_residue'] = lddt_per_res
    print(f"  lDDT: {lddt_global:.3f}")
    if lddt_global > 0.9:
        print(f"  → Excellent quality (>0.9) ✓✓")
    elif lddt_global > 0.7:
        print(f"  → Good quality (>0.7) ✓")
    else:
        print(f"  → Needs improvement (<0.7) ⚠️")
    
    # 3. GDT
    print("\n[3/6] Computing GDT-TS/HA...")
    gdt_ts, gdt_ha = compute_gdt(coords_pred, coords_true)
    results['gdt_ts'] = gdt_ts
    results['gdt_ha'] = gdt_ha
    print(f"  GDT-TS: {gdt_ts:.1f}")
    print(f"  GDT-HA: {gdt_ha:.1f}")
    
    # 4. Radius of gyration
    print("\n[4/6] Computing radius of gyration...")
    rg_pred = compute_radius_of_gyration(coords_pred)
    rg_true = compute_radius_of_gyration(coords_true)
    rg_expected = expected_rg(L)
    results['rg_pred'] = rg_pred
    results['rg_true'] = rg_true
    results['rg_expected'] = rg_expected
    print(f"  Predicted: {rg_pred:.2f}Å")
    print(f"  True: {rg_true:.2f}Å")
    print(f"  Expected (globular): {rg_expected:.2f}Å")
    if abs(rg_pred - rg_true) / rg_true < 0.15:
        print(f"  → Compactness preserved ✓")
    else:
        print(f"  → Significant deviation ⚠️")
    
    # 5. Contact map
    print("\n[5/6] Computing contact map overlap...")
    contacts_pred = compute_contact_map(coords_pred)
    contacts_true = compute_contact_map(coords_true)
    prec, rec, f1 = contact_map_overlap(contacts_pred, contacts_true)
    results['contact_precision'] = prec
    results['contact_recall'] = rec
    results['contact_f1'] = f1
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall: {rec:.3f}")
    print(f"  F1-score: {f1:.3f}")
    if f1 > 0.8:
        print(f"  → Excellent contact preservation (>0.8) ✓✓")
    elif f1 > 0.6:
        print(f"  → Good contact preservation (>0.6) ✓")
    else:
        print(f"  → Poor contact preservation (<0.6) ⚠️")
    
    # 6. RMSD (for comparison)
    print("\n[6/6] Computing RMSD...")
    coords_pred_aligned = kabsch_align(coords_pred, coords_true)
    rmsd = np.sqrt(((coords_pred_aligned - coords_true) ** 2).mean())
    results['rmsd'] = rmsd
    print(f"  RMSD: {rmsd:.3f}Å")
    
    return results


def validate_ensemble(ensemble_pdb, ground_truth_pdb=None):
    """
    Validate a generated ensemble (multi-model PDB).
    
    Returns:
        results: dict with ensemble-specific metrics
    """
    print("Loading ensemble...")
    ensemble_coords = load_ensemble_from_pdb(ensemble_pdb)
    N_models, L, _ = ensemble_coords.shape
    print(f"  Ensemble size: {N_models} models")
    print(f"  Protein length: {L} residues")
    
    results = {}
    
    # 1. RMSF
    print("\n[1/4] Computing RMSF (per-residue flexibility)...")
    rmsf = compute_rmsf(ensemble_coords)
    results['rmsf'] = rmsf
    results['rmsf_mean'] = rmsf.mean()
    results['rmsf_max'] = rmsf.max()
    print(f"  Average RMSF: {rmsf.mean():.3f}Å")
    print(f"  Max RMSF: {rmsf.max():.3f}Å")
    print(f"  Range: {rmsf.min():.3f} - {rmsf.max():.3f}Å")
    
    # 2. Pairwise diversity
    print("\n[2/4] Computing ensemble diversity...")
    mean_div, rmsd_matrix = compute_ensemble_diversity(ensemble_coords)
    results['ensemble_diversity'] = mean_div
    results['pairwise_rmsd_matrix'] = rmsd_matrix
    print(f"  Mean pairwise RMSD: {mean_div:.3f}Å")
    if mean_div > 0.05:
        print(f"  → Sufficient diversity (>0.05Å) ✓")
    else:
        print(f"  → Low diversity (<0.05Å) - may be overly similar")
    
    # 3. Compactness consistency
    print("\n[3/4] Checking compactness consistency...")
    rg_values = [compute_radius_of_gyration(coords) for coords in ensemble_coords]
    rg_mean = np.mean(rg_values)
    rg_std = np.std(rg_values)
    results['rg_mean'] = rg_mean
    results['rg_std'] = rg_std
    print(f"  Mean Rg: {rg_mean:.2f}Å ± {rg_std:.2f}Å")
    print(f"  Expected (globular): {expected_rg(L):.2f}Å")
    
    # 4. Validation vs ground truth (if provided)
    if ground_truth_pdb:
        print("\n[4/4] Validating against ground truth...")
        coords_true = load_ca_coords_from_pdb(ground_truth_pdb)
        
        tm_scores = []
        lddt_scores = []
        for i, coords in enumerate(ensemble_coords):
            tm = compute_tm_score_python(coords, coords_true)
            lddt, _ = compute_lddt(coords, coords_true)
            tm_scores.append(tm)
            lddt_scores.append(lddt)
        
        results['tm_score_mean'] = np.mean(tm_scores)
        results['tm_score_std'] = np.std(tm_scores)
        results['lddt_mean'] = np.mean(lddt_scores)
        results['lddt_std'] = np.std(lddt_scores)
        
        print(f"  TM-score: {np.mean(tm_scores):.3f} ± {np.std(tm_scores):.3f}")
        print(f"  lDDT: {np.mean(lddt_scores):.3f} ± {np.std(lddt_scores):.3f}")
    
    return results


def write_validation_report(results, output_file):
    """Write comprehensive validation report to file"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PROTEIN ENSEMBLE VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TOPOLOGY METRICS:\n")
        f.write("-" * 80 + "\n")
        if 'tm_score' in results:
            f.write(f"TM-score: {results['tm_score']:.3f}\n")
            f.write(f"  Interpretation: ")
            if results['tm_score'] > 0.9:
                f.write("Excellent model (>0.9) ✓✓\n")
            elif results['tm_score'] > 0.7:
                f.write("Good model (>0.7) ✓\n")
            elif results['tm_score'] > 0.5:
                f.write("Same fold (>0.5) ✓\n")
            else:
                f.write("Different fold (<0.5) ✗\n")
        
        f.write("\nLOCAL QUALITY:\n")
        f.write("-" * 80 + "\n")
        if 'lddt' in results:
            f.write(f"lDDT: {results['lddt']:.3f}\n")
            if results['lddt'] > 0.9:
                f.write(f"  Interpretation: Excellent quality (>0.9) ✓✓\n")
            elif results['lddt'] > 0.7:
                f.write(f"  Interpretation: Good quality (>0.7) ✓\n")
            else:
                f.write(f"  Interpretation: Needs improvement (<0.7) ⚠️\n")
        
        if 'gdt_ts' in results:
            f.write(f"\nGDT-TS: {results['gdt_ts']:.1f}\n")
            f.write(f"GDT-HA: {results['gdt_ha']:.1f}\n")
        
        f.write("\nPHYSICAL PLAUSIBILITY:\n")
        f.write("-" * 80 + "\n")
        if 'rg_pred' in results:
            f.write(f"Radius of gyration:\n")
            f.write(f"  Predicted: {results['rg_pred']:.2f}Å\n")
            f.write(f"  True: {results['rg_true']:.2f}Å\n")
            f.write(f"  Expected (globular): {results['rg_expected']:.2f}Å\n")
        
        if 'contact_f1' in results:
            f.write(f"\nContact map preservation:\n")
            f.write(f"  Precision: {results['contact_precision']:.3f}\n")
            f.write(f"  Recall: {results['contact_recall']:.3f}\n")
            f.write(f"  F1-score: {results['contact_f1']:.3f}\n")
        
        if 'rmsf_mean' in results:
            f.write("\nENSEMBLE METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average RMSF: {results['rmsf_mean']:.3f}Å\n")
            f.write(f"Max RMSF: {results['rmsf_max']:.3f}Å\n")
            f.write(f"Ensemble diversity: {results.get('ensemble_diversity', 'N/A'):.3f}Å\n")
        
        f.write("\n" + "=" * 80 + "\n")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive validation metrics for protein structure prediction'
    )
    parser.add_argument('--pred', type=str, help='Predicted structure PDB file')
    parser.add_argument('--true', type=str, help='Ground truth PDB file')
    parser.add_argument('--ensemble', type=str, help='Ensemble PDB file (multi-model)')
    parser.add_argument('--output', type=str, default='validation_report.txt',
                       help='Output report file')
    
    args = parser.parse_args()
    
    if args.pred and args.true:
        # Single structure validation
        print("\n" + "=" * 80)
        print("SINGLE STRUCTURE VALIDATION")
        print("=" * 80)
        results = validate_single_structure(args.pred, args.true)
        write_validation_report(results, args.output)
        print(f"\n✓ Report written to: {args.output}")
    
    elif args.ensemble:
        # Ensemble validation
        print("\n" + "=" * 80)
        print("ENSEMBLE VALIDATION")
        print("=" * 80)
        results = validate_ensemble(args.ensemble, args.true)
        write_validation_report(results, args.output)
        print(f"\n✓ Report written to: {args.output}")
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # Validate single structure:")
        print("  python validation_metrics.py --pred model.pdb --true native.pdb")
        print("\n  # Validate ensemble:")
        print("  python validation_metrics.py --ensemble ensemble.pdb --true native.pdb")


if __name__ == '__main__':
    main()

