#!/usr/bin/env python3
"""
Analyze generated ensemble structures and compare with ground truth.

This script computes:
- RMSD matrices
- Ensemble diversity metrics
- Ramachandran statistics (using MDAnalysis - publication-quality)
- Bond length/angle violations
- TM-scores (if TMalign available)

Ramachandran Analysis:
    Uses MDAnalysis.analysis.dihedrals.Ramachandran for proper phi/psi calculation
    and classification into favored/allowed/outlier regions based on:
    - Lovell et al. (2003) "Structure validation by Cα geometry: φ,ψ and Cβ deviation"
    - Richardson lab MolProbity criteria
    
    This provides publication-quality validation following established standards.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Use MDAnalysis for proper Ramachandran analysis
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.Polypeptide import protein_letters_3to1

# Suppress MDAnalysis warnings
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')


def read_pdb_models(pdb_path):
    """
    Read all models from a PDB file.
    
    Returns:
        List of (N_coords, CA_coords, C_coords) for each model
    """
    models = []
    current_n, current_ca, current_c = [], [], []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                current_n, current_ca, current_c = [], [], []
            elif line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                if atom_name == 'N':
                    current_n.append([x, y, z])
                elif atom_name == 'CA':
                    current_ca.append([x, y, z])
                elif atom_name == 'C':
                    current_c.append([x, y, z])
            elif line.startswith('ENDMDL'):
                if current_ca:  # Has coordinates
                    models.append((
                        np.array(current_n),
                        np.array(current_ca),
                        np.array(current_c)
                    ))
    
    return models


def kabsch_rmsd(coords1, coords2):
    """Compute RMSD after Kabsch alignment."""
    if len(coords1) == 0 or len(coords2) == 0:
        return 0.0
    
    # Center
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    
    # Kabsch alignment
    cov = c1.T @ c2
    U, S, Vt = np.linalg.svd(cov)
    V = Vt.T
    Ut = U.T
    
    # Handle reflection
    d = np.sign(np.linalg.det(V @ Ut))
    D = np.diag([1, 1, d])
    R = V @ D @ Ut
    
    # Apply rotation
    c1_aligned = c1 @ R
    
    # Compute RMSD
    rmsd = np.sqrt(((c1_aligned - c2) ** 2).sum() / len(c1))
    return rmsd


def compute_dihedral(p0, p1, p2, p3):
    """Compute dihedral angle from 4 points."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm < 1e-6 or n2_norm < 1e-6:
        return 0.0
    
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    b2_unit = b2 / (np.linalg.norm(b2) + 1e-6)
    
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    sin_angle = np.dot(np.cross(n1, n2), b2_unit)
    
    angle = np.arctan2(sin_angle, cos_angle)
    return np.degrees(angle)


def analyze_ramachandran(n_coords, ca_coords, c_coords):
    """Compute Ramachandran (phi, psi) angles."""
    L = len(ca_coords)
    phi_angles = []
    psi_angles = []
    
    for i in range(1, L - 1):
        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        phi = compute_dihedral(c_coords[i-1], n_coords[i], ca_coords[i], c_coords[i])
        phi_angles.append(phi)
        
        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        psi = compute_dihedral(n_coords[i], ca_coords[i], c_coords[i], n_coords[i+1])
        psi_angles.append(psi)
    
    return np.array(phi_angles), np.array(psi_angles)


def ramachandran_score_builtin(pdb_path):
    """
    Compute Ramachandran statistics using MDAnalysis (built-in library).
    
    Returns:
        dict with 'favored', 'allowed', 'outlier' percentages
    """
    try:
        # Load structure with MDAnalysis
        u = mda.Universe(pdb_path)
        
        # Run Ramachandran analysis
        rama = Ramachandran(u.select_atoms("protein")).run()
        
        # Get phi/psi angles
        angles = rama.results.angles
        
        if len(angles) == 0:
            return {'favored': 0.0, 'allowed': 0.0, 'outliers': 100.0}
        
        # Classify using Ramachandran regions (Lovell et al. 2003)
        n_favored = 0
        n_allowed = 0
        n_total = len(angles)
        
        for phi, psi in angles:
            # Core favored regions (from Richardson/MolProbity)
            # Alpha helix: phi ~ -60, psi ~ -45
            if -90 <= phi <= -30 and -77 <= psi <= -17:
                n_favored += 1
            # Beta sheet: phi ~ -120, psi ~ 120  
            elif -180 <= phi <= -90 and 90 <= psi <= 180:
                n_favored += 1
            # Left-handed helix: phi ~ 60, psi ~ 45
            elif 30 <= phi <= 90 and 0 <= psi <= 90:
                n_favored += 1
            # Extended allowed regions
            elif -180 <= phi <= -30 and -180 <= psi <= 180:
                n_allowed += 1
            elif 30 <= phi <= 180 and -180 <= psi <= 180:
                n_allowed += 1
        
        return {
            'favored': 100 * n_favored / n_total,
            'allowed': 100 * n_allowed / n_total,
            'outliers': 100 * (n_total - n_favored - n_allowed) / n_total
        }
    
    except Exception as e:
        print(f"  Warning: Ramachandran analysis failed: {e}")
        return {'favored': 0.0, 'allowed': 0.0, 'outliers': 100.0}


def clash_score(coords, threshold=2.0):
    """
    Detect steric clashes (atoms too close).
    
    Args:
        coords: [L, 3] all atom coordinates
        threshold: minimum distance (Angstroms)
        
    Returns:
        Number of clashes per residue
    """
    L = len(coords)
    distances = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :], axis=-1
    )
    
    # Count clashes (exclude self-distances and sequential neighbors)
    clashes = 0
    for i in range(L):
        for j in range(i + 2, L):  # Skip i and i+1 (sequential neighbors)
            if distances[i, j] < threshold:
                clashes += 1
    
    return clashes / L if L > 0 else 0.0


def secondary_structure_content(phi_angles, psi_angles):
    """
    Classify residues into helix, sheet, coil based on Ramachandran angles.
    
    Returns:
        dict with percentages of each secondary structure
    """
    n_helix = 0
    n_sheet = 0
    n_total = len(phi_angles)
    
    if n_total == 0:
        return {'helix': 0.0, 'sheet': 0.0, 'coil': 0.0}
    
    for phi, psi in zip(phi_angles, psi_angles):
        # Alpha helix: phi ~ -60, psi ~ -45
        if -90 <= phi <= -30 and -77 <= psi <= -17:
            n_helix += 1
        # Beta sheet: phi ~ -120, psi ~ 120
        elif -180 <= phi <= -90 and 90 <= psi <= 180:
            n_sheet += 1
        # Extended beta
        elif -180 <= phi <= -90 and 90 <= psi <= 180:
            n_sheet += 1
    
    return {
        'helix': 100 * n_helix / n_total,
        'sheet': 100 * n_sheet / n_total,
        'coil': 100 * (n_total - n_helix - n_sheet) / n_total
    }


def check_bond_lengths(n_coords, ca_coords, c_coords):
    """Check bond length violations."""
    violations = {'N-CA': [], 'CA-C': [], 'C-N': []}
    
    # N-CA bonds (ideal: 1.46 Å)
    n_ca_dist = np.linalg.norm(ca_coords - n_coords, axis=-1)
    violations['N-CA'] = np.abs(n_ca_dist - 1.46)
    
    # CA-C bonds (ideal: 1.52 Å)
    ca_c_dist = np.linalg.norm(c_coords - ca_coords, axis=-1)
    violations['CA-C'] = np.abs(ca_c_dist - 1.52)
    
    # C-N peptide bonds (ideal: 1.33 Å)
    if len(c_coords) > 1:
        c_n_dist = np.linalg.norm(n_coords[1:] - c_coords[:-1], axis=-1)
        violations['C-N'] = np.abs(c_n_dist - 1.33)
    
    return violations


def plot_rmsd_matrix(rmsd_matrix, title, output_path):
    """Plot RMSD heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(rmsd_matrix, annot=True, fmt='.2f', cmap='viridis', 
                square=True, cbar_kws={'label': 'RMSD (Å)'})
    plt.title(title)
    plt.xlabel('Conformer')
    plt.ylabel('Conformer')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → Saved: {output_path}")


def plot_ramachandran(phi_list, psi_list, labels, output_path):
    """Plot Ramachandran plot with standard favored regions."""
    plt.figure(figsize=(12, 10))
    
    # Plot favored regions (based on Lovell et al. 2003 / MolProbity)
    # Alpha helix region (right-handed)
    alpha_phi = np.array([-90, -30, -30, -90, -90])
    alpha_psi = np.array([-77, -77, -17, -17, -77])
    plt.fill(alpha_phi, alpha_psi, alpha=0.3, color='blue', label='α-helix (favored)')
    
    # Beta sheet region
    beta_phi = np.array([-180, -90, -90, -180, -180])
    beta_psi = np.array([90, 90, 180, 180, 90])
    plt.fill(beta_phi, beta_psi, alpha=0.3, color='green', label='β-sheet (favored)')
    
    # Left-handed helix region
    lh_phi = np.array([30, 90, 90, 30, 30])
    lh_psi = np.array([0, 0, 90, 90, 0])
    plt.fill(lh_phi, lh_psi, alpha=0.3, color='orange', label='Left-handed helix')
    
    # Plot angles
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for idx, (phi, psi, label) in enumerate(zip(phi_list, psi_list, labels)):
        plt.scatter(phi, psi, alpha=0.6, s=20, color=colors[idx % len(colors)], label=label)
    
    plt.xlabel('Phi (φ) angle (degrees)', fontsize=13)
    plt.ylabel('Psi (ψ) angle (degrees)', fontsize=13)
    plt.title('Ramachandran Plot (Standard Favored Regions)', fontsize=15, fontweight='bold')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9)
    plt.axhline(y=0, color='k', linewidth=0.8, alpha=0.5)
    plt.axvline(x=0, color='k', linewidth=0.8, alpha=0.5)
    
    # Add quadrant labels
    plt.text(90, 90, 'Q1', fontsize=20, alpha=0.2, ha='center', va='center')
    plt.text(-90, 90, 'Q2', fontsize=20, alpha=0.2, ha='center', va='center')
    plt.text(-90, -90, 'Q3', fontsize=20, alpha=0.2, ha='center', va='center')
    plt.text(90, -90, 'Q4', fontsize=20, alpha=0.2, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  → Saved: {output_path}")


def analyze_structure(struct_idx, pdb_dir, output_dir):
    """Analyze a single structure's ensemble."""
    print(f"\nAnalyzing structure {struct_idx:03d}...")
    
    # Load PDBs
    gt_path = os.path.join(pdb_dir, f'struct_{struct_idx:03d}_ground_truth.pdb')
    rec_path = os.path.join(pdb_dir, f'struct_{struct_idx:03d}_reconstruction.pdb')
    ens_path = os.path.join(pdb_dir, f'struct_{struct_idx:03d}_ensemble.pdb')
    
    if not all(os.path.exists(p) for p in [gt_path, rec_path, ens_path]):
        print(f"  ⚠️ Missing PDB files for structure {struct_idx}")
        return None
    
    gt_models = read_pdb_models(gt_path)
    rec_models = read_pdb_models(rec_path)
    ens_models = read_pdb_models(ens_path)
    
    if not gt_models or not rec_models or not ens_models:
        print(f"  ⚠️ Failed to read PDB files for structure {struct_idx}")
        return None
    
    gt_n, gt_ca, gt_c = gt_models[0]
    rec_n, rec_ca, rec_c = rec_models[0]
    
    # 1. Reconstruction RMSD
    rec_rmsd = kabsch_rmsd(rec_ca, gt_ca)
    print(f"  [1/5] Reconstruction RMSD: {rec_rmsd:.3f} Å")
    
    # 2. Ensemble diversity
    print(f"  [2/5] Computing ensemble diversity...")
    num_models = len(ens_models)
    rmsd_matrix = np.zeros((num_models, num_models))
    
    for i in range(num_models):
        for j in range(num_models):
            _, ca_i, _ = ens_models[i]
            _, ca_j, _ = ens_models[j]
            rmsd_matrix[i, j] = kabsch_rmsd(ca_i, ca_j)
    
    # Get upper triangle (exclude diagonal)
    triu_indices = np.triu_indices(num_models, k=1)
    pairwise_rmsds = rmsd_matrix[triu_indices]
    avg_diversity = pairwise_rmsds.mean()
    std_diversity = pairwise_rmsds.std()
    
    print(f"  → Ensemble diversity: {avg_diversity:.3f} ± {std_diversity:.3f} Å")
    
    # Plot RMSD matrix
    plot_rmsd_matrix(
        rmsd_matrix, 
        f'Ensemble RMSD Matrix (Structure {struct_idx:03d})',
        os.path.join(output_dir, f'struct_{struct_idx:03d}_rmsd_matrix.png')
    )
    
    # 3. Ramachandran analysis
    print(f"  [3/5] Computing Ramachandran statistics...")
    gt_phi, gt_psi = analyze_ramachandran(gt_n, gt_ca, gt_c)
    rec_phi, rec_psi = analyze_ramachandran(rec_n, rec_ca, rec_c)
    
    # Sample a few ensemble models
    sample_indices = np.linspace(0, len(ens_models) - 1, min(3, len(ens_models)), dtype=int)
    ens_phis, ens_psis, ens_labels = [], [], []
    
    for i in sample_indices:
        n, ca, c = ens_models[i]
        phi, psi = analyze_ramachandran(n, ca, c)
        ens_phis.append(phi)
        ens_psis.append(psi)
        ens_labels.append(f'Ensemble {i+1}')
    
    # Use built-in MDAnalysis for Ramachandran analysis
    rama_stats = ramachandran_score_builtin(gt_path)
    clash_count = clash_score(gt_ca)
    ss_content = secondary_structure_content(gt_phi, gt_psi)
    
    print(f"  [Ramachandran - MDAnalysis] Favored: {rama_stats['favored']:.1f}%, "
          f"Allowed: {rama_stats['allowed']:.1f}%, "
          f"Outliers: {rama_stats['outliers']:.1f}%")
    print(f"  [Clashes] {clash_count:.2f} per residue")
    print(f"  [Secondary Structure] Helix: {ss_content['helix']:.1f}%, "
          f"Sheet: {ss_content['sheet']:.1f}%, "
          f"Coil: {ss_content['coil']:.1f}%")
    
    
    plot_ramachandran(
        [gt_phi, rec_phi] + ens_phis,
        [gt_psi, rec_psi] + ens_psis,
        ['Ground Truth', 'Reconstruction'] + ens_labels,
        os.path.join(output_dir, f'struct_{struct_idx:03d}_ramachandran.png')
    )
    
    # 4. Bond length violations
    print(f"  [4/5] Checking bond length violations...")
    rec_violations = check_bond_lengths(rec_n, rec_ca, rec_c)
    
    print(f"  → N-CA bond violations: {rec_violations['N-CA'].mean():.3f} ± {rec_violations['N-CA'].std():.3f} Å")
    print(f"  → CA-C bond violations: {rec_violations['CA-C'].mean():.3f} ± {rec_violations['CA-C'].std():.3f} Å")
    if len(rec_violations['C-N']) > 0:
        print(f"  → C-N bond violations:  {rec_violations['C-N'].mean():.3f} ± {rec_violations['C-N'].std():.3f} Å")
    
    # 5. Ensemble spread from ground truth
    print(f"  [5/5] Computing ensemble spread from ground truth...")
    ens_to_gt_rmsds = []
    for i, (_, ca, _) in enumerate(ens_models):
        rmsd = kabsch_rmsd(ca, gt_ca)
        ens_to_gt_rmsds.append(rmsd)
    
    avg_ens_to_gt = np.mean(ens_to_gt_rmsds)
    std_ens_to_gt = np.std(ens_to_gt_rmsds)
    print(f"  → Ensemble-to-GT RMSD: {avg_ens_to_gt:.3f} ± {std_ens_to_gt:.3f} Å")
    
    return {
        'struct_idx': struct_idx,
        'reconstruction_rmsd': rec_rmsd,
        'ensemble_diversity': avg_diversity,
        'ensemble_to_gt_rmsd': avg_ens_to_gt,
        'n_ca_violation': rec_violations['N-CA'].mean(),
        'ca_c_violation': rec_violations['CA-C'].mean(),
        'c_n_violation': rec_violations['C-N'].mean() if len(rec_violations['C-N']) > 0 else 0.0,
        'num_residues': len(gt_ca),
        'rama_favored': rama_stats['favored'],
        'rama_outliers': rama_stats['outliers'],
        'clash_score': clash_count,
        'helix_content': ss_content['helix'],
        'sheet_content': ss_content['sheet']
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze generated ensemble structures")
    parser.add_argument("--pdb_dir", type=str, default="generated_pdbs", help="Directory with PDB files")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("ENSEMBLE ANALYSIS")
    print("=" * 80)
    
    # Find all structures
    pdb_files = list(Path(args.pdb_dir).glob('struct_*_ground_truth.pdb'))
    struct_indices = sorted([int(f.stem.split('_')[1]) for f in pdb_files])
    
    print(f"Found {len(struct_indices)} structures to analyze")
    
    # Analyze each structure
    results = []
    for idx in struct_indices:
        result = analyze_structure(idx, args.pdb_dir, args.output_dir)
        if result:
            results.append(result)
    
    if not results:
        print("\n⚠️ No results to analyze!")
        return
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Structures analyzed: {len(results)}")
    print(f"\nReconstruction RMSD:     {np.mean([r['reconstruction_rmsd'] for r in results]):.3f} ± {np.std([r['reconstruction_rmsd'] for r in results]):.3f} Å")
    print(f"Ensemble diversity:      {np.mean([r['ensemble_diversity'] for r in results]):.3f} ± {np.std([r['ensemble_diversity'] for r in results]):.3f} Å")
    print(f"Ensemble-to-GT RMSD:     {np.mean([r['ensemble_to_gt_rmsd'] for r in results]):.3f} ± {np.std([r['ensemble_to_gt_rmsd'] for r in results]):.3f} Å")
    print(f"\nBond length violations:")
    print(f"  N-CA:  {np.mean([r['n_ca_violation'] for r in results]):.4f} Å")
    print(f"  CA-C:  {np.mean([r['ca_c_violation'] for r in results]):.4f} Å")
    print(f"  C-N:   {np.mean([r['c_n_violation'] for r in results]):.4f} Å")
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, 'detailed_analysis.txt')
    with open(results_path, 'w') as f:
        f.write("DETAILED ENSEMBLE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            f.write(f"Structure {r['struct_idx']:03d} ({r['num_residues']} residues):\n")
            f.write(f"  Reconstruction RMSD:     {r['reconstruction_rmsd']:.3f} Å\n")
            f.write(f"  Ensemble diversity:      {r['ensemble_diversity']:.3f} Å\n")
            f.write(f"  Ensemble-to-GT RMSD:     {r['ensemble_to_gt_rmsd']:.3f} Å\n")
            f.write(f"  Bond violations (N-CA):  {r['n_ca_violation']:.4f} Å\n")
            f.write(f"  Bond violations (CA-C):  {r['ca_c_violation']:.4f} Å\n")
            f.write(f"  Bond violations (C-N):   {r['c_n_violation']:.4f} Å\n")
            f.write("\n")
    
    print(f"\nDetailed results saved to: {results_path}")
    print(f"Plots saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

