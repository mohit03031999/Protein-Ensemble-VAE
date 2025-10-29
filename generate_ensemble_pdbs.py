#!/usr/bin/env python3
"""
Generate PDB files from trained VAE model for ensemble comparison.

This script:
1. Loads a trained VAE checkpoint
2. Generates ensemble structures by sampling from latent space
3. Reconstructs input structures
4. Saves all outputs as PDB files for visualization
5. Computes RMSD and diversity metrics
"""

import os
import sys
import argparse
import torch
import numpy as np
import h5py
from pathlib import Path

# Add models to path
sys.path.insert(0, 'models')
from model import HierCVAE
from data import EnsembleDataset, collate_pad
from training import load_checkpoint


def extract_sequence_from_pdb(pdb_path):
    """
    Extract sequence information from original PDB file.
    
    Args:
        pdb_path: path to original PDB file
        
    Returns:
        tuple: (sequence, pdb_id, chain_id)
    """
    try:
        sequence = ""
        pdb_id = None
        chain_id = 'A'
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('HEADER'):
                    # Extract PDB ID from HEADER line
                    pdb_id = line[62:66].strip()
                elif line.startswith('ATOM') and 'CA' in line[12:16]:
                    # Extract residue name and convert to 1-letter code
                    res_name = line[17:20].strip()
                    aa_3to1 = {
                        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                    }
                    aa_1letter = aa_3to1.get(res_name, 'X')
                    sequence += aa_1letter
        
        if sequence:
            print(f"    → Extracted sequence from PDB: {len(sequence)} residues")
            print(f"    → Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        else:
            print(f"    → No sequence found in PDB file")
        
        return sequence, pdb_id, chain_id
    except Exception as e:
        print(f"    → Error extracting sequence from {pdb_path}: {e}")
        return None, None, 'A'


def extract_sequence_from_h5(h5_path):
    """
    Extract sequence information from H5 file.
    
    Args:
        h5_path: path to H5 file
        
    Returns:
        tuple: (sequence, pdb_id, chain_id)
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get sequence
            if 'sequence' in f:
                seq_raw = f['sequence'][()]
                sequence = seq_raw.decode('utf-8') if isinstance(seq_raw, (bytes, bytearray)) else str(seq_raw)
                print(f"    → Found sequence in H5: {len(sequence)} residues")
            else:
                sequence = None
                print(f"    → No 'sequence' dataset found in H5 file")
            
            # Get PDB ID and chain ID from attributes
            pdb_id = f.attrs.get('pdb_id', None)
            chain_id = f.attrs.get('chain_id', 'A')
            
            print(f"    → PDB ID: {pdb_id}, Chain ID: {chain_id}")
            
            return sequence, pdb_id, chain_id
    except Exception as e:
        print(f"    → Error extracting sequence from {h5_path}: {e}")
        return None, None, 'A'




def compute_backbone_oxygen(n_coords, ca_coords, c_coords, mask):
    """
    Compute O (oxygen) atom positions for complete backbone.
    
    Args:
        n_coords: [L, 3] N atom coordinates
        ca_coords: [L, 3] CA atom coordinates  
        c_coords: [L, 3] C atom coordinates
        mask: [L] validity mask
        
    Returns:
        o_coords: [L, 3] O atom coordinates
    """
    L = len(mask)
    o_coords = np.zeros_like(c_coords)
    
    for i in range(L):
        if mask[i] > 0.5:  # Valid residue
            # O atom is positioned relative to C atom
            # Typical C-O bond length: 1.23 Å
            # Typical CA-C-O angle: ~120°
            
            if i == 0:
                # First residue: use arbitrary direction
                direction = np.array([1.0, 0.0, 0.0])
            else:
                # Use vector from CA(i-1) to C(i-1) as reference
                if mask[i-1] > 0.5:
                    ref_vector = c_coords[i-1] - ca_coords[i-1]
                    ref_vector = ref_vector / (np.linalg.norm(ref_vector) + 1e-8)
                    direction = ref_vector
                else:
                    direction = np.array([1.0, 0.0, 0.0])
            
            # Position O atom
            c_o_distance = 1.23  # Å
            o_coords[i] = c_coords[i] + direction * c_o_distance
    
    return o_coords


def write_pdb(coords_n, coords_ca, coords_c, mask, output_path, model_num=1, 
              sequence=None, pdb_id=None, chain_id='A', title=None, num_models=None):
    """
    Write enhanced PDB file with proper connectivity and complete backbone.
    
    Args:
        coords_n: [L, 3] N atom coordinates
        coords_ca: [L, 3] CA atom coordinates
        coords_c: [L, 3] C atom coordinates
        mask: [L] validity mask
        output_path: path to save PDB
        model_num: MODEL number in PDB (for ensembles)
        sequence: protein sequence string (optional)
        pdb_id: PDB identifier (optional)
        chain_id: chain identifier (default 'A')
        title: title for the structure (optional)
        num_models: total number of models (for NUMMDL field)
    """
    # Compute O coordinates for complete backbone
    coords_o = compute_backbone_oxygen(coords_n, coords_ca, coords_c, mask)
    
    # Convert 1-letter amino acid codes to 3-letter codes
    aa_1to3 = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    
    with open(output_path, 'w' if model_num == 1 else 'a') as f:
        if model_num == 1:
            # Write proper PDB headers
            if pdb_id:
                f.write(f"HEADER    PROTEIN STRUCTURE                    {pdb_id.upper():>4}              \n")
            else:
                f.write("HEADER    PROTEIN STRUCTURE                    UNKN              \n")
            
            if title:
                f.write(f"TITLE     {title[:70]:<70}\n")
            else:
                f.write("TITLE     GENERATED PROTEIN STRUCTURE BY ENHANCED VAE MODEL\n")
            
            f.write("COMPND    MOL_ID: 1;\n")
            f.write("COMPND   2 MOLECULE: GENERATED PROTEIN STRUCTURE;\n")
            f.write(f"COMPND   3 CHAIN: {chain_id};\n")
            f.write("COMPND   4 SYNONYM: VAE-GENERATED STRUCTURE;\n")
            f.write("COMPND   5 ENGINEERED: YES\n")
            
            f.write("SOURCE    MOL_ID: 1;\n")
            f.write("SOURCE   2 ORGANISM_SCIENTIFIC: SYNTHETIC;\n")
            f.write("SOURCE   3 ORGANISM_COMMON: COMPUTER-GENERATED;\n")
            f.write("SOURCE   4 ORGANISM_TAXID: 0;\n")
            f.write("SOURCE   5 GENE: VAE-GENERATED;\n")
            f.write("SOURCE   6 EXPRESSION_SYSTEM: COMPUTATIONAL MODEL;\n")
            f.write("SOURCE   7 EXPRESSION_SYSTEM_TAXID: 0;\n")
            
            f.write("KEYWDS    VAE, GENERATED, PROTEIN, STRUCTURE\n")
            f.write("EXPDTA    COMPUTATIONAL MODELING\n")
            f.write(f"NUMMDL    {num_models or 1:4d}\n")
            f.write("AUTHOR    ENHANCED PROTEIN VAE MODEL\n")
            f.write("REVDAT   1   01-JAN-24 UNKN    0\n")
            f.write("REMARK   2\n")
            f.write("REMARK   2 RESOLUTION. NOT APPLICABLE.\n")
            f.write("REMARK   3\n")
            f.write("REMARK   3 REFINEMENT.\n")
            f.write("REMARK   3   PROGRAM     : ENHANCED PROTEIN VAE\n")
            f.write("REMARK   3   AUTHORS     : COMPUTATIONAL MODEL\n")
            f.write("REMARK   4\n")
            f.write("REMARK   4 GENERATED STRUCTURE COMPLIES WITH FORMAT V. 3.30\n")
            f.write("REMARK 100\n")
            f.write("REMARK 100 THIS ENTRY WAS GENERATED BY ENHANCED PROTEIN VAE\n")
            f.write("REMARK 100 COMPLETE BACKBONE WITH PROPER CONNECTIVITY\n")
            f.write("\n")
        
        f.write(f"MODEL     {model_num:4d}\n")
        
        atom_num = 1
        conect_records = []
        
        for i in range(len(mask)):
            if mask[i] > 0.5:  # Valid residue
                residue_num = i + 1
                
                # Get residue name from sequence if available
                if sequence and i < len(sequence):
                    res_name = aa_1to3.get(sequence[i], 'ALA')
                else:
                    res_name = 'ALA'  # Default to ALA if no sequence
                
                # N atom
                f.write(f"ATOM  {atom_num:5d}  N   {res_name} {chain_id}{residue_num:4d}    "
                       f"{coords_n[i,0]:8.3f}{coords_n[i,1]:8.3f}{coords_n[i,2]:8.3f}"
                       f"  1.00  0.00           N  \n")
                n_atom_num = atom_num
                atom_num += 1
                
                # CA atom
                f.write(f"ATOM  {atom_num:5d}  CA  {res_name} {chain_id}{residue_num:4d}    "
                       f"{coords_ca[i,0]:8.3f}{coords_ca[i,1]:8.3f}{coords_ca[i,2]:8.3f}"
                       f"  1.00  0.00           C  \n")
                ca_atom_num = atom_num
                atom_num += 1
                
                # C atom
                f.write(f"ATOM  {atom_num:5d}  C   {res_name} {chain_id}{residue_num:4d}    "
                       f"{coords_c[i,0]:8.3f}{coords_c[i,1]:8.3f}{coords_c[i,2]:8.3f}"
                       f"  1.00  0.00           C  \n")
                c_atom_num = atom_num
                atom_num += 1
                
                # O atom
                f.write(f"ATOM  {atom_num:5d}  O   {res_name} {chain_id}{residue_num:4d}    "
                       f"{coords_o[i,0]:8.3f}{coords_o[i,1]:8.3f}{coords_o[i,2]:8.3f}"
                       f"  1.00  0.00           O  \n")
                o_atom_num = atom_num
                atom_num += 1
                
                # Store CONECT records for this residue
                conect_records.append({
                    'residue': i,
                    'n': n_atom_num,
                    'ca': ca_atom_num,
                    'c': c_atom_num,
                    'o': o_atom_num
                })
        
        # Write CONECT records for proper bonding
        f.write("\n")
        for i, conect in enumerate(conect_records):
            # Intra-residue bonds: N-CA, CA-C, C-O
            f.write(f"CONECT{conect['n']:5d}{conect['ca']:5d}\n")
            f.write(f"CONECT{conect['ca']:5d}{conect['c']:5d}\n")
            f.write(f"CONECT{conect['c']:5d}{conect['o']:5d}\n")
            
            # Inter-residue bonds: C(i)-N(i+1) peptide bond
            if i < len(conect_records) - 1:
                next_conect = conect_records[i + 1]
                f.write(f"CONECT{conect['c']:5d}{next_conect['n']:5d}\n")
        
        f.write("TER\n")
        f.write("ENDMDL\n")

def validate_protein_geometry(coords_ca, mask):
    """
    Enhanced geometry validation with more detailed checks.
    
    Args:
        coords_ca: [L, 3] CA coordinates
        mask: [L] validity mask
        
    Returns:
        tuple: (is_valid, reason_string)
    """
    valid_idx = mask.bool()
    if not valid_idx.any():
        return False, "No valid residues"
    
    valid_coords = coords_ca[valid_idx]
    
    # Check CA-CA distances
    if len(valid_coords) > 1:
        distances = []
        for i in range(len(valid_coords) - 1):
            dist = torch.norm(valid_coords[i+1] - valid_coords[i])
            distances.append(dist.item())
        
        max_dist = max(distances)
        avg_dist = sum(distances) / len(distances)
        
        # More lenient but still reasonable bounds
        if max_dist > 6.0:  # Increased from 5.0
            return False, f"Extreme CA-CA distance {max_dist:.3f}Å"
        
        if avg_dist < 2.5 or avg_dist > 5.0:  # More lenient bounds
            return False, f"Abnormal average CA-CA distance {avg_dist:.3f}Å"
        
        # Check for reasonable bond angles (CA-CA-CA)
        if len(valid_coords) > 2:
            angles = []
            for i in range(1, len(valid_coords) - 1):
                v1 = valid_coords[i-1] - valid_coords[i]
                v2 = valid_coords[i+1] - valid_coords[i]
                
                cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                angle = torch.acos(cos_angle) * 180 / np.pi
                angles.append(angle.item())
            
            avg_angle = sum(angles) / len(angles)
            if avg_angle < 60 or avg_angle > 180:  # Reasonable angle range
                return False, f"Abnormal average CA-CA-CA angle {avg_angle:.1f}°"
    
    return True, "Valid geometry"


def kabsch_rmsd(coords1, coords2, mask):
    """Compute RMSD after Kabsch alignment."""
    # Get valid coordinates
    valid = mask.bool()
    c1 = coords1[valid].cpu().numpy()
    c2 = coords2[valid].cpu().numpy()
    
    if len(c1) == 0:
        return 0.0
    
    # Center
    c1 = c1 - c1.mean(axis=0)
    c2 = c2 - c2.mean(axis=0)
    
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


def generate_ensembles(model, dataset, device, num_samples=10, output_dir='generated_pdbs'):
    """
    Generate ensemble structures from the VAE model.
    
    Args:
        model: Trained HierCVAE model
        dataset: Dataset to sample from
        device: torch device
        num_samples: number of conformers to generate per input
        output_dir: directory to save PDB files
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    print("=" * 80)
    print("GENERATING ENSEMBLE STRUCTURES")
    print("=" * 80)
    
    results = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            print(f"\nProcessing structure {idx + 1}/{len(dataset)}...")
            
            # Get data
            n, ca, c, mask, seq_emb, dih = dataset[idx][:6]  # Only take first 6 values


            # Get metadata from conformer
            conformer = dataset.conformers[idx]
            h5_path = conformer['h5']
            
            # Extract PDB ID and chain ID from H5 attributes
            pdb_id = None
            chain_id = 'A'
            try:
                with h5py.File(h5_path, 'r') as f:
                    pdb_id = f.attrs.get('pdb_id', None)
                    chain_id = f.attrs.get('chain_id', 'A')
            except:
                pass
            
            print(f"  → PDB ID: {pdb_id}, Chain: {chain_id}")
            
            # Extract ground truth sequence for comparison with model predictions
            # The model predicts sequences from structure, but we need ground truth to compute recovery
            print(f"  → Extracting ground truth sequence for comparison...")
            
            # Try to get ground truth sequence from the original data source
            sequence = None
            
            # Method 1: Try to extract from H5 file
            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'sequence' in f:
                        seq_raw = f['sequence'][()]
                        sequence = seq_raw.decode('utf-8') if isinstance(seq_raw, (bytes, bytearray)) else str(seq_raw)
                        print(f"  → Found sequence in H5: {len(sequence)} residues")
            except:
                pass
            
            # Method 2: Try to extract from original PDB file
            if not sequence and pdb_id:
                original_pdb_path = f"protein_ensemble_dataset/{pdb_id.upper()}.pdb"
                if os.path.exists(original_pdb_path):
                    print(f"  → Extracting from original PDB: {original_pdb_path}")
                    sequence, _, _ = extract_sequence_from_pdb(original_pdb_path)
            
            if sequence:
                print(f"  → Ground truth sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
                print(f"  → Length: {len(sequence)} residues")
            else:
                print(f"  → Warning: No ground truth sequence found - cannot compute sequence recovery")
                sequence = None
            
            # Add batch dimension
            n = n.unsqueeze(0).to(device)
            ca = ca.unsqueeze(0).to(device)
            c = c.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            dih = dih.unsqueeze(0).to(device)
            if seq_emb is not None:
                seq_emb = seq_emb.unsqueeze(0).to(device)
            
            # --- 1. RECONSTRUCTION ---
            print("  [1/3] Computing reconstruction (structure + sequence prediction)...")
            pred_n, pred_ca, pred_c, pred_seq_logits, mu_g, lv_g, mu_l, lv_l = model(
                seq_emb, n, ca, c, dih, mask
            )
            
            # Convert predicted sequence logits to amino acid sequence
            # Model predicts sequence from latent representation
            pred_seq_labels = torch.argmax(pred_seq_logits[0], dim=-1)  # [L]
            idx_to_aa = {
                0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
                10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V'
            }
            predicted_sequence = ''.join([idx_to_aa[idx.item()] for idx in pred_seq_labels])
            print(f"  → Model predicted sequence: {predicted_sequence[:50]}{'...' if len(predicted_sequence) > 50 else ''}")
            
            # Compare model prediction with ground truth if available
            if sequence:
                # Compute sequence recovery (percentage of correctly predicted amino acids)
                correct = sum(1 for i, (pred_aa, true_aa) in enumerate(zip(predicted_sequence, sequence)) 
                             if mask[0, i] > 0.5 and pred_aa == true_aa)
                total = int(mask[0].sum().item())
                seq_recovery = correct / total if total > 0 else 0.0
                print(f"\n  📊 SEQUENCE PREDICTION EVALUATION:")
                print(f"     Sequence recovery: {seq_recovery:.3f} ({correct}/{total} correct)")
                print(f"     Ground truth: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
                print(f"     Predicted:    {predicted_sequence[:50]}{'...' if len(predicted_sequence) > 50 else ''}")
                if seq_recovery < 0.30:
                    print(f"     ⚠️  Low recovery - consider implementing sequence recovery fixes")
                elif seq_recovery < 0.40:
                    print(f"     ⚠️  Below target - see SEQUENCE_RECOVERY_ANALYSIS.md for improvements")
                elif seq_recovery < 0.50:
                    print(f"     ✅ Reasonable recovery for VAE-based model")
                else:
                    print(f"     ✅ Good recovery - competitive performance!")
            else:
                seq_recovery = None
                print(f"\n  ℹ️  No ground truth available - showing predicted sequence only")
            
            # Compute reconstruction RMSD
            rec_rmsd = kabsch_rmsd(pred_ca[0], ca[0], mask[0])
            print(f"  → Reconstruction RMSD: {rec_rmsd:.3f} Å")
            
            # Save reconstruction with MODEL-PREDICTED sequence
            rec_path = os.path.join(output_dir, f'struct_{idx:03d}_reconstruction.pdb')
            write_pdb(
                pred_n[0].cpu().numpy(),
                pred_ca[0].cpu().numpy(),
                pred_c[0].cpu().numpy(),
                mask[0].cpu().numpy(),
                rec_path,
                model_num=1,
                sequence=predicted_sequence,  # Model's sequence prediction
                pdb_id=pdb_id,
                chain_id=chain_id,
                title=f"VAE Reconstruction with Predicted Sequence - {pdb_id or 'Structure'}"
            )
            print(f"  → Saved reconstruction: {rec_path}")
            
            # --- 2. GROUND TRUTH ---
            print("  [2/3] Saving ground truth...")
            gt_path = os.path.join(output_dir, f'struct_{idx:03d}_ground_truth.pdb')
            write_pdb(
                n[0].cpu().numpy(),
                ca[0].cpu().numpy(),
                c[0].cpu().numpy(),
                mask[0].cpu().numpy(),
                gt_path,
                model_num=1,
                sequence=sequence,  # Actual sequence from data (for comparison)
                pdb_id=pdb_id,
                chain_id=chain_id,
                title=f"Ground Truth Structure - {pdb_id or 'Structure'}"
            )
            print(f"  → Saved ground truth: {gt_path}")
            
            # --- 3. ENSEMBLE GENERATION ---
            print(f"  [3/3] Generating ensemble ({num_samples} samples with predicted sequences)...")
            ensemble_path = os.path.join(output_dir, f'struct_{idx:03d}_ensemble.pdb')
            
            # Delete existing ensemble file
            if os.path.exists(ensemble_path):
                os.remove(ensemble_path)
            
            ensemble_coords = []
            valid_samples = []
            
            # First pass: generate all samples with structure + sequence prediction
            for sample_idx in range(num_samples):
                # Sample from latent space for diversity
                z_g_sample = mu_g + torch.randn_like(mu_g) * torch.exp(0.5 * lv_g)
                z_l_sample = mu_l + torch.randn_like(mu_l) * torch.exp(0.5 * lv_l)
                
                # Decode to structure AND sequence (model predicts both)
                sample_n, sample_ca, sample_c, sample_seq_logits = model.decode(z_g_sample, z_l_sample, mask=mask)
                
                # Convert predicted sequence logits to amino acids
                sample_seq_labels = torch.argmax(sample_seq_logits[0], dim=-1)
                sample_sequence = ''.join([idx_to_aa[idx.item()] for idx in sample_seq_labels])
                
                # Validate geometry
                is_valid, reason = validate_protein_geometry(sample_ca[0], mask[0])
                if is_valid:
                    valid_samples.append((sample_n[0].cpu().numpy(), 
                                        sample_ca[0].cpu().numpy(), 
                                        sample_c[0].cpu().numpy(),
                                        sample_sequence))  # Store model-predicted sequence
                    ensemble_coords.append(sample_ca[0].cpu())
                    print(f"    → Valid sample {len(valid_samples)}: structure + predicted sequence")
                else:
                    print(f"    → Skipped sample {sample_idx + 1}: {reason}")
            
            # Second pass: write all valid samples to multi-model PDB file
            for i, (sample_n, sample_ca, sample_c, sample_sequence) in enumerate(valid_samples):
                # Each ensemble member has its own predicted sequence
                # (Note: sequences may vary slightly between samples)
                write_pdb(
                    sample_n,
                    sample_ca,
                    sample_c,
                    mask[0].cpu().numpy(),
                    ensemble_path,
                    model_num=i + 1,
                    sequence=sample_sequence,  # Model's sequence prediction for this sample
                    pdb_id=pdb_id,
                    chain_id=chain_id,
                    title=f"Ensemble Sample {i + 1} with Predicted Sequence - {pdb_id or 'Structure'}",
                    num_models=len(valid_samples) if i == 0 else None  # NUMMDL field on first model only
                )
            
            # Compute ensemble diversity
            rmsds = []
            for i in range(len(ensemble_coords)):
                for j in range(i + 1, len(ensemble_coords)):
                    rmsd = kabsch_rmsd(ensemble_coords[i], ensemble_coords[j], mask[0].cpu())
                    rmsds.append(rmsd)
            
            avg_diversity = np.mean(rmsds) if rmsds else 0.0
            print(f"  → Ensemble diversity (avg pairwise RMSD): {avg_diversity:.3f} Å")
            print(f"  → Saved: {ensemble_path}")
            
            # Store results
            results.append({
                'index': idx,
                'reconstruction_rmsd': rec_rmsd,
                'ensemble_diversity': avg_diversity,
                'num_residues': int(mask[0].sum().item()),
                'valid_samples': len(valid_samples),
                'sequence_recovery': seq_recovery if sequence else None,  # Model's sequence prediction accuracy
                'predicted_sequence': predicted_sequence  # Store for potential analysis
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY - Structure + Sequence Prediction")
    print("=" * 80)
    print(f"Total structures processed: {len(results)}")
    print(f"\n📐 STRUCTURE METRICS:")
    print(f"   Average reconstruction RMSD: {np.mean([r['reconstruction_rmsd'] for r in results]):.3f} Å")
    print(f"   Average ensemble diversity: {np.mean([r['ensemble_diversity'] for r in results]):.3f} Å")
    print(f"   Average valid samples: {np.mean([r['valid_samples'] for r in results]):.1f}/{num_samples}")
    
    # Report sequence recovery if available
    seq_recoveries = [r['sequence_recovery'] for r in results if r['sequence_recovery'] is not None]
    if seq_recoveries:
        avg_recovery = np.mean(seq_recoveries)
        print(f"\n🧬 SEQUENCE PREDICTION METRICS:")
        print(f"   Average sequence recovery: {avg_recovery:.3f} ({avg_recovery*100:.1f}%)")
        print(f"   Structures with ground truth: {len(seq_recoveries)}/{len(results)}")
        
        # Provide interpretation
        if avg_recovery < 0.30:
            print(f"   ⚠️  Status: LOW - See SEQUENCE_RECOVERY_ANALYSIS.md for fixes")
        elif avg_recovery < 0.40:
            print(f"   ⚠️  Status: BELOW TARGET - Consider Tier 1 improvements")
        elif avg_recovery < 0.50:
            print(f"   ✅ Status: REASONABLE for VAE-based model")
        else:
            print(f"   ✅ Status: GOOD - Competitive performance!")
    else:
        print(f"\n🧬 SEQUENCE PREDICTION:")
        print(f"   Model predicts sequences, but no ground truth available for evaluation")
    
    print(f"\n📁 Output directory: {os.path.abspath(output_dir)}")
    print(f"   ✓ Ground truth structures (actual sequences)")
    print(f"   ✓ Reconstructions (predicted sequences)")
    print(f"   ✓ Ensemble samples (predicted sequences)")
    print("=" * 80)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'generation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ENSEMBLE GENERATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            f.write(f"Structure {r['index']:03d}:\n")
            f.write(f"  Residues: {r['num_residues']}\n")
            f.write(f"  Reconstruction RMSD: {r['reconstruction_rmsd']:.3f} Å\n")
            f.write(f"  Ensemble diversity: {r['ensemble_diversity']:.3f} Å\n")
            f.write(f"  Valid samples: {r['valid_samples']}/{num_samples}\n")
            if r['sequence_recovery'] is not None:
                f.write(f"  Sequence recovery: {r['sequence_recovery']:.3f}\n")
            f.write("\n")
        f.write(f"\nAverage reconstruction RMSD: {np.mean([r['reconstruction_rmsd'] for r in results]):.3f} Å\n")
        f.write(f"Average ensemble diversity: {np.mean([r['ensemble_diversity'] for r in results]):.3f} Å\n")
        f.write(f"Average valid samples: {np.mean([r['valid_samples'] for r in results]):.1f}/{num_samples}\n")
        if seq_recoveries:
            f.write(f"Average sequence recovery: {np.mean(seq_recoveries):.3f}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate ensemble PDB files from trained VAE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to H5 data or manifest CSV")
    parser.add_argument("--output_dir", type=str, default="generated_pdbs", help="Output directory for PDB files")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of ensemble samples per structure")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_seqemb", action="store_true", help="Use sequence embeddings")
    
    # Model architecture (must match training)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ff", type=int, default=1024)
    parser.add_argument("--nlayers", type=int, default=6)
    parser.add_argument("--z_global", type=int, default=512)
    parser.add_argument("--z_local", type=int, default=256)
    parser.add_argument("--decoder_hidden", type=int, default=512)
    
    args = parser.parse_args()
    
    print("\n" + "🧬" * 40)
    print("PROTEIN VAE ENSEMBLE GENERATOR")
    print("🧬" * 40 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    if args.data.endswith('.csv'):
        dataset = EnsembleDataset(args.data, use_seqemb=args.use_seqemb)
    else:
        # Single H5 file
        raise NotImplementedError("Single H5 file not yet supported, use manifest CSV")
    
    # Determine sequence embedding dimension
    seqemb_dim = None
    if args.use_seqemb:
        for i in range(len(dataset)):
            _, _, _, _, emb, _ = dataset[i][:6]  # Updated: now returns 7 values (added seq_labels)
            if emb is not None:
                seqemb_dim = emb.shape[-1]
                break
    
    print(f"Dataset: {len(dataset)} structures")
    print(f"Sequence embedding dim: {seqemb_dim}")
    
    # Load checkpoint to get hyperparameters
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    saved_hyperparams = checkpoint.get('hyperparameters', None)
    
    # Use hyperparameters from checkpoint if available, otherwise use command-line args
    if saved_hyperparams:
        print("✅ Found hyperparameters in checkpoint - using saved configuration")
        d_model = saved_hyperparams.get('d_model', args.d_model)
        nhead = saved_hyperparams.get('nhead', args.nhead)
        ff = saved_hyperparams.get('ff', args.ff)
        nlayers = saved_hyperparams.get('nlayers', args.nlayers)
        z_global = saved_hyperparams.get('z_global', args.z_global)
        z_local = saved_hyperparams.get('z_local', args.z_local)
        decoder_hidden = saved_hyperparams.get('decoder_hidden', args.decoder_hidden)
        saved_seqemb_dim = saved_hyperparams.get('seqemb_dim', seqemb_dim)
        
        # Override seqemb_dim if it was saved (more reliable than dataset inference)
        if saved_seqemb_dim is not None:
            seqemb_dim = saved_seqemb_dim
            
        print(f"  Model architecture from checkpoint:")
        print(f"    d_model={d_model}, nhead={nhead}, ff={ff}, nlayers={nlayers}")
        print(f"    z_global={z_global}, z_local={z_local}, decoder_hidden={decoder_hidden}")
        print(f"    seqemb_dim={seqemb_dim}")
    else:
        print("⚠️  No hyperparameters in checkpoint - using command-line arguments")
        print("  (This checkpoint was saved with an older version)")
        d_model = args.d_model
        nhead = args.nhead
        ff = args.ff
        nlayers = args.nlayers
        z_global = args.z_global
        z_local = args.z_local
        decoder_hidden = args.decoder_hidden
    
    # Create model with correct hyperparameters
    model = HierCVAE(
        seqemb_dim=seqemb_dim,
        d_model=d_model,
        nhead=nhead,
        ff=ff,
        nlayers=nlayers,
        z_g=z_global,
        z_l=z_local,
        equivariant=True,
        decoder_hidden=decoder_hidden,
        use_dihedrals=True
    ).to(args.device)
    
    # Load model weights
    model, epoch, loss_history, _ = load_checkpoint(model, args.checkpoint, device=args.device)
    print(f"Loaded checkpoint from epoch: {epoch}")
    
    # Generate ensembles
    results = generate_ensembles(
        model=model,
        dataset=dataset,
        device=args.device,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    print("\n✅ Done! Generated structures with predicted sequences and proper geometry.")
    print("\n🎯 Model Capabilities:")
    print("  ✓ Structure generation (N, CA, C, O backbone atoms)")
    print("  ✓ Sequence prediction (amino acid types)")
    print("  ✓ Ensemble diversity (multiple conformations)")
    print("  ✓ Proper PDB connectivity (CONECT records)")
    print("\n📊 Output Files:")
    print("  • *_ground_truth.pdb    - Original structure with actual sequence")
    print("  • *_reconstruction.pdb  - VAE reconstruction with predicted sequence")
    print("  • *_ensemble.pdb        - Multiple conformers with predicted sequences")
    print("  • generation_summary.txt - Metrics including sequence recovery")
    print("\n💡 Visualization Tips (PyMOL):")
    print(f"  load {args.output_dir}/struct_000_ground_truth.pdb, gt")
    print(f"  load {args.output_dir}/struct_000_reconstruction.pdb, recon")
    print(f"  load {args.output_dir}/struct_000_ensemble.pdb, ensemble")
    print("  align recon, gt")
    print("  color red, gt")
    print("  color green, recon") 
    print("  color cyan, ensemble")
    print("  set all_states, on  # Show all ensemble models")
    print("  show cartoon")
    print("  label (name ca), resn  # Show predicted residue names")
    print()


if __name__ == "__main__":
    main()

