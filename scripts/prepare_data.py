#!/usr/bin/env python3
import os
import csv
import argparse
import random
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import h5py
from tqdm import tqdm
import requests

from Bio.PDB import MMCIFParser, MMCIF2Dict
from Bio.Data.IUPACData import protein_letters_3to1
from rcsbapi.search import AttributeQuery
from rcsbapi.search import search_attributes as attrs
from sklearn.model_selection import train_test_split

from Bio import pairwise2
from Bio.Align import substitution_matrices
import json


def pairwise_rmsd_matrix(coords: np.ndarray, mask: np.ndarray, min_common: int = 8) -> np.ndarray:
    """
    Compute K x K pairwise RMSD over common observed Cα positions.
    coords: [K, L, 3], mask: [K, L]
    Returns an array with np.nan where too few common positions.
    """
    K, L, _ = coords.shape
    rmsd = np.full((K, K), np.nan, dtype=np.float64)
    for i in range(K):
        for j in range(i+1, K):
            common = mask[i].astype(bool) & mask[j].astype(bool)
            idx = np.where(common)[0]
            if idx.size >= min_common:
                P, Q = coords[i, idx], coords[j, idx]
                R, t = kabsch(P, Q)
                aligned = (P @ R.T) + t
                diff = aligned - Q
                val = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
                rmsd[i, j] = rmsd[j, i] = val
    np.fill_diagonal(rmsd, 0.0)
    return rmsd


def choose_medoid(coords: np.ndarray, mask: np.ndarray, min_common: int = 8) -> int:
    """
    Choose medoid model = argmin over i of nanmean(RMSD(i, *)).
    Falls back to 0 if all-NaN (extremely rare with decent data).
    """
    D = pairwise_rmsd_matrix(coords, mask, min_common=min_common)
    means = np.nanmean(D, axis=1)  # ignore pairs with too few common residues
    if np.all(np.isnan(means)):
        return 0
    # deterministic tie-breaker
    medoid = int(np.nanargmin(means))
    return medoid


def align_to_reference(coords: np.ndarray, mask: np.ndarray, ref_idx: int, use_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Align each model to coords[ref_idx] using Kabsch on the intersection mask,
    optionally restricted by use_mask (e.g., core residues).
    """
    K, L, _ = coords.shape
    aligned = coords.copy()
    ref = coords[ref_idx]
    ref_mask = mask[ref_idx].astype(bool)
    if use_mask is None:
        use_mask = np.ones(L, dtype=bool)
    fit_mask_ref = ref_mask & use_mask
    for k in range(K):
        common = fit_mask_ref & mask[k].astype(bool)
        idx = np.where(common)[0]
        if idx.size >= 3:
            P, Q = coords[k, idx], ref[idx]
            R, t = kabsch(P, Q)
            aligned[k] = (coords[k] @ R.T) + t
    return aligned


def detect_core_mask(coords_aligned: np.ndarray, mask: np.ndarray, core_frac: float = 0.7, min_core_len: int = 30) -> np.ndarray:
    """
    Compute per-residue variance across models (x+y+z variances) and select
    the lowest-variance residues as 'core'.
    Only residues observed in >= half of models are eligible.
    """
    K, L, _ = coords_aligned.shape
    present_counts = mask.sum(axis=0)  # [L]
    eligible = present_counts >= (K // 2 + 1)

    arr = coords_aligned.astype(np.float64).copy()
    # NaN-out missing
    for k in range(K):
        arr[k, ~mask[k].astype(bool)] = np.nan
    # variance across models per residue (sum of per-axis variance)
    var_xyz = np.nanvar(arr, axis=0)  # [L, 3]
    var_score = np.nansum(var_xyz, axis=1)  # [L]

    # rank eligible residues by variance
    idx_eligible = np.where(eligible)[0]
    if idx_eligible.size == 0:
        # fallback: no eligible residues → take all observed
        return present_counts > 0

    n_core = max(min_core_len, int(np.ceil(core_frac * idx_eligible.size)))
    order = idx_eligible[np.argsort(var_score[idx_eligible])]
    core_idx = order[:n_core]
    core_mask = np.zeros(L, dtype=bool)
    core_mask[core_idx] = True
    return core_mask


def align_core_fit(coords: np.ndarray, mask: np.ndarray, core_frac: float, min_core_len: int,
                   min_common: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Full pipeline:
      1) pick medoid,
      2) initial align to medoid using all common residues,
      3) detect core on aligned coords,
      4) realign using only core residues.
    Returns: coords_aligned, medoid_idx, core_mask
    """
    medoid = choose_medoid(coords, mask, min_common=min_common)
    # initial alignment (all common)
    aligned0 = align_to_reference(coords, mask, ref_idx=medoid, use_mask=None)
    # core detection
    core_mask = detect_core_mask(aligned0, mask, core_frac=core_frac, min_core_len=min_core_len)
    # core-fit alignment
    aligned = align_to_reference(coords, mask, ref_idx=medoid, use_mask=core_mask)
    return aligned, medoid, core_mask


def compute_rmsf_core(coords_aligned: np.ndarray, mask: np.ndarray, use_mask: Optional[np.ndarray]) -> np.ndarray:
    """
    RMSF per residue across models on aligned coords. If use_mask is provided,
    RMSF outside use_mask is still reported but will include NaNs converted to 0.
    """
    arr = coords_aligned.astype(np.float64).copy()
    m = mask.astype(bool)
    if use_mask is not None:
        # keep values only where residue is in use_mask; NaN elsewhere
        for k in range(arr.shape[0]):
            m_k = m[k] & use_mask
            arr[k, ~m_k] = np.nan
    else:
        for k in range(arr.shape[0]):
            arr[k, ~m[k]] = np.nan
    mean = np.nanmean(arr, axis=0)    # [L, 3]
    diffs = arr - mean
    sq = np.nansum(diffs**2, axis=2)  # [K, L]
    rmsf = np.sqrt(np.nanmean(sq, axis=0))  # [L]
    return np.nan_to_num(rmsf).astype(np.float32)

def query_nmr_entries(min_models: int, shuffle_seed: int) -> List[str]:
    """
    Query RCSB for entries with NMR method and at least min_models deposited models.
    Returns a shuffled list of PDB IDs (4-char strings).

    Args:
        min_models: The minimum number of deposited models.
        shuffle_seed: A seed for shuffling the results to ensure reproducibility.

    Returns:
        A list of PDB IDs.
    """
    # Query for NMR experimental methods
    q_nmr = AttributeQuery(
        attribute="exptl.method",
        operator="in",
        value=["SOLUTION NMR", "SOLID-STATE NMR"]
    )
    
    # Query for entries with at least min_models
    q_models = attrs.rcsb_entry_info.deposited_model_count >= min_models
    
    # Combine the queries using a logical AND
    combined_query = q_nmr & q_models
    
    # Execute the query and convert the iterator to a list of PDB IDs
    pdb_ids = list(combined_query())
    
    # Shuffle the list for randomization
    random.Random(shuffle_seed).shuffle(pdb_ids)
    
    return pdb_ids


def download_mmcif(pdb_id: str, out_dir: str, retries: int, backoff: float) -> Optional[str]:
    """
    Download mmCIF from RCSB and return file path, or None on failure.
    Uses the plain CIF endpoint (not gz) to avoid decompression.
    """
    pdb_id = pdb_id.lower()
    os.makedirs(out_dir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    dest = os.path.join(out_dir, f"{pdb_id}.cif")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    headers = {"User-Agent": "ProteinEnsembleVAE/1.0"}
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.ok and len(r.content) > 0:
                with open(dest, "wb") as f:
                    f.write(r.content)
                return dest
        except Exception:
            pass
        time.sleep(delay)
        delay *= backoff
    return None


# ---------------------------
# Geometry helpers
# ---------------------------

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return rotation R and translation t aligning P -> Q (least squares)."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = Q.mean(axis=0) - (R @ P.mean(axis=0))
    return R, t

# ---------------------------
# Extended geometry helpers
# ---------------------------

def dihedral(p0, p1, p2, p3) -> float:
    """
    Return dihedral angle (radians) for 4 points p0..p3.
    """
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    # normalize b1
    b1n = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - (b0 @ b1n) * b1n
    w = b2 - (b2 @ b1n) * b1n
    x = (v @ w)
    y = np.linalg.norm(np.cross(b1n, v)) * np.linalg.norm(w) * np.sign(np.dot(np.cross(b1n, v), w))
    return np.arctan2(y + 1e-12, x + 1e-12)

def build_local_frame(N, CA, C):
    """
    Build a right-handed local frame at residue using N, CA, C.
    Returns a 3x3 rotation matrix R whose columns are (ex, ey, ez).
    - ex: normalized CA->C
    - t: normalized CA->N
    - ez: normalized ex x t
    - ey: ez x ex
    If inputs are invalid, returns None.
    """
    v1 = C - CA
    v2 = N - CA
    if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        return None
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    ex = v1 / n1
    t  = v2 / n2
    ez = np.cross(ex, t)
    nz = np.linalg.norm(ez)
    if nz < 1e-6:
        return None
    ez = ez / nz
    ey = np.cross(ez, ex)
    R = np.stack([ex, ey, ez], axis=1)  # 3x3
    return R

def safe_sincos(x):
    s = np.sin(x); c = np.cos(x)
    return np.stack([s, c], axis=-1)  # [..., 2]

def compute_backbone_torsions(N_all, CA_all, C_all, mask_all):
    """
    Compute ϕ/ψ/ω torsions per residue per model.
    Inputs: N_all, CA_all, C_all: [K, L, 3]; mask_all: [K, L] (1 if residue exists with all needed atoms)
    Returns dict with sincos arrays of shape [K, L, 2] each.
    Undefined angles (ends or missing neighbors) -> sin,cos = 0,0 (mask them with a separate mask if needed).
    """
    K, L, _ = CA_all.shape
    phi = np.zeros((K, L), dtype=np.float32)
    psi = np.zeros((K, L), dtype=np.float32)
    omg = np.zeros((K, L), dtype=np.float32)

    # A residue i needs neighbors:
    # phi(i) uses C_{i-1}, N_i, CA_i, C_i
    # psi(i) uses N_i, CA_i, C_i, N_{i+1}
    # omega(i) uses CA_{i-1}, C_{i-1}, N_i, CA_i   (peptide bond)
    # We require presence of all atoms for those positions.
    m = mask_all.astype(bool)

    for k in range(K):
        N = N_all[k]; CA = CA_all[k]; C = C_all[k]
        mk = m[k]
        for i in range(L):
            # phi
            if i-1 >= 0 and mk[i] and mk[i-1]:
                a = C[i-1]; b = N[i]; c = CA[i]; d = C[i]
                if not (np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)) or np.any(np.isnan(d))):
                    phi[k, i] = dihedral(a, b, c, d)
            # psi
            if i+1 < L and mk[i] and mk[i+1]:
                a = N[i]; b = CA[i]; c = C[i]; d = N[i+1]
                if not (np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)) or np.any(np.isnan(d))):
                    psi[k, i] = dihedral(a, b, c, d)
            # omega (peptide bond) around C_{i-1}-N_i; approximated with CA anchors
            if i-1 >= 0 and mk[i] and mk[i-1]:
                a = CA[i-1]; b = C[i-1]; c = N[i]; d = CA[i]
                if not (np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)) or np.any(np.isnan(d))):
                    omg[k, i] = dihedral(a, b, c, d)

    phi_sc = safe_sincos(phi)
    psi_sc = safe_sincos(psi)
    omg_sc = safe_sincos(omg)
    return {"phi_sincos": phi_sc, "psi_sincos": psi_sc, "omega_sincos": omg_sc}

def compute_pair_features_trrosetta(coords_ca, N_all, CA_all, C_all, mask_all, models, medoid_idx):
    """
    Compute pairwise features (d, omega, theta, phi) in trRosetta spirit.
    - d: Cα-Cα distance
    - omega: dihedral between residue-plane normals around inter-residue vector
    - theta, phi: spherical orientation of j seen from i in i's local frame.
    Returns dict of LxL (medoid) or KxLxL (all models). We default to medoid to save space.
    """
    K, L, _ = coords_ca.shape
    use_models = [medoid_idx] if models == "medoid" else list(range(K))

    def per_model(k):
        ca = coords_ca[k]       # [L,3]
        N = N_all[k]; CA = CA_all[k]; C = C_all[k]
        m = mask_all[k].astype(bool)
        # distances
        diff = ca[:, None, :] - ca[None, :, :]  # [L,L,3]
        d = np.linalg.norm(diff, axis=-1)       # [L,L]
        d[~(m[:, None] & m[None, :])] = np.nan

        # local frames for orientation
        frames = [build_local_frame(N[i], CA[i], C[i]) if m[i] else None for i in range(L)]

        # theta, phi: spherical coords of r_ij in i-frame
        theta = np.full((L, L), np.nan, dtype=np.float32)
        phi   = np.full((L, L), np.nan, dtype=np.float32)
        # omega (dihedral between residue planes around r_ij axis)
        omega = np.full((L, L), np.nan, dtype=np.float32)

        for i in range(L):
            if (not m[i]) or (frames[i] is None):
                continue
            Ri = frames[i]             # 3x3
            for j in range(L):
                if not m[j] or i == j:
                    continue
                rij = ca[j] - ca[i]
                nr = np.linalg.norm(rij)
                if nr < 1e-6:
                    continue
                # spherical in i-frame
                v = Ri.T @ (rij / nr)  # coordinates in i frame
                x, y, z = v
                # theta: angle from +z (0..pi), phi: azimuth (-pi..pi)
                th = np.arccos(np.clip(z, -1.0, 1.0))
                ph = np.arctan2(y, x)
                theta[i, j] = th
                phi[i, j]   = ph

                # omega: dihedral between normals of planes (Ni,CAi,Ci) and (Nj,CAj,Cj) around axis rij
                if frames[j] is not None:
                    # plane normals (ez from our local frames)
                    ez_i = Ri[:, 2]
                    Rj = frames[j]; ez_j = Rj[:, 2]
                    # Project normals to plane orthogonal to rij
                    axis = rij / nr
                    ez_i_perp = ez_i - (ez_i @ axis) * axis
                    ez_j_perp = ez_j - (ez_j @ axis) * axis
                    ni = np.linalg.norm(ez_i_perp); nj = np.linalg.norm(ez_j_perp)
                    if ni > 1e-6 and nj > 1e-6:
                        ei = ez_i_perp / ni
                        ej = ez_j_perp / nj
                        # signed angle from ei to ej around axis
                        x = np.clip(ei @ ej, -1.0, 1.0)
                        y = np.dot(axis, np.cross(ei, ej))
                        omega[i, j] = np.arctan2(y + 1e-12, x + 1e-12)

        return {
            "d": d.astype(np.float32),
            "omega": omega.astype(np.float32),
            "theta": theta.astype(np.float32),
            "phi": phi.astype(np.float32),
        }

    out = per_model(use_models[0])
    if models == "all":
        D = np.zeros((K, L, L), np.float32) * np.nan
        Om = np.zeros_like(D); Th = np.zeros_like(D); Ph = np.zeros_like(D)
        for idx, k in enumerate(use_models):
            o = per_model(k)
            D[k]  = o["d"];  Om[k] = o["omega"]; Th[k] = o["theta"]; Ph[k] = o["phi"]
        return {"d": D, "omega": Om, "theta": Th, "phi": Ph, "models": models}
    else:
        return {"d": out["d"], "omega": out["omega"], "theta": out["theta"], "phi": out["phi"], "models": models}



# ---------------------------
# Parsing helpers
# ---------------------------

def is_poly_peptide_residue(res) -> bool:
    hetfield, resseq, icode = res.id
    return hetfield == " "  # standard residue (not HETATM)


def chain_is_polypeptide(chain) -> bool:
    for res in chain.get_residues():
        if is_poly_peptide_residue(res) and res.has_id("CA"):
            return True
    return False


def build_reference_residue_list(model, chain_id) -> Tuple[List[Tuple[int, str]], List[str]]:
    """Return [(resseq, icode)], [resname] for a chain in the given model."""
    chain = model[chain_id]
    ref_keys, resnames = [], []
    for res in chain.get_residues():
        if not is_poly_peptide_residue(res):
            continue
        ref_keys.append((res.id[1], res.id[2]))  # (resseq, icode)
        resnames.append(res.get_resname())
    return ref_keys, resnames


def sequence_from_resnames(resnames: List[str]) -> str:
    """
    Convert 3-letter residue names to 1-letter amino acid sequence.
    Handles standard and some common non-standard residue names.
    """
    # Extended mapping including common non-standard amino acids
    extended_mapping = {
        # Standard 20 amino acids
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        # Common alternative names and protonation states
        'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'HID': 'H', 'HIE': 'H', 'HIP': 'H',  # Histidine variants
        'CYX': 'C', 'CYM': 'C',  # Cysteine variants (disulfide, deprotonated)
        'ASH': 'D', 'GLH': 'E',  # Protonated asp/glu
        'LYN': 'K',  # Deprotonated lysine
        # Selenomethionine and other modifications
        'MSE': 'M',  # Selenomethionine (very common in X-ray structures)
        'SEP': 'S', 'TPO': 'T', 'PTR': 'Y',  # Phosphorylated residues
        # Other common modifications
        'MLY': 'K', 'ALY': 'K',  # Modified lysines
        'HYP': 'P',  # Hydroxyproline
        'CSO': 'C', 'CSS': 'C',  # Oxidized cysteines
    }
    
    chars = []
    unknown_residues = set()
    
    for rn in resnames:
        # Clean up residue name
        rn_clean = rn.strip().upper()
        
        # Try extended mapping first
        if rn_clean in extended_mapping:
            aa = extended_mapping[rn_clean]
        # Then try BioPython's dictionary
        elif rn_clean in protein_letters_3to1:
            aa = protein_letters_3to1[rn_clean]
        else:
            # Unknown residue - mark as X and track it
            aa = "X"
            unknown_residues.add(rn)
        
        chars.append(aa)
    
    # Warn about unknown residues (but don't fail)
    if unknown_residues:
        print(f"  ⚠️  Unknown residue names (converted to X): {', '.join(sorted(unknown_residues))}")
    
    return "".join(chars)


def extract_ca_coords_for_chain(structure, chain_id: str, ref_keys: List[Tuple[int, str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract CA coords per model for chain in the order of ref_keys."""
    models = list(structure.get_models())
    K = len(models)
    L = len(ref_keys)
    coords = np.full((K, L, 3), np.nan, dtype=np.float32)
    mask = np.zeros((K, L), dtype=np.uint8)

    for k, model in enumerate(models):
        if chain_id not in [c.id for c in model]:
            continue
        chain = model[chain_id]
        res_map: Dict[Tuple[int, str], object] = {}
        for res in chain.get_residues():
            if is_poly_peptide_residue(res):
                res_map[(res.id[1], res.id[2])] = res
        for j, key in enumerate(ref_keys):
            res = res_map.get(key)
            if res is not None and res.has_id("CA"):
                coords[k, j] = res["CA"].get_coord().astype(np.float32)
                mask[k, j] = 1
    return coords, mask

def extract_backbone_coords_for_chain(structure, chain_id: str, ref_keys: List[Tuple[int, str]]):
    """
    Extract per-model N, CA, C coords for residues matching ref_keys order.
    Returns (N, CA, C, mask) with shapes:
      N, CA, C: [K, L, 3] float32; mask: [K, L] uint8 (1 if all three atoms present)
    """
    models = list(structure.get_models())
    K = len(models); L = len(ref_keys)
    N_all  = np.full((K, L, 3), np.nan, dtype=np.float32)
    CA_all = np.full((K, L, 3), np.nan, dtype=np.float32)
    C_all  = np.full((K, L, 3), np.nan, dtype=np.float32)
    mask   = np.zeros((K, L), dtype=np.uint8)

    for k, model in enumerate(models):
        if chain_id not in [c.id for c in model]:
            continue
        chain = model[chain_id]
        res_map: Dict[Tuple[int, str], object] = {}
        for res in chain.get_residues():
            if is_poly_peptide_residue(res):
                res_map[(res.id[1], res.id[2])] = res
        for j, key in enumerate(ref_keys):
            res = res_map.get(key)
            if res is None:
                continue
            hasN = res.has_id("N"); hasCA = res.has_id("CA"); hasC = res.has_id("C")
            if hasN:  N_all[k, j]  = res["N"].get_coord().astype(np.float32)
            if hasCA: CA_all[k, j] = res["CA"].get_coord().astype(np.float32)
            if hasC:  C_all[k, j]  = res["C"].get_coord().astype(np.float32)
            if hasN and hasCA and hasC:
                mask[k, j] = 1
    return N_all, CA_all, C_all, mask

# ---------------------------
# Cross-PDB helpers
# ---------------------------

def global_align_map(base_seq: str, cand_seq: str, match, mismatch, gap_open, gap_extend):
    """
    Global alignment (Needleman-Wunsch) to map base positions (0..L-1) to candidate positions.
    Returns an index array 'map_idx' of length L with candidate positions or -1 for gaps.
    """
    if not base_seq or not cand_seq:
        return np.full((len(base_seq),), -1, dtype=np.int32)
    matrix = substitution_matrices.load_matrix("BLOSUM62")
    alns = pairwise2.align.globalds(base_seq, cand_seq, matrix, gap_open, gap_extend)
    if not alns:
        return np.full((len(base_seq),), -1, dtype=np.int32)
    a_base, a_cand, _, _, _ = alns[0]
    map_idx = []
    i = j = 0
    for ab, ac in zip(a_base, a_cand):
        if ab != '-' and ac != '-':
            map_idx.append(j)
            i += 1; j += 1
        elif ab != '-' and ac == '-':
            map_idx.append(-1); i += 1
        elif ab == '-' and ac != '-':
            j += 1
    return np.array(map_idx, dtype=np.int32)

def extract_metadata_mmcif(cif_path: str):
    """
    Read common metadata (method, resolution, pH, temperature, ligands) from mmCIF via MMCIF2Dict.
    """
    try:
        d = MMCIF2Dict(cif_path)
    except Exception:
        return {}
    md = {}
    # methods (may be multi-row)
    method = d.get('_exptl.method', None)
    if isinstance(method, list): method = method[0]
    md["method"] = str(method) if method else ""
    # resolution
    res = d.get('_refine.ls_d_res_high', d.get('_em_3d_reconstruction.resolution', None))
    if isinstance(res, list): res = res[0]
    try:
        md["resolution"] = float(res) if res not in (None, '?', '.') else np.nan
    except Exception:
        md["resolution"] = np.nan
    # pH
    ph = d.get('_exptl_crystal.pH', d.get('_pH', None))
    if isinstance(ph, list): ph = ph[0]
    try:
        md["pH"] = float(ph) if ph not in (None, '?', '.') else np.nan
    except Exception:
        md["pH"] = np.nan
    # temperature (Kelvin)
    temp = d.get('_diffrn.ambient_temp', d.get('_em_3d_reconstruction.temperature', None))
    if isinstance(temp, list): temp = temp[0]
    try:
        md["temperature_K"] = float(temp) if temp not in (None, '?', '.') else np.nan
    except Exception:
        md["temperature_K"] = np.nan
    # ligands: all non-water HET codes present
    chem_ids = d.get('_chem_comp.id', [])
    chem_types = d.get('_chem_comp.type', [])
    ligs = []
    if isinstance(chem_ids, list) and isinstance(chem_types, list):
        for c, t in zip(chem_ids, chem_types):
            if c and c not in ("HOH", "WAT") and str(t).lower().startswith(("non-polymer", "ligand")):
                ligs.append(c)
    md["ligands"] = "+".join(sorted(set(ligs))) if ligs else ""
    return md

def compute_identity_coverage(base_seq: str, cand_seq: str, map_idx: np.ndarray):
    """
    Simple identity and coverage using the mapping base->cand.
    """
    matches = 0; covered = 0
    j_prev = -2
    for i, j in enumerate(map_idx):
        if j >= 0 and j < len(cand_seq):
            covered += 1
            if base_seq[i] == cand_seq[j]:
                matches += 1
    identity = matches / max(1, covered)
    coverage = covered / max(1, len(base_seq))
    return identity, coverage

def build_coords_from_mapping(model, chain_id, ref_keys, map_idx):
    """
    For the first model in a non-NMR entry (usually single model), build Lx3 arrays for N, CA, C.
    """
    L = len(ref_keys)
    N = np.full((L, 3), np.nan, dtype=np.float32)
    CA = np.full((L, 3), np.nan, dtype=np.float32)
    C = np.full((L, 3), np.nan, dtype=np.float32)
    mask = np.zeros((L,), dtype=np.uint8)
    if chain_id not in [c.id for c in model]:
        return N, CA, C, mask
    chain = model[chain_id]
    # Build candidate residues by sequence order
    cand_res = [res for res in chain.get_residues() if is_poly_peptide_residue(res)]
    for i, j in enumerate(map_idx):
        if j < 0 or j >= len(cand_res):
            continue
        res = cand_res[j]
        has = res.has_id("N") and res.has_id("CA") and res.has_id("C")
        if has:
            N[i]  = res["N"].get_coord().astype(np.float32)
            CA[i] = res["CA"].get_coord().astype(np.float32)
            C[i]  = res["C"].get_coord().astype(np.float32)
            mask[i] = 1
    return N, CA, C, mask

def get_uniprot_accessions_from_cif(cif_path: str) -> List[str]:
    """
    Try to pull UniProt accessions from mmCIF via _struct_ref.* (SIFTS link).
    Returns a list of strings (may be empty).
    """
    accs = []
    try:
        d = MMCIF2Dict(cif_path)
        db_names = d.get('_struct_ref.db_name', [])
        db_codes = d.get('_struct_ref.db_code', [])
        if not isinstance(db_names, list): db_names = [db_names]
        if not isinstance(db_codes, list): db_codes = [db_codes]
        for db, code in zip(db_names, db_codes):
            if str(db).strip().upper() in ('UNP', 'UNIPROT') and code not in ('?', '.'):
                accs.append(str(code).strip())
    except Exception:
        pass
    return sorted(list(set(accs)))

def find_crosspdb_candidates_by_uniprot(uniprot_accs: List[str], max_hits: int) -> List[str]:
    """
    Use rcsbapi to fetch PDB entry IDs that contain polymer entities mapped to any UniProt in uniprot_accs.
    Returns a list of 4-char PDB IDs.
    """
    if not uniprot_accs:
        return []
    q_dbname  = attrs.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name == "UniProt"
    q_dbacc   = AttributeQuery(
        attribute="rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
        operator="in",
        value=uniprot_accs
    )
    q_polypep = attrs.entity_poly.entity_type == "polypeptide(L)"
    q = q_dbname & q_dbacc & q_polypep
    # rcsbapi returns entry IDs for entity-level queries, typically 4-char codes
    hits = list(q())
    # Normalize & uniq
    hits = [h.lower() for h in hits if isinstance(h, str) and len(h) == 4]
    # De-duplicate, cap
    out = []
    seen = set()
    for h in hits:
        if h not in seen:
            out.append(h); seen.add(h)
        if len(out) >= max_hits:
            break
    return out

def append_crosspdb_conformers(
    base_pdb_id: str,
    base_chain_id: str,
    cif_path: str,
    ref_keys: List[Tuple[int, str]],
    base_sequence: str,
    medoid_idx: int,
    core_mask: np.ndarray,
    coords_ca_aligned: np.ndarray,   # [K,L,3] (aligned)
    raw_dir: str,
    h5_path: str,
    identity_thresh: float,
    coverage_thresh: float,
    max_models: int
) -> int:
    """
    Discover same-UniProt entries, align their chains into the base frame, and append to /crosspdb.
    Returns number of conformers appended.
    """
    # 1) Discover candidates
    unp = get_uniprot_accessions_from_cif(cif_path)
    cand_pdbs = find_crosspdb_candidates_by_uniprot(unp, max_hits=1000) if unp else []
    # Drop self
    cand_pdbs = [p for p in cand_pdbs if p != base_pdb_id.lower()]
    if not cand_pdbs:
        return 0

    parser = MMCIFParser(QUIET=True)
    L = len(ref_keys)
    added = 0
    coords_list = []
    mask_list   = []
    meta_list   = []

    for cand_pdb in cand_pdbs:
        cand_cif = download_mmcif(cand_pdb, raw_dir, retries=3, backoff=1.5)
        if not cand_cif:
            continue
        try:
            st = parser.get_structure(cand_pdb, cand_cif)
        except Exception:
            continue

        meta_base = extract_metadata_mmcif(cand_cif)
        # For each model (usually 1) and chain
        for cand_model in st.get_models():
            for cand_chain in cand_model:
                # build candidate sequence
                resnames = [res.get_resname() for res in cand_chain.get_residues() if is_poly_peptide_residue(res)]
                if not resnames:
                    continue
                cand_seq = sequence_from_resnames(resnames)

                # global align map base->candidate
                map_idx = global_align_map(base_sequence, cand_seq, match=2, mismatch=-1, gap_open=-5, gap_extend=-1)
                identity, coverage = compute_identity_coverage(base_sequence, cand_seq, map_idx)
                if identity < identity_thresh or coverage < coverage_thresh:
                    continue

                # build N/CA/C arrays and mask in base L-order for FIRST model only
                N_c, CA_c, C_c, m_c = build_coords_from_mapping(cand_model, cand_chain.id, ref_keys, map_idx)
                if m_c.sum() < 8:
                    continue

                # align candidate CA to base medoid using core residues
                common = (m_c.astype(bool) & core_mask)
                if common.sum() < 8:
                    continue
                R, t = kabsch(CA_c[common], coords_ca_aligned[medoid_idx, common])
                CA_c_aln = (CA_c @ R.T) + t

                # collect metadata/state
                ligs = meta_base.get("ligands", "")
                state = "apo" if ligs == "" else f"holo-{ligs}"
                meta_entry = {
                    "pdb_id": cand_pdb.lower(),
                    "chain_id": cand_chain.id,
                    "method": meta_base.get("method", ""),
                    "resolution": float(meta_base.get("resolution", np.nan)) if meta_base.get("resolution", "") != "" else np.nan,
                    "pH": float(meta_base.get("pH", np.nan)) if meta_base.get("pH","") != "" else np.nan,
                    "temperature_K": float(meta_base.get("temperature_K", np.nan)) if meta_base.get("temperature_K","") != "" else np.nan,
                    "ligands": ligs,
                    "state": state,
                    "identity": float(identity),
                    "coverage": float(coverage),
                }

                coords_list.append(CA_c_aln.astype(np.float32))
                mask_list.append(m_c.astype(np.uint8))
                meta_list.append(json.dumps(meta_entry))
                added += 1

                if added >= max_models:
                    break
            if added >= max_models:
                break
        if added >= max_models:
            break

    if added == 0:
        return 0

    # Write to /crosspdb in the same H5
    with h5py.File(h5_path, "a") as f:
        g = f.require_group("crosspdb")
        g.create_dataset("coords_ca", data=np.stack(coords_list, axis=0), compression="gzip")  # [K2,L,3]
        g.create_dataset("mask_ca",   data=np.stack(mask_list,  axis=0), compression="gzip")  # [K2,L]
        dt = h5py.string_dtype(encoding='utf-8')
        g.create_dataset("meta_json", data=np.array(meta_list, dtype=dt))
    return added


# ---------------------------
# Per-entry processing
# ---------------------------

def process_entry(pdb_id: str,
                  raw_dir: str,
                  proc_dir: str,
                  min_models: int,
                  min_len: int,
                  max_len: int,
                  max_missing_frac: float) -> List[dict]:
    """
    Process one PDB entry (NMR multi-model). Returns a list of manifest rows, one per chain saved.
    Writes:
      - coords_{ca,N,C}, mask_ca, core_mask
      - rmsf_ca, rmsf_core_ca
      - torsion_{phi,psi,omega}_sincos
      - pair_medoid/{d_ca, omega, theta, phi}
      - crosspdb/{coords_ca, mask_ca, meta_json}  (same-UniProt conformers; best-effort)
    """
    out_rows: List[dict] = []

    cif_path = download_mmcif(pdb_id, raw_dir, retries=3, backoff=1.5)
    if cif_path is None:
        return out_rows

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id.lower(), cif_path)
    except Exception:
        return out_rows

    models = list(structure.get_models())
    if len(models) < min_models:
        return out_rows

    model0 = models[0]
    chain_ids = [c.id for c in model0 if chain_is_polypeptide(c)]
    if not chain_ids:
        return out_rows

    os.makedirs(proc_dir, exist_ok=True)

    for chain_id in chain_ids:
        # Build reference indexing from model 0
        ref_keys, resnames = build_reference_residue_list(model0, chain_id)
        L = len(ref_keys)
        
        # Debug: Show first few residue names from CIF
        if len(resnames) > 0:
            print(f"  Chain {chain_id}: {L} residues, first 10 resnames: {resnames[:10]}")
        
        if L < min_len or L > max_len:
            continue

        # Extract backbone for all models
        N_all, CA_all, C_all, mask_all_backbone = extract_backbone_coords_for_chain(structure, chain_id, ref_keys)
        K, L, _ = CA_all.shape

        # Missingness check using CA presence
        ca_present = np.isfinite(CA_all).all(axis=2)  # [K,L]
        observed_cols = (ca_present.sum(axis=0) >= (K // 2 + 1)).sum()
        miss_frac = 1.0 - (observed_cols / float(L))
        if miss_frac > max_missing_frac:
            continue

        mask_ca = ca_present.astype(np.uint8)
        coords_ca = CA_all.copy()

        # Align to medoid with core-fit (CA only)
        coords_ca_aligned, medoid_idx, core_mask = align_core_fit(
            coords_ca, mask_ca, core_frac=0.7, min_core_len=30, min_common=8
        )

        # Carry N and C through the same rigid transforms (via CA fit)
        def align_backbone_to_reference(N_in, CA_in, C_in, mask_ca_in, ref_idx, use_mask):
            K, L, _ = CA_in.shape
            N_out = N_in.copy(); CA_out = CA_in.copy(); C_out = C_in.copy()
            ref = CA_in[ref_idx]
            ref_mask = mask_ca_in[ref_idx].astype(bool)
            fit_mask_ref = ref_mask & (use_mask if use_mask is not None else np.ones(L, dtype=bool))
            for k in range(K):
                common = fit_mask_ref & mask_ca_in[k].astype(bool)
                idx = np.where(common)[0]
                if idx.size >= 3:
                    P = CA_in[k, idx]; Q = ref[idx]
                    R, t = kabsch(P, Q)
                    N_out[k]  = (N_in[k]  @ R.T) + t
                    CA_out[k] = (CA_in[k] @ R.T) + t
                    C_out[k]  = (C_in[k]  @ R.T) + t
            return N_out, CA_out, C_out

        N_aligned, CA_aligned, C_aligned = align_backbone_to_reference(
            N_all, coords_ca, C_all, mask_ca, ref_idx=medoid_idx, use_mask=core_mask
        )
        coords_ca_aligned = CA_aligned  # keep naming consistent downstream

        # Flexibility (CA)
        rmsf = compute_rmsf_core(coords_ca_aligned, mask_ca, use_mask=None)
        rmsf_core = compute_rmsf_core(coords_ca_aligned, mask_ca, use_mask=core_mask)

        # Sequence
        sequence = sequence_from_resnames(resnames)
        
        # Verify sequence extraction was successful
        if sequence:
            x_count = sequence.count('X')
            if x_count > 0:
                print(f"  ⚠️  Sequence contains {x_count}/{len(sequence)} unknown residues (X)")
            else:
                print(f"  ✅ Sequence extracted successfully: {sequence[:60]}{'...' if len(sequence) > 60 else ''}")
        else:
            print(f"  ❌ ERROR: Failed to extract sequence!")

        # Per-residue torsions (ϕ/ψ/ω as sin/cos). Require N/C present as well.
        tors = compute_backbone_torsions(
            N_aligned, CA_aligned, C_aligned,
            mask_all=(mask_ca & (np.isfinite(N_aligned).all(axis=2)) & (np.isfinite(C_aligned).all(axis=2))).astype(np.uint8)
        )

        # Per-pair features (medoid only by default)
        pair = compute_pair_features_trrosetta(
            coords_ca_aligned, N_aligned, CA_aligned, C_aligned, mask_ca,
            models="medoid", medoid_idx=medoid_idx
        )
        # Optional: shrink file size
        # for k_ in ("d", "omega", "theta", "phi"):
        #     pair[k_] = pair[k_].astype(np.float16)

        # Persist NMR ensemble + features
        h5_name = f"{pdb_id.lower()}_{chain_id}_ensemble.h5"
        h5_path = os.path.join(proc_dir, h5_name)
        with h5py.File(h5_path, "w") as f:
            # coordinates & masks
            f.create_dataset("coords_ca", data=coords_ca_aligned, compression="gzip")  # [K,L,3]
            f.create_dataset("coords_N",  data=N_aligned,        compression="gzip")   # [K,L,3]
            f.create_dataset("coords_C",  data=C_aligned,        compression="gzip")   # [K,L,3]
            f.create_dataset("mask_ca",   data=mask_ca,          compression="gzip")   # [K,L]
            f.create_dataset("core_mask", data=core_mask.astype(np.uint8), compression="gzip")

            # flexibility
            f.create_dataset("rmsf_ca",       data=rmsf,       compression="gzip")  # [L]
            f.create_dataset("rmsf_core_ca",  data=rmsf_core,  compression="gzip")  # [L]

            # sequence / indexing
            f.create_dataset("sequence", data=np.string_(sequence))
            f.create_dataset("resseq",   data=np.array([rk[0] for rk in ref_keys], dtype=np.int32))
            f.create_dataset("icode",    data=np.array([(rk[1] or " ").encode("ascii") for rk in ref_keys], dtype="S1"))

            # per-residue torsions (sincos)
            f.create_dataset("torsion_phi_sincos",   data=tors["phi_sincos"],   compression="gzip")  # [K,L,2]
            f.create_dataset("torsion_psi_sincos",   data=tors["psi_sincos"],   compression="gzip")
            f.create_dataset("torsion_omega_sincos", data=tors["omega_sincos"], compression="gzip")

            # per-pair features (medoid)
            g_pair = f.create_group("pair_medoid")
            g_pair.create_dataset("d_ca",   data=pair["d"],     compression="gzip")  # [L,L]
            g_pair.create_dataset("omega",  data=pair["omega"], compression="gzip")
            g_pair.create_dataset("theta",  data=pair["theta"], compression="gzip")
            g_pair.create_dataset("phi",    data=pair["phi"],   compression="gzip")

            # attrs
            f.attrs["pdb_id"] = pdb_id.lower()
            f.attrs["chain_id"] = chain_id
            f.attrs["num_models"] = K
            f.attrs["num_residues"] = L
            f.attrs["method"] = "NMR"
            f.attrs["alignment_reference"] = f"medoid_model_index={medoid_idx}"
            f.attrs["coordinates"] = "N/CA/C, rigidly aligned to medoid with core-fit"

        # ----- Add cross-PDB conformers (same UniProt), best-effort -----
        try:
            _ = append_crosspdb_conformers(
                base_pdb_id=pdb_id.lower(),
                base_chain_id=chain_id,
                cif_path=cif_path,
                ref_keys=ref_keys,
                base_sequence=sequence,
                medoid_idx=medoid_idx,
                core_mask=core_mask,
                coords_ca_aligned=coords_ca_aligned,
                raw_dir=raw_dir,
                h5_path=h5_path,
                identity_thresh=0.95,
                coverage_thresh=0.90,
                max_models=200
            )
        except Exception:
            # cross-PDB is optional; ignore failures
            pass

        out_rows.append({
            "pdb_id": pdb_id.lower(),
            "chain_id": chain_id,
            "h5_path": os.path.abspath(h5_path),
            "num_models": K,
            "num_residues": L,
            "miss_frac": round(miss_frac, 4),
            "method": "NMR"
        })

    return out_rows


# ---------------------------
# Orchestrator (build dataset)
# ---------------------------

def build_dataset(output_dir: str,
                  target_chains: int,
                  min_models: int,
                  min_len: int,
                  max_len: int,
                  max_missing_frac: float,
                  max_entries_to_try: int,
                  shuffle_seed: int):
    """
    Build a dataset with at least `target_chains` chains.
    """
    random.seed(shuffle_seed)
    raw_dir = os.path.join(output_dir, "raw")
    proc_dir = os.path.join(output_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    # Query candidate entries
    print("Querying NMR entries from RCSB...")
    pdb_ids = query_nmr_entries(min_models=min_models, shuffle_seed=shuffle_seed)
    if not pdb_ids:
        raise RuntimeError("No NMR entries returned by RCSB query.")
    pdb_ids = pdb_ids[:max_entries_to_try]

    # Process until we hit target_chains
    total_rows: List[dict] = []
    pbar = tqdm(pdb_ids, desc="Processing entries")
    for pdb_id in pbar:
        rows = process_entry(
            pdb_id=pdb_id,
            raw_dir=raw_dir,
            proc_dir=proc_dir,
            min_models=min_models,
            min_len=min_len,
            max_len=max_len,
            max_missing_frac=max_missing_frac
        )
        if rows:
            total_rows.extend(rows)
            pbar.set_postfix_str(f"chains={len(total_rows)}")
        if len(total_rows) >= target_chains:
            break

    if len(total_rows) < target_chains:
        print(f"Only {len(total_rows)} chains collected; consider loosening filters or increasing max_entries_to_try.")

    # Write manifest
    def write_manifest(rows, path):
        with open(path, "w", newline="") as csvfile:
            fieldnames = ["pdb_id", "chain_id", "h5_path", "num_models", "num_residues", "miss_frac", "method"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    # 80% train, 10% val, 10% test
    train_rows, test_rows = train_test_split(total_rows, test_size=0.1, random_state=shuffle_seed)
    train_rows, val_rows = train_test_split(train_rows, test_size=0.1111, random_state=shuffle_seed)  # 0.1111 ≈ 10% of remaining

    # Save manifests
    write_manifest(train_rows, os.path.join(output_dir, "manifest_train.csv"))
    write_manifest(val_rows,   os.path.join(output_dir, "manifest_val.csv"))
    write_manifest(test_rows,  os.path.join(output_dir, "manifest_test.csv"))

    # Summary
    n_entries = len({r["pdb_id"] for r in total_rows})
    n_chains = len(total_rows)
    lengths = [r["num_residues"] for r in total_rows]
    ks = [r["num_models"] for r in total_rows]
    print(
        f"Done. Entries: {n_entries}, Chains: {n_chains}, "
        f"L median={int(np.median(lengths)) if lengths else 0}, "
        f"K median={int(np.median(ks)) if ks else 0}"
    )
    print(f"Train/Val/Test split: {len(train_rows)} / {len(val_rows)} / {len(test_rows)}")
    print(f"Data root: {os.path.abspath(output_dir)}")


def parse_args():
    ap = argparse.ArgumentParser(description="Build an NMR ensemble dataset (≥150 protein chains) for a Protein Ensemble VAE.")
    ap.add_argument("--output", type=str, default="protein_ensemble_dataset", help="Output directory")
    ap.add_argument("--target_chains", type=int, default=150, help="Minimum number of chains to collect")
    ap.add_argument("--min_models", type=int, default=20, help="Minimum number of models per entry (NMR)")
    ap.add_argument("--min_len", type=int, default=50, help="Minimum chain length to keep")
    ap.add_argument("--max_len", type=int, default=600, help="Maximum chain length to keep")
    ap.add_argument("--max_missing_frac", type=float, default=0.20, help="Max allowed fraction of residues missing in ≥ half models")
    ap.add_argument("--max_entries_to_try", type=int, default=2000, help="Max PDB entries to attempt before stopping")
    ap.add_argument("--shuffle_seed", type=int, default=13, help="Seed for shuffling PDB IDs")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        output_dir=args.output,
        target_chains=args.target_chains,
        min_models=args.min_models,
        min_len=args.min_len,
        max_len=args.max_len,
        max_missing_frac=args.max_missing_frac,
        max_entries_to_try=args.max_entries_to_try,
        shuffle_seed=args.shuffle_seed
    )