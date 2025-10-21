#!/usr/bin/env python3
"""
Visualize the train split of a Protein Ensemble VAE dataset.

What this script does
---------------------
- Loads a train manifest CSV (with at least columns: pdb_id, chain_id, h5_path or {pdb_id, chain_id}).
- Aggregates per-chain metrics by reading each HDF5 ensemble:
    * K (# of models), L (# of residues), missing fraction, mean/median/max RMSF.
- Creates plots:
    * Histogram of chain lengths (L)
    * Histogram of # models per chain (K)
    * Histogram of missing fraction
    * Histogram of mean RMSF
    * Scatter: L vs K
    * Scatter: L vs mean RMSF
    * Per-residue RMSF line plot for the "most flexible" chain
    * 3D overlay (matplotlib) of Cα backbones for the same chain
- Writes a summary CSV with one row per chain.
- Saves all artifacts in an output folder.

"""

import os
import argparse
import math
import h5py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt


def find_manifest(path_candidates):
    for p in path_candidates:
        if p and os.path.exists(p):
            return p
    return None


def infer_h5_path(row, manifest_path):
    """If 'h5_path' not present, infer relative to dataset root: <root>/processed/<pdb>_<chain>_ensemble.h5"""
    root = os.path.dirname(os.path.dirname(manifest_path))  # assume manifest at <root>/manifest_train.csv
    cand = os.path.join(root, "processed", f"{row['pdb_id']}_{row['chain_id']}_ensemble.h5")
    return cand


def robust_mean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.nanmean(x))


def compute_missing_fraction(mask):
    """
    Our definition: fraction of residues missing in at least half of the models.
    mask: [K,L]
    """
    K = mask.shape[0]
    observed_cols = (mask.sum(axis=0) >= (K // 2 + 1)).sum()
    L = mask.shape[1]
    miss_frac = 1.0 - (observed_cols / float(L))
    return float(miss_frac)


def compute_summary(manifest_path, outdir, limit=0):
    df = pd.read_csv(manifest_path)
    # Minimal columns check
    required_any = ({"h5_path"} <= set(df.columns)) or ({"pdb_id", "chain_id"} <= set(df.columns))
    if not required_any:
        raise ValueError("Manifest must contain either 'h5_path' or both 'pdb_id' and 'chain_id'.")
    if "h5_path" not in df.columns:
        df["h5_path"] = df.apply(lambda r: infer_h5_path(r, manifest_path), axis=1)

    rows = []
    processed = 0
    for _, r in df.iterrows():
        if limit and processed >= limit:
            break
        h5_path = r["h5_path"]
        if not isinstance(h5_path, str) or not os.path.exists(h5_path):
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                coords = f["coords_ca"][:]
                mask = f["mask_ca"][:].astype(bool)
                rmsf = f["rmsf_ca"][:] if "rmsf_ca" in f else None
                K, L = coords.shape[0], coords.shape[1]
                if rmsf is None or rmsf.shape[0] != L:
                    # compute RMSF from coords if not present
                    arr = coords.astype(np.float64).copy()
                    # nan-out missing
                    for k in range(K):
                        arr[k, ~mask[k]] = np.nan
                    mean = np.nanmean(arr, axis=0)             # [L,3]
                    diffs = arr - mean
                    sq = np.nansum(diffs**2, axis=2)           # [K,L]
                    rmsf = np.sqrt(np.nanmean(sq, axis=0))     # [L]
                    rmsf = np.nan_to_num(rmsf).astype(np.float32)

                mean_rmsf = float(np.nanmean(rmsf))
                median_rmsf = float(np.nanmedian(rmsf))
                max_rmsf = float(np.nanmax(rmsf))
                miss_frac = compute_missing_fraction(mask)

                rows.append({
                    "pdb_id": str(f.attrs.get("pdb_id", r.get("pdb_id", ""))),
                    "chain_id": str(f.attrs.get("chain_id", r.get("chain_id", ""))),
                    "h5_path": os.path.abspath(h5_path),
                    "num_models": int(f.attrs.get("num_models", K)),
                    "num_residues": int(f.attrs.get("num_residues", L)),
                    "miss_frac": float(r.get("miss_frac", miss_frac)),
                    "mean_rmsf": mean_rmsf,
                    "median_rmsf": median_rmsf,
                    "max_rmsf": max_rmsf,
                })
                processed += 1
        except Exception as e:
            # Skip problematic files but continue
            print(f"Warning: failed {h5_path}: {e}")
            continue

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(outdir, "train_summary.csv"), index=False)
    return out_df


def save_hist(series, title, xlabel, out_path, bins=30):
    plt.figure()
    plt.hist(series.dropna().values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_scatter(x, y, title, xlabel, ylabel, out_path):
    plt.figure()
    plt.scatter(x.values, y.values, s=10, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize train data for Protein Ensemble VAE")
    parser.add_argument("--manifest", type=str, default="protein_ensemble_dataset/manifest_train.csv", help="Path to manifest_train.csv")
    parser.add_argument("--outdir", type=str, default="train_viz", help="Directory to save plots and summary")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of chains processed (0 = no limit)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    manifest = find_manifest([args.manifest, os.path.join(os.getcwd(), "manifest_train.csv")])
    if manifest is None:
        print("Manifest not found. Pass --manifest /path/to/manifest_train.csv")
        return

    summary = compute_summary(manifest, args.outdir, limit=args.limit)

    if summary.empty:
        print("No valid chains found in manifest; nothing to plot.")
        return

    # Save plots
    save_hist(summary["num_residues"], "Chain length distribution", "Residues (L)", os.path.join(args.outdir, "hist_length.png"))
    save_hist(summary["num_models"], "Ensemble size distribution", "Models per chain (K)", os.path.join(args.outdir, "hist_models.png"))
    save_hist(summary["miss_frac"], "Missing fraction distribution", "Missing fraction (≥ half models)", os.path.join(args.outdir, "hist_missing_frac.png"))
    save_hist(summary["mean_rmsf"], "Mean RMSF distribution", "Mean RMSF (Å)", os.path.join(args.outdir, "hist_mean_rmsf.png"))

    save_scatter(summary["num_residues"], summary["num_models"],
                 "L vs K", "Residues (L)", "Models per chain (K)",
                 os.path.join(args.outdir, "scatter_L_vs_K.png"))

    save_scatter(summary["num_residues"], summary["mean_rmsf"],
                 "L vs mean RMSF", "Residues (L)", "Mean RMSF (Å)",
                 os.path.join(args.outdir, "scatter_L_vs_meanRMSF.png"))

    # Chain with highest mean RMSF for detailed view
    top = summary.sort_values("mean_rmsf", ascending=False).iloc[0]            

    # Write a small text summary
    text = []
    text.append(f"Chains: {len(summary)}")
    text.append(f"L (residues): median={int(np.median(summary['num_residues']))}, min={int(summary['num_residues'].min())}, max={int(summary['num_residues'].max())}")
    text.append(f"K (models): median={int(np.median(summary['num_models']))}, min={int(summary['num_models'].min())}, max={int(summary['num_models'].max())}")
    text.append(f"Missing fraction: median={summary['miss_frac'].median():.3f}")
    text.append(f"Mean RMSF (Å): median={summary['mean_rmsf'].median():.2f}")
    with open(os.path.join(args.outdir, "summary.txt"), "w") as fout:
        fout.write("\n".join(text))

    print("\n".join(text))
    print(f"Artifacts saved under: {os.path.abspath(args.outdir)}")
    print("Files:")
    for fn in ["train_summary.csv",
               "hist_length.png","hist_models.png","hist_missing_frac.png","hist_mean_rmsf.png",
               "scatter_L_vs_K.png","scatter_L_vs_meanRMSF.png",
               "rmsf_line_top.png","summary.txt"]:
        fp = os.path.join(args.outdir, fn)
        print(" -", fp)


if __name__ == "__main__":
    main()
