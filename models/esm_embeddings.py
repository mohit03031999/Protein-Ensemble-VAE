#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import hashlib

import h5py
import torch
import esm


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load_esm_model(model_name: str = "esm2_t33_650M_UR50D", device: str = None):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, alphabet, device


@torch.no_grad()
def get_esm_embeddings(sequence: str, model, alphabet, device: str, layer: int = 33) -> torch.Tensor:
    """
    Returns per-residue embeddings of shape [L, D] (on CPU tensor).
    Uses final layer (33) by default for esm2_t33_650M_UR50D.
    """
    if not sequence or not isinstance(sequence, str):
        raise ValueError("Empty or invalid sequence.")

    # ESM2 token limit (incl. special tokens). t33 650M supports ~1024 total.
    # Your dataset caps L<=600, so this is safe. Check defensively anyway.
    if len(sequence) > 1022:
        raise ValueError(f"Sequence length {len(sequence)} exceeds ESM2 token limit (1022).")

    batch_converter = alphabet.get_batch_converter()
    data = [("chain", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[layer], return_contacts=False)
    reps = out["representations"][layer]  # [B=1, T, D]
    # Strip special tokens: [CLS] seq [EOS]
    residue_reps = reps[0, 1:1+len(sequence)]  # [L, D]
    return residue_reps.cpu()


def read_manifest_paths(*manifest_csvs):
    """Read unique h5 paths from one or more manifest CSVs (expects a column 'h5_path')."""
    h5_paths = []
    for m in manifest_csvs:
        if m is None:
            continue
        m = Path(m)
        if not m.exists():
            print(f"[WARN] Manifest not found: {m}")
            continue
        with m.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("h5_path")
                if p:
                    h5_paths.append(p)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in h5_paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def add_embeddings_to_h5(h5_path: Path, model, alphabet, device: str,
                         model_name: str, layer: int = 33,
                         dtype: str = "float32", overwrite: bool = False) -> bool:
    """
    Add ESM embeddings into the given H5 file.
    Returns True if written, False if skipped.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"[SKIP] Missing H5: {h5_path}")
        return False

    # dtype handling
    if dtype not in {"float16", "float32"}:
        raise ValueError("dtype must be 'float16' or 'float32'")
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    with h5py.File(h5_path, "a") as f:
        # get sequence (bytes → str)
        if "sequence" not in f:
            print(f"[SKIP] No 'sequence' dataset in {h5_path}")
            return False
        seq_raw = f["sequence"][()]
        sequence = seq_raw.decode("utf-8") if isinstance(seq_raw, (bytes, bytearray)) else str(seq_raw)
        L = len(sequence)

        # optional sanity vs stored L
        L_attr = int(f.attrs.get("num_residues", L))
        if L_attr != L:
            print(f"[WARN] num_residues attr ({L_attr}) != len(sequence) ({L}) in {h5_path.name}. Continuing.")

        # where we store embeddings
        grp_root = f.require_group("seq_embed")
        grp_model = grp_root.require_group(model_name)
        dset_name = f"layer_{layer}"

        # skip or overwrite
        if dset_name in grp_model and not overwrite:
            print(f"[SKIP] {h5_path.name}: embeddings already present at seq_embed/{model_name}/{dset_name}")
            return False
        if dset_name in grp_model and overwrite:
            del grp_model[dset_name]

        # compute embeddings
        emb = get_esm_embeddings(sequence, model, alphabet, device=device, layer=layer)  # [L, D]
        D = emb.shape[-1]

        # write with compression; cast dtype if requested
        emb_np = emb.to(torch_dtype).cpu().numpy()
        dset = grp_model.create_dataset(
            dset_name, data=emb_np, compression="gzip"
        )

        # attach metadata
        grp_model.attrs["model_name"] = model_name
        grp_model.attrs["layer"] = layer
        grp_model.attrs["embedding_dim"] = D
        grp_model.attrs["sequence_md5"] = md5(sequence)
        grp_model.attrs["device_used"] = device
        grp_model.attrs["dtype"] = dtype

        print(f"[OK] {h5_path.name}: wrote seq_embed/{model_name}/{dset_name} with shape [{L}, {D}] (dtype {dtype})")
        return True


def main():
    ap = argparse.ArgumentParser(description="Add ESM2 embeddings to processed H5 chain files.")
    ap.add_argument("--manifest_train", type=str, default=None)
    ap.add_argument("--manifest_val", type=str, default=None)
    ap.add_argument("--manifest_test", type=str, default=None)
    ap.add_argument("--model_name", type=str, default="esm2_t33_650M_UR50D")
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"],
                    help="Storage dtype for embeddings in H5.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing embeddings if present.")
    args = ap.parse_args()

    # collect files
    h5_paths = read_manifest_paths(args.manifest_train, args.manifest_val, args.manifest_test)
    if not h5_paths:
        print("[ERROR] No H5 paths found. Provide one or more manifest CSVs with a 'h5_path' column.")
        return

    # load model once
    model, alphabet, device = load_esm_model(args.model_name)
    print(f"[INFO] Loaded {args.model_name} on {device}")

    # process
    written = skipped = 0
    for p in h5_paths:
        ok = add_embeddings_to_h5(
            p, model, alphabet, device,
            model_name=args.model_name, layer=args.layer,
            dtype=args.dtype, overwrite=args.overwrite
        )
        if ok:
            written += 1
        else:
            skipped += 1
    print(f"[DONE] wrote={written}, skipped={skipped}")


if __name__ == "__main__":
    main()
