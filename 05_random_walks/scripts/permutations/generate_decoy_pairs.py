#!/usr/bin/env python3
"""
generate_decoy_pairs.py
=========================
Generates the real + decoy LR pair list ONCE and saves it to a fixed CSV,
so permutation_test_percell_sbm.py and permutation_test_perspot_spaflow_null.py
(and any other script) can load the identical pair list rather than each
regenerating decoys independently from a shared --decoy_seed.

WHY THIS MATTERS
------------------
Matching --decoy_seed across two separate scripts only guarantees identical
output if every upstream input is also identical: gene panel ordering
(sorted(gene_set)), which real pairs get filtered out (real_set), RNG
library version, etc. Single-cell and spot-level adata SHOULD have the same
var_names (aggregation doesn't drop genes), but relying on that silently
matching across two independently-run scripts is fragile and unverifiable
after the fact -- if it ever quietly diverges, you'd get two different
decoy sets while both scripts report the same seed, with no way to detect
it later. Generating once and loading from a file removes that whole
failure mode: both scripts test literally the same rows.

Usage:
    python generate_decoy_pairs.py --lr_mode decoy_random --n_background 3000
    python generate_decoy_pairs.py --lr_mode decoy_scramble
"""
import argparse, json, os, re, time
import numpy as np
import pandas as pd
import scanpy as sc
from shapely.geometry import Point, Polygon as ShapelyPolygon

BASE_DIR   = "/scratch/users/k22026807/masters/project/random_walks"
ADATA_PATH = "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
LR_PATH    = os.path.join(BASE_DIR, "data/cellchat_full.csv")
ROI_PATH   = "/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson"
OUT_DIR    = os.path.join(BASE_DIR, "data")

ap = argparse.ArgumentParser()
ap.add_argument("--lr_mode", choices=["decoy_scramble", "decoy_random"],
                 required=True)
ap.add_argument("--n_background", type=int, default=3000,
                 help="Number of random background pairs for decoy_random.")
ap.add_argument("--decoy_seed", type=int, default=2024)
ap.add_argument("--skiprows_lr", type=int, default=1,
                 help="Rows to skip in cellchat_full.csv when reading -- "
                      "set to 0 if the file has no header. RUN "
                      "`head -2 data/cellchat_full.csv` FIRST to check "
                      "this before trusting the default.")
args = ap.parse_args()

print(f"[{time.strftime('%H:%M:%S')}] Loading single-cell adata as the "
      f"canonical gene panel source...")
adata = sc.read_h5ad(ADATA_PATH)
with open(ROI_PATH) as f:
    roi = json.load(f)
poly = ShapelyPolygon(roi["features"][0]["geometry"]["coordinates"][0])
mask = np.array([poly.contains(Point(x, y)) for x, y in adata.obsm["spatial"]])
adata = adata[mask].copy()

PROBE_RX = re.compile(
    r'^(Human|Mouse|BLANK|NegControl|NegControlProbe|NegControlCodeword'
    r'|antisense|UnassignedCodeword|DeprecatedCodeword|Intergenic)',
    re.IGNORECASE)
keep_genes = [g for g in adata.var_names if not PROBE_RX.match(g)]
n_drop = adata.n_vars - len(keep_genes)
adata = adata[:, keep_genes].copy()
print(f"  Dropped {n_drop} control/mutation probes; {adata.n_vars} genes remain")

gene_set = set(adata.var_names)

def genes_present(complex_name):
    return all(g in gene_set for g in complex_name.split("_"))

lr_df = pd.read_csv(LR_PATH, header=None, skiprows=args.skiprows_lr,
                    names=["ligand", "receptor", "pathway", "category"])
real = lr_df[["ligand", "receptor"]].drop_duplicates().reset_index(drop=True)
real = real[real["ligand"].apply(genes_present)
            & real["receptor"].apply(genes_present)].reset_index(drop=True)
real_set = set(map(tuple, real.values))
print(f"  Real CellChat pairs (gene-filtered): {len(real)}")

rng_d = np.random.default_rng(args.decoy_seed)
real2 = real.copy()
real2["pair_type"] = "real"

if args.lr_mode == "decoy_scramble":
    rec_shuffled = rng_d.permutation(real["receptor"].values)
    decoy = pd.DataFrame({"ligand": real["ligand"].values,
                          "receptor": rec_shuffled})
    decoy = decoy[~decoy.apply(lambda r: (r.ligand, r.receptor) in real_set, axis=1)]
    decoy = decoy[decoy["ligand"] != decoy["receptor"]].drop_duplicates()
    decoy["pair_type"] = "decoy"
    print(f"  Scramble decoys generated: {len(decoy)} "
          f"(same ligand/receptor gene sets, re-paired)")

elif args.lr_mode == "decoy_random":
    genes = sorted(gene_set)
    decoys = set()
    while len(decoys) < args.n_background:
        l, r = rng_d.choice(genes), rng_d.choice(genes)
        if l != r and (l, r) not in real_set:
            decoys.add((l, r))
    decoy = pd.DataFrame(list(decoys), columns=["ligand", "receptor"])
    decoy["pair_type"] = "decoy"
    print(f"  Random background decoys generated: {len(decoy)} "
          f"(unrestricted gene-gene pairs from the full panel)")

lr_pairs = pd.concat([real2, decoy], ignore_index=True)

out_path = os.path.join(OUT_DIR, f"lr_pairs_{args.lr_mode}_seed{args.decoy_seed}.csv")
lr_pairs.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
print(f"  Total pairs: {len(lr_pairs)} "
      f"(real={ (lr_pairs['pair_type']=='real').sum() }, "
      f"decoy={ (lr_pairs['pair_type']=='decoy').sum() })")
print(f"\nGene panel fingerprint (for cross-checking against the spot-level "
      f"adata's var_names before running the spot-level script):")
print(f"  n_genes={len(gene_set)}, "
      f"first 5 sorted: {sorted(gene_set)[:5]}, "
      f"last 5 sorted: {sorted(gene_set)[-5:]}")
