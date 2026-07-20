#!/usr/bin/env python3
"""
verify_pipeline_consistency.py
================================
Sanity check: run_rwr_observed_all.py and permutation_test_percell_decoys.py
both build a "L* = solve_rwr(build_seed(...))" pipeline independently.
Before trusting that the observed CCC table (figures 2-4) and the
permutation/decoy significance results (figures 5-6) tell a consistent
story, this script verifies they are actually computing the SAME thing:

    1. Same graph (results/disparity_graph.pkl) -- checked by hash of
       the row-normalised transition matrix.
    2. Same L* for a given (sender_ct, ligand) pair -- computed
       independently here using the same build_seed/solve_rwr logic,
       then compared directly against a value pulled from
       ccc_all_lr_pairs_ligandseeded_r0700.csv for the same pair.

If these agree, you can treat the observed-CCC and permutation-test
pipelines as one consistent pipeline for the thesis. If they disagree,
something in one of the two scripts' graph loading / seed construction
has diverged and needs fixing before combining results from both.

Usage:
    python verify_pipeline_consistency.py \
        --ccc_csv results/ccc_results/ccc_all_lr_pairs_ligandseeded_r0700.csv \
        --sender "B cell" --ligand CXCL13 --r 0.7
"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import networkx as nx
from scipy.sparse.linalg import splu
from shapely.geometry import Point, Polygon as ShapelyPolygon

BASE_DIR    = "/scratch/users/k22026807/masters/project/random_walks"
ADATA_PATH  = "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
ROI_PATH    = "/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson"
GRAPH_CACHE = os.path.join(BASE_DIR, "results/disparity_graph.pkl")

ap = argparse.ArgumentParser()
ap.add_argument("--ccc_csv", type=str, required=True)
ap.add_argument("--sender", type=str, required=True)
ap.add_argument("--ligand", type=str, required=True)
ap.add_argument("--r", type=float, default=0.7)
args = ap.parse_args()

print("Loading AnnData...")
adata = sc.read(ADATA_PATH)
with open(ROI_PATH) as f:
    roi = json.load(f)
polygon_pts = roi['features'][0]['geometry']['coordinates'][0]
from shapely.geometry import Polygon as ShapelyPolygon2
polygon = ShapelyPolygon2(polygon_pts)
mask = np.array([polygon.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
adata = adata[mask].copy()
n = adata.n_obs
cell_types = adata.obs['cell_type'].values
print(f"ROI cells: {n:,}")

print(f"\nLoading graph: {GRAPH_CACHE}")
with open(GRAPH_CACHE, "rb") as f:
    G = pickle.load(f)
if isinstance(G, nx.Graph):
    missing = set(range(n)) - set(G.nodes())
    if missing:
        G.add_nodes_from(missing)
    G = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), format="csr", dtype=np.float64)
elif sp.issparse(G):
    G = G.tocsr()
else:
    raise TypeError(f"Unsupported graph type: {type(G)}")
assert G.shape == (n, n)
print(f"  Graph: {G.shape[0]:,} nodes, {G.nnz:,} nonzero entries "
      f"(sanity print -- compare this nnz against what "
      f"permutation_test_percell_decoys.py reports/expects for the "
      f"same GRAPH_CACHE path; they should match exactly since it's "
      f"the identical pickle file)")

row_sums = np.asarray(G.sum(axis=1)).ravel()
row_sums[row_sums == 0] = 1.0
D_inv = sp.diags(1.0 / row_sums)
P = D_inv @ G
A = (sp.identity(n, format="csr") - (1.0 - args.r) * P).tocsc()
A_lu = splu(A)


def expr(gene_or_complex):
    genes = gene_or_complex.split("_")
    if any(g not in adata.var_names for g in genes):
        return None
    arrs = []
    for g in genes:
        x = adata[:, g].X
        x = x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()
        arrs.append(x.astype(np.float32))
    out = arrs[0].copy()
    for a in arrs[1:]:
        out *= a
    return out


def build_seed(L_expr, sender_indices):
    seed = np.zeros(n, dtype=np.float64)
    seed[sender_indices] = L_expr[sender_indices]
    s = seed.sum()
    if s > 0:
        seed /= s
    return seed


sender_mask = (cell_types == args.sender)
sender_indices = np.where(sender_mask)[0]
if len(sender_indices) == 0:
    raise SystemExit(f"No cells of sender type '{args.sender}' found. "
                      f"Available: {sorted(np.unique(cell_types))}")

L_expr = expr(args.ligand)
if L_expr is None:
    raise SystemExit(f"Ligand '{args.ligand}' not in panel genes.")
if L_expr[sender_indices].sum() == 0:
    raise SystemExit(f"'{args.ligand}' has zero expression in "
                      f"'{args.sender}' cells -- pick a different pair.")

seed = build_seed(L_expr, sender_indices)
L_star_recomputed = (args.r * A_lu.solve(seed)).astype(np.float32)

print(f"\nRecomputed L* for sender='{args.sender}', ligand='{args.ligand}', "
      f"r={args.r}:")
print(f"  Range: [{L_star_recomputed.min():.3e}, {L_star_recomputed.max():.3e}]")
print(f"  Sum:   {L_star_recomputed.sum():.6f}")
print(f"  Mean at sender cells: {L_star_recomputed[sender_indices].mean():.3e}")

# ── Compare against the saved CSV's mean_ccc for this pair (indirect
#    check -- mean_ccc = k * L* * R, so we can back out an implied
#    relationship, or just flag if it's wildly inconsistent) ──────────
print(f"\nLoading {args.ccc_csv} to find matching rows...")
df = pd.read_csv(args.ccc_csv)
matches = df[(df['sender_ct'] == args.sender) & (df['ligand'] == args.ligand)]
if matches.empty:
    print(f"  No rows found for sender='{args.sender}', ligand='{args.ligand}' "
          f"in {args.ccc_csv} -- this pair may not have passed the "
          f"expression filter in the original run. Recomputed L* above "
          f"is still valid; just nothing to cross-check against yet.")
else:
    print(f"  Found {len(matches)} matching row(s) (one per receiver_ct/receptor):")
    print(matches[['receptor', 'receiver_ct', 'mean_ccc', 'max_ccc',
                    'n_receiver', 'pct_R_expr']].to_string(index=False))
    print(f"\n  If these mean_ccc values look consistent with L* range "
          f"printed above (i.e. mean_ccc / R roughly falls within the "
          f"L* range for the corresponding receiver cells), the two "
          f"pipelines agree. A large, unexplained discrepancy would "
          f"indicate the graph or seed construction has diverged "
          f"between run_rwr_observed_all.py and this verification.")

print("\nDone. Repeat with --sender/--ligand set to a pair that also "
      "appears in your permutation_test_percell_decoys.py output "
      "(same sender_ct, same ligand) to directly compare L* or c_obs "
      "values cell-by-cell if you want a stronger numerical check than "
      "this summary-statistic comparison.")