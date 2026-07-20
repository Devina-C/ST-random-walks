#!/usr/bin/env python3
"""
permutation_test_perspot_null.py
==========================================
Spot-level RWR significance test using SpaFlow's own null procedure
instead of the SBM null: permute spot coordinates, rebuild the
(disparity-filtered) graph from the permuted coordinates, rerun the
diffusion model, repeat B times per pathway/LR-pair test.

WHY THIS NULL, AND WHAT IT TRADES AWAY
-----------------------------------------
This is a deliberate methodological choice, not a fix for the false-
positive inflation diagnosed earlier. Global coordinate permutation is
exactly the null fit_sbm_null.py's docstring identifies as the cause of
the ~86% inflated significance rate on the single-cell graph -- using it
here will very likely reproduce that inflation at spot resolution too.
The point of this script is a MATCHED comparison: SpaFlow's own
Methods section uses this identical null (permute spot coordinates,
rebuild the Laplacian, rerun). Testing your RWR under the same null
isolates whether any difference in significance rate between the two
methods comes from the diffusion model itself, rather than from which
null each one happens to use. Report this alongside (not instead of)
your SBM-null results -- this is the "fair vs prior work" number, the
SBM version is the "calibrated" number.

NO SENDER-CELL-TYPE RESTRICTION
---------------------------------
Unlike permutation_test_perspot_sbm.py (which loops over --sender_idx
and restricts the seed to one cell type's spots), this script builds
the seed from ligand expression across ALL spots, matching SpaFlow's
own initial_concentration() -- every spot's ligand concentration is
seeded from its own expression, with no sender-type gating. This is
the base per-spot CCC score, matching what SpaFlow actually reports
(spatial maps of complex concentration per LR pair). If you want the
sender-attribution analysis (which cell type is driving signal at a
given receiver), run permutation_test_perspot_sbm.py's sender-restricted
version separately -- that's a different, additional analysis, not a
substitute for this base test.

EFFICIENCY NOTE
------------------
The null graph depends only on permuted COORDINATES, not on expression,
so for a fixed permutation round the resulting graph (and its LU
factorization) is identical across every LR pair. B null graphs are
therefore built and factorised ONCE per chunk, then reused (multiple
right-hand-side solves against the same factorisation) across all LR
pairs in the chunk -- same efficient structure as the SBM script, just
generating the null graphs on the fly instead of loading cached ones.
Benchmarked at ~130ms/round (build Delaunay + disparity filter +
factorise) at n=2900 spots -- B=1000 costs about 2 minutes of graph
setup, reused for every LR pair tested afterward. This is substantially
cheaper than SpaFlow's own null generation, which reruns a full
500-iteration ODE solve per permutation round per pathway.

Usage:
    python permutation_test_perspot_spaflow_null.py --chunk 0 --total 16 \
        --r 0.7 --B 1000 --disparity_alpha 0.3
"""

import argparse, os, time
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import networkx as nx
from scipy.spatial import Delaunay
from scipy.sparse.linalg import splu

BASE_DIR    = "/scratch/users/k22026807/masters/project/random_walks"
SPOT_ADATA  = os.path.join(BASE_DIR, "results/pseudo_visium_adata.h5ad")
#LR_PATH     = os.path.join(BASE_DIR, "data/cellchat_full.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results/permutation_perspot_null")

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, required=True)
ap.add_argument("--total", type=int, default=16)
ap.add_argument("--r", type=float, default=0.7,
                 help="RWR restart probability (SpaFlow has no direct "
                      "analogue -- this is your own alpha).")
ap.add_argument("--B", type=int, default=1000,
                 help="Number of coordinate-permutation rounds.")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--disparity_alpha", type=float, default=0.3,
                 help="Disparity filter significance threshold, reused "
                      "identically for the observed graph and every "
                      "permuted null graph. Match whatever value you "
                      "settled on for pseudo_visium_disparity_graph.pkl "
                      "after checking its isolation rate.")
ap.add_argument("--k", type=int, default=6,
                 help="k for the spot-level KNN graph (SpaFlow default is 6).")
ap.add_argument("--lr_file", type=str, default=None,
                 help="Path to a fixed real+decoy pair CSV from "
                      "generate_decoy_pairs.py (columns: ligand, receptor, "
                      "pair_type). If not given, falls back to reading "
                      "LR_PATH as real-pairs-only (pair_type='real').")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)

R_TAG = f"r{int(args.r*1000):04d}"
OUT_DIR = os.path.join(RESULTS_DIR, R_TAG)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[{time.strftime('%H:%M:%S')}] Loading pseudo-Visium spot data...")
adata = sc.read_h5ad(SPOT_ADATA)
sc.pp.normalize_total(adata)
n = adata.n_obs
positions = adata.obsm["spatial"]
spot_ids = adata.obs.index.values
print(f"Spots: {n:,}")

# lr_df = pd.read_csv(LR_PATH, header=None,
#                     names=["ligand", "receptor", "pathway", "category"])
# lr_pairs = lr_df[["ligand", "receptor"]].drop_duplicates().reset_index(drop=True)

gene_set = set(adata.var_names)
def genes_present(complex_name):
    return all(g in gene_set for g in complex_name.split("_"))

if args.lr_file:
    lr_pairs = pd.read_csv(args.lr_file)
    print(f"Loaded fixed pair list: {args.lr_file}")
else:
    lr_df = pd.read_csv(LR_PATH, header=None, skiprows=1,
                        names=["ligand", "receptor", "pathway", "category"])
    lr_pairs = lr_df[["ligand", "receptor"]].drop_duplicates().reset_index(drop=True)
    lr_pairs["pair_type"] = "real"

_ok = lr_pairs["ligand"].apply(genes_present) & lr_pairs["receptor"].apply(genes_present)
if (~_ok).any():
    print(f"  Dropping {(~_ok).sum()} pair(s) referencing genes not in this "
          f"adata's panel (expected if this run's gene filtering differs "
          f"slightly from generate_decoy_pairs.py's)")
lr_pairs = lr_pairs[_ok].reset_index(drop=True)
print(f"LR pairs after gene filtering: {len(lr_pairs)}")

chunk_size = int(np.ceil(len(lr_pairs) / args.total))
start = args.chunk * chunk_size
end = min(start + chunk_size, len(lr_pairs))
lr_chunk = lr_pairs.iloc[start:end].reset_index(drop=True)
print(f"Chunk {args.chunk}/{args.total}: LR pairs [{start}:{end}]")


# ── disparity filter (same as build_pseudo_visium_disparity_graph.py) ─
def disparity_filter(G, alpha):
    backbone = nx.Graph()
    backbone.add_nodes_from(G.nodes())
    for i in G.nodes():
        neighbours = list(G[i])
        k_i = len(neighbours)
        if k_i == 0:
            continue
        if k_i == 1:
            j = neighbours[0]
            backbone.add_edge(i, j, weight=G[i][j]["weight"])
            continue
        s_i = sum(G[i][jj]["weight"] for jj in neighbours)
        for j in neighbours:
            w_ij = G[i][j]["weight"]
            p_ij = w_ij / s_i
            alpha_ij = (1.0 - p_ij) ** (k_i - 1)
            if alpha_ij < alpha:
                backbone.add_edge(i, j, weight=w_ij)
    return backbone


# def build_disparity_graph(pos, alpha):
#     tri = Delaunay(pos)
#     edges = set()
#     for simplex in tri.simplices:
#         for a in range(3):
#             for b in range(a + 1, 3):
#                 i, j = simplex[a], simplex[b]
#                 edges.add((min(i, j), max(i, j)))
#     G_cand = nx.Graph()
#     G_cand.add_nodes_from(range(len(pos)))
#     for i, j in edges:
#         d = np.linalg.norm(pos[i] - pos[j])
#         if d > 0:
#             G_cand.add_edge(i, j, weight=1.0 / d)
#     return disparity_filter(G_cand, alpha)


from sklearn.neighbors import kneighbors_graph

def build_knn_graph(pos, k):
    D = kneighbors_graph(pos, n_neighbors=min(k, len(pos) - 1),
                          mode="distance", include_self=False, n_jobs=-1)
    D.data = np.where(D.data > 0, 1.0 / D.data, 0.0)
    W = D.maximum(D.T).tocsr()
    W.eliminate_zeros()
    return nx.from_scipy_sparse_array(W, edge_attribute="weight")

def graph_to_factorised_rwr(G, n, restart_r):
    A = nx.to_scipy_sparse_array(G, nodelist=list(range(n)),
                                  format="csr", dtype=np.float64)
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv @ A
    M = (sp.identity(n, format="csr") - (1.0 - restart_r) * P).tocsc()
    return splu(M)


def solve_rwr(lu, seed_vec, restart_r):
    return (restart_r * lu.solve(seed_vec)).astype(np.float32)


# ── observed graph (real coordinates, unpermuted) ──────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Building observed disparity graph...")
t0 = time.time()
# G_obs = build_disparity_graph(positions, args.disparity_alpha)
G_obs = build_knn_graph(positions, args.k)
isolated_obs = sum(1 for i in G_obs.nodes() if G_obs.degree(i) == 0)
print(f"  Nodes: {G_obs.number_of_nodes():,}  Edges: {G_obs.number_of_edges():,}  "
      f"Isolated: {isolated_obs} ({100*isolated_obs/n:.1f}%)  "
      f"[{time.time()-t0:.1f}s]")
A_obs_lu = graph_to_factorised_rwr(G_obs, n, args.r)


# ── SpaFlow-style null: permute coordinates, rebuild graph, refactorise
#    -- B rounds, done ONCE per chunk, reused for every LR pair ────────
print(f"[{time.strftime('%H:%M:%S')}] Building {args.B} coordinate-"
      f"permuted null graphs (shared across all LR pairs in this chunk)...")
t0 = time.time()
null_lus = []
isolated_counts = []
for b in range(args.B):
    perm = rng.permutation(n)
    pos_perm = positions[perm]  # row i now sits where row perm[i] used to be
    # G_b = build_disparity_graph(pos_perm, args.disparity_alpha)
    G_b = build_knn_graph(pos_perm, args.k)
    isolated_counts.append(sum(1 for i in G_b.nodes() if G_b.degree(i) == 0))
    null_lus.append(graph_to_factorised_rwr(G_b, n, args.r))
    if (b + 1) % 100 == 0:
        elapsed = time.time() - t0
        rate = (b + 1) / elapsed
        eta = (args.B - b - 1) / rate
        print(f"  [{b+1}/{args.B}] {rate:.1f} rounds/s, ETA {eta:.0f}s")

print(f"  Null graph setup done in {(time.time()-t0)/60:.1f} min. "
      f"Mean isolated-node fraction across null draws: "
      f"{100*np.mean(isolated_counts)/n:.1f}% (compare to observed "
      f"{100*isolated_obs/n:.1f}% -- large systematic differences here "
      f"would indicate the null graphs are structurally unlike the "
      f"observed graph, which would bias the comparison independent of "
      f"any real LR signal).")


def expr(gene_or_complex):
    genes = gene_or_complex.split("_")
    arrs = []
    for g in genes:
        x = adata[:, g].X
        x = x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()
        arrs.append(x.astype(np.float32))
    if len(arrs) == 1:
        return arrs[0]
    out = arrs[0].copy()
    for arr in arrs[1:]:
        out *= arr
    return out


def build_seed(L_expr):
    # whole-tissue seed, no sender-type restriction -- matches SpaFlow's
    # initial_concentration(): every spot's own ligand expression seeds
    # the diffusion, normalised to a probability distribution for RWR.
    s = L_expr.sum()
    if s <= 0:
        return None
    return (L_expr / s).astype(np.float64)


rows = []
t_loop = time.time()

for lr_idx, row in lr_chunk.iterrows():
    lig = row["ligand"]
    rec = row["receptor"]

    L_expr = expr(lig)
    R_expr = expr(rec)

    if R_expr.max() == 0 or L_expr.sum() == 0:
        continue

    seed = build_seed(L_expr)
    if seed is None:
        continue

    L_star_obs = solve_rwr(A_obs_lu, seed, args.r)
    c_obs = L_star_obs * R_expr

    # rank-based right-tailed p-value, matching SpaFlow's
    # calculate_pvalues_rank_based exactly: p = 1 - rank/B, rank = number
    # of null draws <= observed at that spot (searchsorted on sorted null)
    null_scores = np.empty((n, args.B), dtype=np.float32)
    for b, lu_null in enumerate(null_lus):
        L_star_b = solve_rwr(lu_null, seed, args.r)
        null_scores[:, b] = L_star_b * R_expr

    p_value = np.empty(n, dtype=np.float32)
    null_mean = null_scores.mean(axis=1)
    null_std = null_scores.std(axis=1)
    for i in range(n):
        sorted_null = np.sort(null_scores[i, :])
        rank = np.searchsorted(sorted_null, c_obs[i], side="left")
        p_value[i] = 1.0 - rank / args.B

    with np.errstate(divide="ignore", invalid="ignore"):
        SES = np.where(null_mean > 0, c_obs / null_mean, np.nan)

    keep = c_obs > 0
    if keep.sum() == 0:
        continue
    idx = np.where(keep)[0]

    rows.append(pd.DataFrame({
        "ligand": lig,
        "receptor": rec,
        "spot_id": spot_ids[idx],
        "x": positions[idx, 0],
        "y": positions[idx, 1],
        "obs_ccc": c_obs[idx].astype(np.float32),
        "null_mean": null_mean[idx].astype(np.float32),
        "null_std": null_std[idx].astype(np.float32),
        "p_value": p_value[idx],
        "SES": SES[idx].astype(np.float32),
    }))

    if (lr_idx + 1) % 10 == 0:
        elapsed = time.time() - t_loop
        rate = (lr_idx + 1) / elapsed
        eta = (len(lr_chunk) - lr_idx - 1) / rate
        print(f"[{time.strftime('%H:%M:%S')}] LR {lr_idx+1}/{len(lr_chunk)} "
              f"| {rate:.3f} pairs/s | ETA {eta/60:.1f} min")

if not rows:
    print("No rows to write.")
    raise SystemExit(0)

df = pd.concat(rows, ignore_index=True)
out_path = os.path.join(OUT_DIR, f"perspot_spaflownull_{R_TAG}_chunk{args.chunk:02d}.parquet")
df.to_parquet(out_path, compression="snappy", index=False)

print(f"Saved: {out_path}")
print(f"Rows: {len(df):,}")
print(f"Total time: {(time.time() - t_loop)/60:.1f} min")