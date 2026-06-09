#!/usr/bin/env python3
"""
merge_and_BH_correction.py
==========================
Merge chunk outputs from permutation_test.py at a given r value,
estimate lambda per sender from the RWR L* field, and apply BH-FDR
correction with effective sample size n_eff.

Reads RWR_R from the top of the file. Tag-matched to permutation_test.py
output files via R_TAG.
"""
import os, glob, json, pickle, time
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import splu
from shapely.geometry import Point, Polygon as ShapelyPolygon

# ── parameters ────────────────────────────────────────────────────────
RWR_R                  = 0.1          # CHANGE THIS to match the permutation run
FDR_Q                  = 0.05
GLOBAL_LAMBDA_FALLBACK = 60.0         # used only if radial fit fails

R_TAG                  = f"r{int(RWR_R * 1000):04d}"

# ── paths ─────────────────────────────────────────────────────────────
BASE_DIR     = "/scratch/users/k22026807/masters/project/random_walks"
ADATA_PATH   = "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
ROI_PATH     = "/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson"
RESULTS      = os.path.join(BASE_DIR, "results/ccc_results")
PATTERN      = os.path.join(RESULTS, f"permutation_test_results_chunk*_{R_TAG}.csv")
OUT_CSV      = os.path.join(RESULTS, f"permutation_test_results_{R_TAG}.csv")
GRAPH_CACHE  = os.path.join(BASE_DIR, "results/disparity_graph.pkl")


# ── helpers ───────────────────────────────────────────────────────────
def build_resolvent(g, n_total, r=RWR_R):
    g = g.copy()
    missing = set(range(n_total)) - set(g.nodes())
    if missing:
        g.add_nodes_from(missing)
    nodes = list(range(n_total))
    A = nx.to_scipy_sparse_array(g, nodelist=nodes, format='csr', dtype=np.float64)
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    D_inv = csr_matrix(
        (1.0/deg, (np.arange(len(deg)), np.arange(len(deg)))),
        shape=A.shape, dtype=np.float64
    )
    P_hat = D_inv @ A
    n = A.shape[0]
    M = sparse_eye(n, format='csc') - (1.0 - r) * P_hat.tocsc()
    return splu(M), n


def rwr_from_lu(lu, seed_indices, n, r=RWR_R):
    if len(seed_indices) == 0:
        return np.zeros(n, dtype=np.float32)
    q = np.zeros(n, dtype=np.float64)
    q[seed_indices] = 1.0 / len(seed_indices)
    return (r * lu.solve(q)).astype(np.float32)


def estimate_lambda_radial(L_star, pos, seed_indices,
                            default=GLOBAL_LAMBDA_FALLBACK,
                            min_lambda=1.0, max_lambda=200.0):
    """
    Fit L* = L0 * exp(-d / lambda) where d = distance to nearest seed.
    Returns the fitted lambda, or default if fit fails or is out of range.
    """
    if len(seed_indices) == 0:
        return default

    seed_pos = pos[seed_indices]
    tree = cKDTree(seed_pos)
    d_to_seed, _ = tree.query(pos, k=1)

    threshold = L_star.max() * 1e-4
    mask = (d_to_seed > 0) & (L_star > threshold)
    if mask.sum() < 100:
        return default

    d = d_to_seed[mask]
    L = L_star[mask]
    log_L = np.log(L)
    slope, _ = np.polyfit(d, log_L, 1)

    if slope >= 0:
        return default

    lam = float(-1.0 / slope)
    return lam if min_lambda <= lam <= max_lambda else default


def effective_n(n, dx, lam):
    return max(1.0, n * dx**2 / (8 * np.pi * lam**2))


def bh_correct(p_values, n_eff, q=FDR_Q):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    thresholds = (np.arange(1, n+1) * q) / n_eff
    passes = p_sorted <= thresholds
    if passes.any():
        cutoff = np.where(passes)[0].max()
        passes[:cutoff+1] = True
    sig = np.zeros(n, dtype=bool)
    sig[order] = passes
    return sig


def bh_qvalues(p_values, n_eff):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    q_sorted = np.minimum(1.0, p_sorted * n_eff / np.arange(1, n+1))
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.zeros(n)
    q[order] = q_sorted
    return q


# ── main ──────────────────────────────────────────────────────────────
def main():
    print(f"=== merge_and_BH_correction | r={RWR_R} | tag={R_TAG} ===")

    chunks = sorted(glob.glob(PATTERN))
    if not chunks:
        raise FileNotFoundError(f"No chunks found at {PATTERN}")
    print(f"\nFound {len(chunks)} chunk files matching {R_TAG}")
    df = pd.concat([pd.read_csv(c) for c in chunks], ignore_index=True)
    print(f"Merged: {len(df):,} rows")
    df = df.drop_duplicates(subset=['sender_ct','ligand','receptor','receiver_ct'])

    # Load tissue
    print("\nLoading AnnData and graph...")
    adata = sc.read(ADATA_PATH)
    with open(ROI_PATH) as f:
        roi = json.load(f)
    poly = ShapelyPolygon(roi['features'][0]['geometry']['coordinates'][0])
    mask = np.array([poly.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
    adata = adata[mask].copy()
    n_cells = adata.shape[0]
    pos = adata.obsm['spatial']
    tree = cKDTree(pos)
    nn, _ = tree.query(pos, k=2)
    dx = float(np.median(nn[:, 1]))
    print(f"  N cells: {n_cells:,}, dx: {dx:.2f} um")

    # Graph
    if os.path.exists(GRAPH_CACHE):
        print(f"\nLoading cached graph from {GRAPH_CACHE}...")
        with open(GRAPH_CACHE, 'rb') as f:
            g_obs = pickle.load(f)
        print(f"  Edges: {g_obs.number_of_edges():,}")
    else:
        raise FileNotFoundError(
            f"Graph cache not found: {GRAPH_CACHE}. "
            "Run permutation_test.py first to build it."
        )

    # Resolvent at this r
    print(f"\nFactorising resolvent at r={RWR_R}...")
    t0 = time.time()
    lu, n_nodes = build_resolvent(g_obs, n_total=n_cells)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Lambda per sender
    print(f"\nEstimating lambda per sender from L* (r={RWR_R})...")
    unique_senders = sorted(df['sender_ct'].unique())
    print(f"  {len(unique_senders)} unique senders")

    lambda_table = {}
    for sender in unique_senders:
        sender_mask = (adata.obs['cell_type'] == sender).values
        seed_idx = np.where(sender_mask)[0]
        if len(seed_idx) == 0:
            lambda_table[sender] = GLOBAL_LAMBDA_FALLBACK
            continue
        L_star = rwr_from_lu(lu, seed_idx, n_nodes)
        lam = estimate_lambda_radial(L_star, pos, seed_idx)
        n_eff = effective_n(n_cells, dx, lam)
        lambda_table[sender] = lam
        print(f"    {sender:30s} | seeds={len(seed_idx):5d} | "
              f"lambda={lam:7.2f} um | n_eff={n_eff:8.1f}")

    # Apply to df
    df['lambda_est'] = df['sender_ct'].map(lambda_table).fillna(GLOBAL_LAMBDA_FALLBACK)
    df['n_eff']      = df['lambda_est'].apply(lambda l: effective_n(n_cells, dx, l))
    df['rwr_r']      = RWR_R
    df['significant'] = False
    df['q_value']     = 1.0

    print("\nApplying BH-FDR per LR pair...")
    for (lig, rec), group in df.groupby(['ligand','receptor']):
        n_eff_pair = group['n_eff'].median()
        sig = bh_correct(group['p_value'].values, n_eff_pair, q=FDR_Q)
        qv  = bh_qvalues(group['p_value'].values, n_eff_pair)
        df.loc[group.index, 'significant'] = sig
        df.loc[group.index, 'q_value']     = qv

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Total tests: {len(df):,}")
    print(f"Significant: {df['significant'].sum():,} ({100*df['significant'].mean():.1f}%)")

    print("\n=== Top 20 significant by SES ===")
    top = df[df['significant']].nlargest(20, 'SES')
    print(top[['sender_ct','ligand','receptor','receiver_ct',
               'obs_mean_ccc','SES','p_value','q_value','lambda_est','n_eff']
              ].to_string(index=False))


if __name__ == '__main__':
    main()