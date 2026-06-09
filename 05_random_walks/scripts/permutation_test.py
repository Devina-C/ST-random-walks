#!/usr/bin/env python3
"""
permutation_test.py
===================
Spatial permutation test (H_0^space) for CCC scores.

Implementation of Algorithm 1 from the manuscript. The graph topology
is fixed (positions don't change), and cell identities are permuted
across nodes — this is mathematically equivalent to the manuscript's
"shuffle positions, rebuild graph" formulation.

To rerun at a different restart probability, change RWR_R below. All
output filenames are tagged with R_TAG so multiple runs do not overwrite.

Environment variables:
    SLURM_ARRAY_TASK_ID : chunk index (0-indexed)
    N_CHUNKS            : total number of chunks
"""
import os, sys, json, time, pickle
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import splu
from shapely.geometry import Point, Polygon as ShapelyPolygon
from joblib import Parallel, delayed

from graph import disparity_filter, disparity_filter_alpha_cut

# ── parameters ────────────────────────────────────────────────────────
RWR_R        = 0.7        # CHANGE THIS to rerun at different restart probabilities
B            = 1000       # permutations per triplet
ALPHA_FILTER = 0.005      # disparity filter alpha
RADIUS       = 200        # cKDTree neighbour radius (um)

MIN_PCT_R    = 1.0        # receptor expressed in >=1% of any receiver
MIN_OBS_CCC  = 1e-7       # observed signal floor
MIN_N_RECV   = 50         # min cells in receiver population

N_JOBS       = 1          # joblib workers per chunk

R_TAG        = f"r{int(RWR_R * 1000):04d}"   # e.g. 'r0700', 'r0100'

# Chunking via SLURM env vars
CHUNK_IDX    = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
N_CHUNKS     = int(os.environ.get('N_CHUNKS', 1))

# ── paths ─────────────────────────────────────────────────────────────
BASE_DIR     = "/scratch/users/k22026807/masters/project/random_walks"
ADATA_PATH   = "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
ROI_PATH     = "/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson"
OBSERVED_CSV = os.path.join(BASE_DIR, f"results/ccc_results/ccc_all_lr_pairs_{R_TAG}.csv")
OUT_DIR      = os.path.join(BASE_DIR, "results/ccc_results")
GRAPH_CACHE  = os.path.join(BASE_DIR, "results/disparity_graph.pkl")
os.makedirs(OUT_DIR, exist_ok=True)


# ── graph construction ────────────────────────────────────────────────
def build_disparity_graph(pos):
    """Build disparity-filtered spatial graph. Nodes are integers 0..n-1."""
    n = len(pos)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=RADIUS, output_type='ndarray')
    rows, cols = pairs[:, 0], pairs[:, 1]
    dists = np.linalg.norm(pos[rows] - pos[cols], axis=1)
    ID = csr_matrix((1.0/dists, (rows, cols)), shape=(n, n))
    ID = ID + ID.T
    g = nx.from_scipy_sparse_array(ID)
    g = disparity_filter(g)
    g = disparity_filter_alpha_cut(g, alpha_t=ALPHA_FILTER)
    return g


# ── resolvent (depends on RWR_R) ─────────────────────────────────────
def build_resolvent(g, n_total, r=RWR_R):
    """Pre-factorise M = I - (1-r) * P_hat. Adds isolated cells as 0-rows."""
    g = g.copy()
    missing = set(range(n_total)) - set(g.nodes())
    if missing:
        g.add_nodes_from(missing)
        print(f"  Added {len(missing)} isolated nodes (dropped by disparity filter)")
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
    """One sparse back-substitution."""
    if len(seed_indices) == 0:
        return np.zeros(n, dtype=np.float32)
    q = np.zeros(n, dtype=np.float64)
    q[seed_indices] = 1.0 / len(seed_indices)
    return (r * lu.solve(q)).astype(np.float32)


# ── expression helpers ────────────────────────────────────────────────
def get_receptor_expression(adata, receptor_str):
    subs = receptor_str.split('_')
    if any(s not in adata.var_names for s in subs):
        return None
    Rs = []
    for s in subs:
        r = adata[:, s].X
        r = r.toarray().flatten() if hasattr(r, 'toarray') else np.array(r).flatten()
        Rs.append(r.astype(np.float32))
    R = Rs[0].copy()
    for arr in Rs[1:]:
        R *= arr
    return R


# ── single permutation ────────────────────────────────────────────────
def single_permutation(b, lu, n, sender_indices, R, recv_masks, rng_seed):
    rng = np.random.default_rng(rng_seed + b)
    perm = rng.permutation(n)
    perm_inv = np.argsort(perm)
    new_sender_nodes = perm[sender_indices]
    R_at_nodes = R[perm_inv]
    L_perm = rwr_from_lu(lu, new_sender_nodes, n)
    if b == 0:
        pass  # could time here if needed
    c_perm = L_perm * R_at_nodes
    return {ct: float(c_perm[m[perm_inv]].mean()) if m.any() else 0.0
            for ct, m in recv_masks.items()}


# ── test one triplet ──────────────────────────────────────────────────
def test_one_triplet(adata, lu, n, sender_ct, ligand, receptor, B=B):
    R = get_receptor_expression(adata, receptor)
    if R is None or R.sum() == 0:
        return None
    sender_mask = (adata.obs['cell_type'] == sender_ct).values
    if sender_mask.sum() == 0:
        return None
    sender_indices = np.where(sender_mask)[0]

    L_obs = rwr_from_lu(lu, sender_indices, n)
    c_obs = L_obs * R

    cts = adata.obs['cell_type'].unique().tolist()
    recv_masks = {ct: (adata.obs['cell_type'] == ct).values for ct in cts}

    null_runs = Parallel(n_jobs=N_JOBS, backend='loky', verbose=0)(
        delayed(single_permutation)(b, lu, n, sender_indices, R, recv_masks,
                                     rng_seed=42)
        for b in range(B)
    )

    rows = []
    for ct in cts:
        if recv_masks[ct].sum() < MIN_N_RECV:
            continue
        obs = float(c_obs[recv_masks[ct]].mean())
        null = np.array([nr[ct] for nr in null_runs])
        p = float((null >= obs).sum() / B)
        ses = float(obs / null.mean()) if null.mean() > 0 else np.nan
        rows.append({
            'sender_ct':    sender_ct,
            'ligand':       ligand,
            'receptor':     receptor,
            'receiver_ct':  ct,
            'n_receiver':   int(recv_masks[ct].sum()),
            'obs_mean_ccc': obs,
            'null_mean':    float(null.mean()),
            'null_std':     float(null.std()),
            'p_value':      p,
            'SES':          ses,
        })
    return pd.DataFrame(rows) if rows else None


# ── main ──────────────────────────────────────────────────────────────
def main():
    print(f"=== Chunk {CHUNK_IDX + 1}/{N_CHUNKS} | B={B} | r={RWR_R} | tag={R_TAG} ===")
    print(f"N_JOBS={N_JOBS}")
    print(f"Reading observed CCC from: {OBSERVED_CSV}")

    print("\nLoading AnnData...")
    adata = sc.read(ADATA_PATH)
    with open(ROI_PATH) as f:
        roi = json.load(f)
    poly = ShapelyPolygon(roi['features'][0]['geometry']['coordinates'][0])
    mask = np.array([poly.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
    adata = adata[mask].copy()
    print(f"  ROI cells: {adata.shape[0]:,}")

    # Graph: load from cache if available, else build and cache
    if os.path.exists(GRAPH_CACHE):
        print(f"Loading cached graph from {GRAPH_CACHE}...")
        t0 = time.time()
        with open(GRAPH_CACHE, 'rb') as f:
            g_obs = pickle.load(f)
        print(f"  Edges: {g_obs.number_of_edges():,} | load time: {time.time()-t0:.1f}s")
    else:
        print("Building disparity graph (one-time)...")
        t0 = time.time()
        g_obs = build_disparity_graph(adata.obsm['spatial'])
        print(f"  Edges: {g_obs.number_of_edges():,} | build time: {time.time()-t0:.1f}s")
        os.makedirs(os.path.dirname(GRAPH_CACHE), exist_ok=True)
        try:
            with open(GRAPH_CACHE, 'wb') as f:
                pickle.dump(g_obs, f)
            print(f"  Cached to {GRAPH_CACHE}")
        except Exception as e:
            print(f"  Could not cache graph: {e}")

    # Resolvent depends on r — rebuild each run
    print(f"Factorising resolvent at r={RWR_R}...")
    t0 = time.time()
    lu, n_nodes = build_resolvent(g_obs, n_total=adata.shape[0])
    print(f"  Done in {time.time() - t0:.1f}s")

    # Test set
    print("\nBuilding test set...")
    obs_ccc = pd.read_csv(OBSERVED_CSV)
    test_df = (obs_ccc
               .groupby(['sender_ct','ligand','receptor'])
               .agg(max_pct_R=('pct_R_expr','max'),
                    max_ccc=('mean_ccc','max'))
               .reset_index())
    test_df = test_df[(test_df['max_pct_R'] > MIN_PCT_R) &
                      (test_df['max_ccc'] > MIN_OBS_CCC)]
    test_df = test_df.sort_values(['sender_ct','ligand','receptor']).reset_index(drop=True)
    print(f"  Total triplets: {len(test_df):,}")

    chunk_size = int(np.ceil(len(test_df) / N_CHUNKS))
    start = CHUNK_IDX * chunk_size
    end = min(start + chunk_size, len(test_df))
    test_df = test_df.iloc[start:end].reset_index(drop=True)
    print(f"  This chunk: triplets [{start}:{end}] = {len(test_df)}")
    if len(test_df) > 0:
        print(f"  First: {test_df.iloc[0]['sender_ct']} | "
              f"{test_df.iloc[0]['ligand']} -> {test_df.iloc[0]['receptor']}")
        print(f"  Last:  {test_df.iloc[-1]['sender_ct']} | "
              f"{test_df.iloc[-1]['ligand']} -> {test_df.iloc[-1]['receptor']}")

    out_csv = os.path.join(OUT_DIR,
                           f"permutation_test_results_chunk{CHUNK_IDX:03d}_{R_TAG}.csv")

    all_results = []
    for i, row in test_df.iterrows():
        t0 = time.time()
        print(f"\n[{i+1}/{len(test_df)}] {row['sender_ct']} | "
              f"{row['ligand']} -> {row['receptor']}")
        try:
            out = test_one_triplet(adata, lu, n_nodes,
                                   row['sender_ct'], row['ligand'], row['receptor'],
                                   B=B)
            if out is not None:
                all_results.append(out)
                print(f"  Done in {time.time()-t0:.1f}s -> {len(out)} rows")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            continue

        if (i+1) % 5 == 0 and all_results:
            partial = pd.concat(all_results, ignore_index=True)
            partial.to_csv(out_csv + '.partial', index=False)

    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        df.to_csv(out_csv, index=False)
        print(f"\nDone. Wrote {len(df):,} rows -> {out_csv}")
        partial_path = out_csv + '.partial'
        if os.path.exists(partial_path):
            os.remove(partial_path)
    else:
        print("\nNo results to write.")


if __name__ == '__main__':
    main()