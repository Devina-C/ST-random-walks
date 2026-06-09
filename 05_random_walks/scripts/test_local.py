#!/usr/bin/env python3
"""
test_local.py
=============
Local smoke test for permutation_test.py logic.
Runs B=20 permutations on 3 triplets so the whole pipeline (graph build,
permutation, lambda estimation, output schema) can be verified in ~10 min.

Outputs a CSV alongside the real pipeline outputs but with a _test suffix
so it doesn't overwrite real results.
"""
import os, sys, json
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from shapely.geometry import Point, Polygon as ShapelyPolygon
from joblib import Parallel, delayed

# Make sure graph.py is importable. Adjust if your local layout differs.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import disparity_filter, disparity_filter_alpha_cut

# ── LOCAL paths ───────────────────────────────────────────────────────
# Update these to your local Windows paths
BASE_DIR     = "C:/Users/Devin/Documents/ST_ccc/05_random_walks"
ADATA_PATH   = "C:/Users/Devin/Documents/ST_ccc/02_cell_typing/results/celltype_output/BC_prime/refined_annotations.h5ad"
ROI_PATH     = "C:/Users/Devin/Documents/ST_ccc/03_roi_extraction/results/region1_xenium.geojson"
OBSERVED_CSV = os.path.join(BASE_DIR, "results/ccc_results/ccc_all_lr_pairs.csv")
OUT_CSV      = os.path.join(BASE_DIR, "results/ccc_results/permutation_test_LOCAL_TEST.csv")

# ── TEST parameters (small) ───────────────────────────────────────────
B            = 20         # tiny — just to verify the loop runs
N_TRIPLETS   = 3          # how many triplets to test
ALPHA_FILTER = 0.005
RADIUS       = 200
RWR_R        = 0.7
MIN_PCT_R    = 1.0
MIN_OBS_CCC  = 1e-7
MIN_N_RECV   = 50
N_JOBS       = 1          # use 4 cores locally so your laptop stays usable


# ── graph + RWR (identical to permutation_test.py) ───────────────────
def build_disparity_graph(pos, alpha=ALPHA_FILTER, radius=RADIUS):
    n = len(pos)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')
    rows, cols = pairs[:, 0], pairs[:, 1]
    dists = np.linalg.norm(pos[rows] - pos[cols], axis=1)
    ID = csr_matrix((1.0/dists, (rows, cols)), shape=(n, n))
    ID = ID + ID.T
    g = nx.from_scipy_sparse_array(ID)
    g = disparity_filter(g)
    g = disparity_filter_alpha_cut(g, alpha_t=alpha)
    return g


def rwr_scores(g, seed_node_ids, cell_ids, r=RWR_R):
    seeds_in = [s for s in seed_node_ids if s in g]
    if not seeds_in:
        return np.zeros(len(cell_ids), dtype=np.float32)
    personalization = {c: 1.0/len(seeds_in) for c in seeds_in}
    scores = nx.pagerank(g, alpha=1-r, personalization=personalization,
                         max_iter=200, tol=1e-8)
    return np.array([scores.get(c, 0.0) for c in cell_ids], dtype=np.float32)


def get_receptor_expression(adata, receptor_str):
    subunits = receptor_str.split('_')
    if any(s not in adata.var_names for s in subunits):
        return None
    R_sub = []
    for s in subunits:
        r = adata[:, s].X
        r = r.toarray().flatten() if hasattr(r, 'toarray') else np.array(r).flatten()
        R_sub.append(r.astype(np.float32))
    R = R_sub[0].copy()
    for arr in R_sub[1:]:
        R *= arr
    return R


def estimate_lambda(c_obs, pos, max_dist=300, n_bins=20, default=60.0):
    rng = np.random.default_rng(0)
    sample = rng.choice(len(pos), size=min(2000, len(pos)), replace=False)
    tree = cKDTree(pos)
    bins = np.linspace(0, max_dist, n_bins+1)
    rhos = []
    for i in range(n_bins):
        d_lo, d_hi = bins[i], bins[i+1]
        a, b = [], []
        for s in sample:
            for j in tree.query_ball_point(pos[s], d_hi):
                d = np.linalg.norm(pos[s] - pos[j])
                if d_lo < d <= d_hi and s != j:
                    a.append(c_obs[s]); b.append(c_obs[j])
        if len(a) > 20 and np.std(a) > 0 and np.std(b) > 0:
            rhos.append(((d_lo+d_hi)/2, np.corrcoef(a, b)[0, 1]))
    if len(rhos) < 3:
        return default
    d_arr, r_arr = np.array(rhos).T
    valid = (r_arr > 0.01) & (r_arr < 1.0)
    if valid.sum() < 3:
        return default
    slope = np.polyfit(d_arr[valid], np.log(r_arr[valid]), 1)[0]
    if slope >= 0:
        return default
    lam = float(-np.sqrt(2)/slope)
    return lam if 10.0 <= lam <= 500.0 else default


def single_permutation(b, pos, cell_ids, sender_mask, R, recv_masks, rng_seed):
    rng = np.random.default_rng(rng_seed + b)
    perm = rng.permutation(len(pos))
    pos_perm = pos[perm]
    g_perm = build_disparity_graph(pos_perm)
    g_perm = nx.relabel_nodes(g_perm, {i: cell_ids[i] for i in range(len(cell_ids))})
    seed_node_ids = [cell_ids[i] for i in range(len(cell_ids)) if sender_mask[i]]
    L_perm = rwr_scores(g_perm, seed_node_ids, cell_ids)
    c_perm = L_perm * R
    return {ct: float(c_perm[m].mean()) if m.any() else 0.0
            for ct, m in recv_masks.items()}


def test_one_triplet(adata, g_obs, cell_ids, sender_ct, ligand, receptor, B=B):
    R = get_receptor_expression(adata, receptor)
    if R is None or R.sum() == 0:
        print("  Receptor missing or zero expression - skip")
        return None

    sender_mask = (adata.obs['cell_type'] == sender_ct).values
    if sender_mask.sum() == 0:
        return None

    seed_node_ids = [cell_ids[i] for i in range(len(cell_ids)) if sender_mask[i]]

    print(f"  Computing observed L*... ({sender_mask.sum()} seeds)")
    L_obs = rwr_scores(g_obs, seed_node_ids, cell_ids)
    c_obs = L_obs * R

    cts = adata.obs['cell_type'].unique().tolist()
    recv_masks = {ct: (adata.obs['cell_type'] == ct).values for ct in cts}

    print("  Estimating lambda...")
    lam = estimate_lambda(c_obs, adata.obsm['spatial'])
    print(f"    lambda_est = {lam:.1f} um")

    print(f"  Running {B} permutations on {N_JOBS} cores...")
    pos = adata.obsm['spatial']
    null_runs = Parallel(n_jobs=N_JOBS, backend='loky', verbose=5)(
        delayed(single_permutation)(b, pos, cell_ids, sender_mask, R,
                                     recv_masks, rng_seed=42)
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
            'lambda_est':   lam,
        })
    return pd.DataFrame(rows) if rows else None


# ── main ──────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print(f"LOCAL TEST: B={B}, N_TRIPLETS={N_TRIPLETS}")
    print("="*60)

    print("\nLoading AnnData...")
    adata = sc.read(ADATA_PATH)
    with open(ROI_PATH) as f:
        roi = json.load(f)
    poly = ShapelyPolygon(roi['features'][0]['geometry']['coordinates'][0])
    mask = np.array([poly.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
    adata = adata[mask].copy()
    cell_ids = adata.obs_names.astype(str).tolist()
    print(f"  ROI cells: {adata.shape[0]:,}")

    print("\nBuilding observed disparity graph (one-time)...")
    g_obs = build_disparity_graph(adata.obsm['spatial'])
    g_obs = nx.relabel_nodes(g_obs, {i: cell_ids[i] for i in range(len(cell_ids))})
    print(f"  Edges: {g_obs.number_of_edges():,}")

    # Pick test triplets — three of your top observed CCCs, biologically diverse
    print("\nSelecting test triplets...")
    obs_ccc = pd.read_csv(OBSERVED_CSV)
    test_df = (obs_ccc
               .groupby(['sender_ct','ligand','receptor'])
               .agg(max_pct_R=('pct_R_expr','max'),
                    max_ccc=('mean_ccc','max'))
               .reset_index())
    test_df = test_df[(test_df['max_pct_R'] > MIN_PCT_R) &
                      (test_df['max_ccc'] > MIN_OBS_CCC)]
    test_df = (test_df
               .sort_values('max_ccc', ascending=False)
               .drop_duplicates('sender_ct')   # one per sender for diversity
               .head(N_TRIPLETS))
    print(test_df.to_string(index=False))

    # Run tests
    all_results = []
    for i, row in test_df.iterrows():
        print(f"\n--- Triplet {len(all_results)+1}/{N_TRIPLETS} ---")
        print(f"  Sender:   {row['sender_ct']}")
        print(f"  Ligand:   {row['ligand']}")
        print(f"  Receptor: {row['receptor']}")
        out = test_one_triplet(adata, g_obs, cell_ids,
                               row['sender_ct'], row['ligand'], row['receptor'],
                               B=B)
        if out is not None:
            all_results.append(out)

    if not all_results:
        print("\nNo results produced. Something is wrong.")
        return

    df = pd.concat(all_results, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*60}")
    print(f"Wrote {len(df)} rows -> {OUT_CSV}")
    print(f"{'='*60}")

    # Sanity checks
    print("\n=== Sanity check: schema ===")
    print("Columns:", list(df.columns))
    print(f"Dtypes:\n{df.dtypes}")

    print("\n=== Sanity check: p-value distribution ===")
    print(df['p_value'].describe())

    print("\n=== Sanity check: SES distribution ===")
    print(df['SES'].describe())

    print("\n=== Sanity check: lambda estimates ===")
    print(df.groupby(['ligand','receptor'])['lambda_est'].first())

    print("\n=== Full output ===")
    print(df.to_string(index=False))

    print("\nIf this looks sensible, the full HPC run should work.")
    print("Key things to check:")
    print("  - All columns populated")
    print("  - p_value between 0 and 1")
    print("  - SES > 1 for biologically real interactions (e.g. autocrine)")
    print("  - lambda_est mostly NOT exactly 60.0 (the fallback)")
    print("  - n_receiver matches expected cell type counts")


if __name__ == '__main__':
    main()