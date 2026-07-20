#!/usr/bin/env python3
"""
run_rwr_observed_sweep.py
===========================
Runs the ligand-seeded observed CCC pipeline (same logic as
run_rwr_observed_all.py) across several restart probabilities r, to
replace the old r-sweep CSVs that were built on the non-ligand-specific
MultiXrank seeding. This is the manuscript's "Sensitivity Analysis over
alpha" (eq. \\ref{eq:alpha_range}): report results across a range of r
values since the underlying biophysical parameters D_L and mu carry
uncertainty.

Only ONE thing changes per r: the factorised matrix A = I - (1-r)P.
The graph P itself is loaded once and reused across all r values, so
this costs (n_r_values x one factorisation) rather than reloading
AnnData or rebuilding the graph per r.

Usage:
    python run_rwr_observed_sweep.py --r_values 0.05,0.1,0.3,0.5,0.7
"""
import argparse, json, os, pickle, time
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
LR_PATH     = os.path.join(BASE_DIR, "data/cellchat_full.csv")
GRAPH_CACHE = os.path.join(BASE_DIR, "results/disparity_graph.pkl")
OUT_DIR     = os.path.join(BASE_DIR, "results/ccc_results")
os.makedirs(OUT_DIR, exist_ok=True)

SEED_TYPES = {
    "malignant_cell":              "Malignant cell",
    "t_cell":                      "T cell",
    "myeloid_cell":                "Myeloid cell",
    "fibroblast":                  "Fibroblast",
    "endothelial_cell":            "Endothelial cell",
    "b_cell":                      "B cell",
    "pericyte":                    "Pericyte",
    "epithelial_cell":             "Epithelial cell",
    "plasmacytoid_dendritic_cell": "Plasmacytoid dendritic cell",
    "mast_cell":                   "Mast cell",
}


def get_expressed_ligands(adata, cell_type, ligands, min_expr=0.01):
    ct_mask = (adata.obs['cell_type'] == cell_type).values
    X_ct = adata.X[ct_mask]
    X_ct = X_ct.toarray() if hasattr(X_ct, 'toarray') else np.asarray(X_ct)
    var_index = {g: i for i, g in enumerate(adata.var_names)}
    expressed = []
    for gene in ligands:
        if gene not in var_index:
            continue
        frac = (X_ct[:, var_index[gene]] > 0).mean()
        if frac >= min_expr:
            expressed.append(gene)
    return set(expressed)


def expr(adata, gene_or_complex):
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


def build_seed(L_expr, sender_indices, n):
    seed = np.zeros(n, dtype=np.float64)
    seed[sender_indices] = L_expr[sender_indices]
    s = seed.sum()
    if s > 0:
        seed /= s
    return seed


def run_for_r(r, adata, P, n, cell_types, df_lr, min_expr, k_binding):
    R_TAG = f"r{int(r*1000):04d}"
    print(f"\n{'='*60}\nr = {r}  ({R_TAG})\n{'='*60}")

    t0 = time.time()
    A = (sp.identity(n, format="csr") - (1.0 - r) * P).tocsc()
    A_lu = splu(A)
    print(f"  Factorisation done in {time.time()-t0:.1f}s")

    def solve_rwr(seed_vec):
        return (r * A_lu.solve(seed_vec)).astype(np.float32)

    results = []
    n_solves = 0
    t_loop = time.time()

    for seed_label, seed_ct in SEED_TYPES.items():
        sender_mask = (cell_types == seed_ct)
        sender_indices = np.where(sender_mask)[0]
        if len(sender_indices) == 0:
            continue

        all_ligands = set(df_lr['ligand'].tolist())
        expressed_ligands = get_expressed_ligands(adata, seed_ct, all_ligands, min_expr)
        lr_this = df_lr[df_lr['ligand'].isin(expressed_ligands)].copy()

        L_star_cache = {}
        for _, row in lr_this.iterrows():
            ligand, receptor, pathway = row['ligand'], row['receptor'], row['pathway']

            if ligand not in L_star_cache:
                L_expr = expr(adata, ligand)
                if L_expr is None or L_expr[sender_indices].sum() == 0:
                    L_star_cache[ligand] = None
                else:
                    seed = build_seed(L_expr, sender_indices, n)
                    L_star_cache[ligand] = solve_rwr(seed)
                    n_solves += 1

            L_star = L_star_cache[ligand]
            if L_star is None:
                continue

            R = expr(adata, receptor)
            if R is None:
                continue
            c_hat = k_binding * L_star * R
            if c_hat.sum() == 0:
                continue

            for recv_ct in np.unique(cell_types):
                recv_mask = (cell_types == recv_ct)
                if recv_mask.sum() == 0:
                    continue
                results.append({
                    'sender_ct': seed_ct, 'seed_label': seed_label,
                    'ligand': ligand, 'receptor': receptor, 'pathway': pathway,
                    'receiver_ct': recv_ct,
                    'mean_ccc': float(c_hat[recv_mask].mean()),
                    'max_ccc': float(c_hat[recv_mask].max()),
                    'n_receiver': int(recv_mask.sum()),
                    'pct_R_expr': float((R[recv_mask] > 0).mean() * 100),
                    'r': r,
                })

    print(f"  {n_solves} RWR solves, {len(results):,} rows, "
          f"{(time.time()-t_loop)/60:.1f} min")

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, f"ccc_all_lr_pairs_ligandseeded_{R_TAG}.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r_values", type=str, default="0.05,0.1,0.3,0.5,0.7",
                     help="Comma-separated list of restart probabilities.")
    ap.add_argument("--min_expr", type=float, default=0.01)
    ap.add_argument("--k_binding", type=float, default=1.0)
    args = ap.parse_args()

    r_values = [float(x) for x in args.r_values.split(",")]
    print(f"=== r-sweep (ligand-seeded): r in {r_values} ===")

    print("\nLoading AnnData...")
    adata = sc.read(ADATA_PATH)
    with open(ROI_PATH) as f:
        roi = json.load(f)
    polygon = ShapelyPolygon(roi['features'][0]['geometry']['coordinates'][0])
    mask = np.array([polygon.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
    adata = adata[mask].copy()
    n = adata.n_obs
    cell_types = adata.obs['cell_type'].values
    print(f"ROI cells: {n:,}")

    df_lr = pd.read_csv(LR_PATH, header=0, names=['ligand', 'receptor', 'pathway', 'type'])
    panel_genes = set(adata.var_names)
    df_lr = df_lr[df_lr['ligand'].isin(panel_genes)].copy()
    print(f"CellChatDB pairs with ligand in panel: {len(df_lr):,}")

    print(f"\nLoading cached graph (shared across all r values)...")
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

    row_sums = np.asarray(G.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv @ G  # transition matrix, r-independent, built ONCE

    all_dfs = []
    for r in r_values:
        df_r = run_for_r(r, adata, P, n, cell_types, df_lr, args.min_expr, args.k_binding)
        all_dfs.append(df_r)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "ccc_all_lr_pairs_ligandseeded_sweep_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved combined sweep file: {combined_path}")
    print(f"Total rows across all r values: {len(combined):,}")

    # Quick concordance check: does the top-ranked pair change across r?
    print("\nTop 5 pairs by mean_ccc at each r (paracrine only):")
    for r in r_values:
        sub = combined[(combined['r'] == r) & (combined['sender_ct'] != combined['receiver_ct'])]
        top = sub.sort_values('mean_ccc', ascending=False).head(5)
        print(f"\n  r={r}:")
        print(top[['sender_ct', 'ligand', 'receptor', 'receiver_ct', 'mean_ccc']]
              .to_string(index=False))


if __name__ == '__main__':
    main()