#!/usr/bin/env python3
"""
lr_score.py
===========
Generate CCC = k * L* * R per (sender, ligand, receptor, receiver) using
RWR rankings produced by run_rwr.py at a given RWR_RESTART value.

Output filename is r-tagged so multiple r values coexist.
"""
import json, os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point, Polygon as ShapelyPolygon

# ── parameters ────────────────────────────────────────────────────────
RWR_RESTART = 0.7        # MUST match run_rwr.py / update_seeds.py
R_TAG       = f"r{int(RWR_RESTART * 1000):04d}"

k = 1.0                  # binding ratio
min_expr = 0.01          # ligand expressed in >=1% of seed cells

# ── paths ─────────────────────────────────────────────────────────────
BASE_DIR   = "/scratch/users/k22026807/masters/project/random_walks"
ADATA_PATH = "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
ROI_PATH   = "/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson"
LR_PATH    = os.path.join(BASE_DIR, "data/cellchat_full.csv")
OUT_DIR    = os.path.join(BASE_DIR, "results/ccc_results")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, f'ccc_all_lr_pairs_{R_TAG}.csv')


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
    ct_mask = adata.obs['cell_type'] == cell_type
    adata_ct = adata[ct_mask]
    if hasattr(adata_ct.X, 'toarray'):
        X_ct = adata_ct.X.toarray()
    else:
        X_ct = np.array(adata_ct.X)
    expressed = []
    for gene in ligands:
        if gene not in adata.var_names:
            continue
        idx = list(adata.var_names).index(gene)
        frac = (X_ct[:, idx] > 0).mean()
        if frac >= min_expr:
            expressed.append(gene)
    return set(expressed)


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
    for r in R_sub[1:]:
        R *= r
    return R


def load_rwr_scores(seed_label, cell_ids, n_seeds, r_tag):
    """Load L* from r-tagged RWR ranking, map to AnnData cell order."""
    path = os.path.join(BASE_DIR,
        f"rwr_runs/{seed_label}_{r_tag}/cell_ranking.tsv/multiplex_1.tsv")
    if not os.path.exists(path):
        return None
    ranking = pd.read_csv(path, sep='\t')
    ranking['node'] = ranking['node'].astype(str)
    scores = dict(zip(ranking['node'], ranking['score']))
    L_star = np.array([scores.get(c, 0.0) for c in cell_ids], dtype=np.float32)
    return L_star


def main():
    print(f"=== lr_score | r={RWR_RESTART} | tag={R_TAG} ===")
    print(f"Output: {OUT_CSV}")

    print("\nLoading AnnData...")
    adata = sc.read(ADATA_PATH)
    with open(ROI_PATH) as f:
        roi = json.load(f)
    polygon = ShapelyPolygon(roi['features'][0]['geometry']['coordinates'][0])
    mask = np.array([polygon.contains(Point(x, y))
                     for x, y in adata.obsm['spatial']])
    adata = adata[mask].copy()
    cell_ids = adata.obs_names.astype(str).tolist()
    print(f"ROI cells: {adata.shape[0]:,}")

    df_lr = pd.read_csv(LR_PATH, header=0,
                        names=['ligand','receptor','pathway','type'])
    print(f"CellChatDB pairs: {len(df_lr):,}")
    panel_genes = set(adata.var_names)
    df_lr = df_lr[df_lr['ligand'].isin(panel_genes)].copy()
    print(f"Ligand in panel: {len(df_lr):,}")

    results = []

    for seed_label, seed_ct in SEED_TYPES.items():
        print(f"\nSeed: {seed_ct}")
        n_seeds = int((adata.obs['cell_type'] == seed_ct).sum())
        L_star = load_rwr_scores(seed_label, cell_ids, n_seeds, R_TAG)
        if L_star is None:
            print(f"  No RWR found for {seed_label}_{R_TAG}. Skipping.")
            continue
        print(f"  L* loaded. Range [{L_star.min():.2e}, {L_star.max():.2e}]")

        all_ligands = set(df_lr['ligand'].tolist())
        expressed_ligands = get_expressed_ligands(
            adata, seed_ct, all_ligands, min_expr)
        print(f"  Expressed ligands (>={min_expr*100:.0f}%): {len(expressed_ligands):,}")

        lr_this = df_lr[df_lr['ligand'].isin(expressed_ligands)].copy()
        print(f"  Valid LR pairs: {len(lr_this):,}")

        for _, row in lr_this.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            pathway = row['pathway']

            R = get_receptor_expression(adata, receptor)
            if R is None:
                continue
            c_hat = k * L_star * R
            if c_hat.sum() == 0:
                continue

            for recv_ct in adata.obs['cell_type'].unique():
                recv_mask = (adata.obs['cell_type'] == recv_ct).values
                if recv_mask.sum() == 0:
                    continue
                results.append({
                    'sender_ct':   seed_ct,
                    'seed_label':  seed_label,
                    'ligand':      ligand,
                    'receptor':    receptor,
                    'pathway':     pathway,
                    'receiver_ct': recv_ct,
                    'mean_ccc':    float(c_hat[recv_mask].mean()),
                    'max_ccc':     float(c_hat[recv_mask].max()),
                    'n_receiver':  int(recv_mask.sum()),
                    'pct_R_expr':  float((R[recv_mask] > 0).mean() * 100),
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\nTotal interactions: {len(results_df):,}")
    print(f"Saved: {OUT_CSV}")

    top = results_df.sort_values('mean_ccc', ascending=False).head(30)
    print("\nTop 30 CCC interactions:")
    print(top[['sender_ct','ligand','receptor','receiver_ct','mean_ccc','pathway']].to_string())

    # Sender→Receiver aggregate matrix (r-tagged output too)
    agg = (results_df
           .groupby(['sender_ct','receiver_ct'])['mean_ccc']
           .mean().reset_index()
           .sort_values('mean_ccc', ascending=False))
    agg_csv = os.path.join(OUT_DIR, f'ccc_sender_receiver_matrix_{R_TAG}.csv')
    agg.to_csv(agg_csv, index=False)
    print(f"\nAggregate saved: {agg_csv}")

    # Heatmap (r-tagged filename)
    all_cts = sorted(set(agg['sender_ct'].tolist() + agg['receiver_ct'].tolist()))
    matrix = agg.pivot(index='sender_ct', columns='receiver_ct',
                       values='mean_ccc').fillna(0)
    matrix = matrix.reindex(index=all_cts, columns=all_cts, fill_value=0)
    matrix_values = matrix.values.astype(np.float64)
    for i in range(len(all_cts)):
        matrix_values[i, i] = np.nan

    fig, ax = plt.subplots(figsize=(13, 10))
    finite = matrix_values[~np.isnan(matrix_values) & (matrix_values > 0)]
    if len(finite) > 0:
        im = ax.imshow(matrix_values, cmap='YlOrRd', aspect='auto',
                       norm=mcolors.LogNorm(vmin=finite.min(), vmax=finite.max()))
        plt.colorbar(im, ax=ax, shrink=0.6, label='Mean CCC score (log scale)')
    ax.set_xticks(range(len(all_cts)))
    ax.set_yticks(range(len(all_cts)))
    ax.set_xticklabels(all_cts, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(all_cts, fontsize=9)
    ax.set_xlabel('Receiver cell type')
    ax.set_ylabel('Sender cell type')
    ax.set_title(f'RWR CCC sender-receiver (r={RWR_RESTART}, autocrine excluded)',
                 fontweight='bold')
    ax.set_facecolor('#dddddd')
    plt.tight_layout()
    heatmap_path = os.path.join(OUT_DIR, f'sender_receiver_heatmap_{R_TAG}.png')
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")


if __name__ == '__main__':
    main()