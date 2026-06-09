#!/usr/bin/env python3
"""
visualise.py
============
Visualise RWR scores for one seed type.
Usage: python visualise.py malignant_cell "Malignant cell"
"""

import json, sys, os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon as ShapelyPolygon

BASE_DIR   = "C:/Users/Devin/Documents/ST_ccc/05_random_walks"
ADATA_PATH = "C:/Users/Devin/Documents/ST_ccc/02_cell_typing/results/celltype_output/BC_prime/refined_annotations.h5ad"
ROI_PATH   = "C:/Users/Devin/Documents/ST_ccc/03_roi_extraction/results/region1_xenium.geojson"

SEED_LABEL = sys.argv[1] if len(sys.argv) > 1 else "malignant_cell"
SEED_CT    = sys.argv[2] if len(sys.argv) > 2 else "Malignant cell"
RUN_DIR    = os.path.join(BASE_DIR, f"rwr_runs/{SEED_LABEL}")
OUT_DIR    = os.path.join(BASE_DIR, "viz")
os.makedirs(OUT_DIR, exist_ok=True)

ct_palette = {
    "Myeloid cell":                "#e6550d",
    "T cell":                      "#5b5bd6",
    "NK cell":                     "#a63603",
    "B cell":                      "#984ea3",
    "Plasmacytoid dendritic cell": "#20b2aa",
    "Fibroblast":                  "#d8b365",
    "Pericyte":                    "#67a9cf",
    "Endothelial cell":            "#66c2a5",
    "Epithelial cell":             "#636363",
    "Megakaryocyte":               "#fb9a99",
    "Mast cell":                   "#ffd92f",
    "Malignant cell":              "#e31a1c",
}

print(f"Visualising: {SEED_CT} (label: {SEED_LABEL})")
# Load data
print("Loading AnnData...")
adata = sc.read(ADATA_PATH)
with open(ROI_PATH) as f:
    roi = json.load(f)
polygon = ShapelyPolygon(roi['features'][0]['geometry']['coordinates'][0])
mask = np.array([polygon.contains(Point(x, y))
                 for x, y in adata.obsm['spatial']])
adata = adata[mask].copy()

# Attach scores
ranking = pd.read_csv(f"{RUN_DIR}/cell_ranking.tsv/multiplex_1.tsv", sep='\t')
ranking['node'] = ranking['node'].astype(str)
scores = dict(zip(ranking['node'], ranking['score']))
adata.obs['rwr_score'] = adata.obs_names.astype(str).map(scores).fillna(0)
adata.obs['rwr_log10'] = np.log10(adata.obs['rwr_score'] + 1e-10)

print(f"Total score: {adata.obs['rwr_score'].sum():.4f}  (should be ~1.0)")

pos       = adata.obsm['spatial']
seed_mask = (adata.obs['cell_type'] == SEED_CT).values
n_seeds   = seed_mask.sum()

# ── Spatial plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')

sc_plot = ax.scatter(
    pos[~seed_mask, 0], pos[~seed_mask, 1],
    c=adata.obs['rwr_log10'].values[~seed_mask],
    cmap='viridis', s=2, alpha=0.8,
    linewidths=0, rasterized=True, zorder=1)
plt.colorbar(sc_plot, ax=ax, shrink=0.6, label='RWR score (log₁₀)')

# Seeds — white dots, contrasts against all viridis colours
ax.scatter(
    pos[seed_mask, 0], pos[seed_mask, 1],
    c='navy', marker='o', s=8,
    linewidths=0, alpha=0.85, zorder=3,
    label=f'{SEED_CT} seeds (n={n_seeds:,})')

ax.set_title(f'RWR — seeds: {SEED_CT}',
             fontsize=13, fontweight='bold')
ax.set_aspect('equal')
ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
ax.legend(fontsize=9, markerscale=2, loc='upper right', framealpha=0.7)
plt.tight_layout()

out = os.path.join(OUT_DIR, f"{SEED_LABEL}_rwr_spatial.png")
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── Boxplot ───────────────────────────────────────────────────────────────────
non_seed = adata.obs[adata.obs['cell_type'] != SEED_CT].copy()
ct_summary = (non_seed
              .groupby('cell_type', observed=True)['rwr_score']
              .agg(['mean', 'median', 'count'])
              .sort_values('mean', ascending=False))
ct_summary.to_csv(os.path.join(OUT_DIR, f"{SEED_LABEL}_rwr_by_celltype.csv"))
print(f"\nCell type ranking:\n{ct_summary.to_string()}")

order  = ct_summary.index.tolist()
data   = [non_seed[non_seed['cell_type'] == ct]['rwr_score'].values
          for ct in order]
colors = [ct_palette.get(ct, '#888888') for ct in order]

fig, ax = plt.subplots(figsize=(13, 6), facecolor='white')
bp = ax.boxplot(data, labels=order, showfliers=False,
                patch_artist=True,
                medianprops=dict(color='black', lw=2))
for patch, col in zip(bp['boxes'], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)

ax.set_yscale('log')
ax.set_ylabel('RWR score (log scale)', fontsize=11)
ax.set_title(f'RWR score by cell type — seeds: {SEED_CT}\n'
             f'higher = closer to {SEED_CT} in network space',
             fontsize=11, fontweight='bold')
plt.xticks(rotation=40, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

out = os.path.join(OUT_DIR, f"{SEED_LABEL}_rwr_by_celltype.png")
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print("Done.")