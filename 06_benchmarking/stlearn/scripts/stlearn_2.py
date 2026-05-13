### STLEARN CELL-CELL INTERACTION ANALYSIS ###

# Xenium single-cell data → grid-based CCC analysis
# Steps:
#   1. Load + ROI subset
#   2. QC plots
#   3. Preprocessing (filter, normalise, PCA, Leiden)
#   4. UMAP visualisation
#   5. Spatial cluster plots
#   6. Gridding + grid validation plots
#   7. LR permutation test + plots
#   8. Cell-type CCI + plots
#   9. Save

import os
import warnings
import logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scanpy as sc
import anndata as ad
import stlearn as st
import json
from shapely.geometry import Point, Polygon as ShapelyPolygon

mpl.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

path = "/scratch/users/k22026807/masters/project/benchmarking/stlearn/"
os.chdir(path)
os.makedirs('results', exist_ok=True)
os.makedirs('results/qc',       exist_ok=True)
os.makedirs('results/umap',     exist_ok=True)
os.makedirs('results/spatial',  exist_ok=True)
os.makedirs('results/grid',     exist_ok=True)
os.makedirs('results/lr',       exist_ok=True)
os.makedirs('results/cci',      exist_ok=True)

custom_palette = {
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
    "Malignant cell":              "#999999",
}

# ─────────────────────────────────────────────────────────────────
# 1. LOAD DATA + ROI SUBSET
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

adata = ad.read_h5ad(
    "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
)
adata.var_names_make_unique()
print(f"Full dataset: {adata.shape[0]:,} cells, {adata.shape[1]} genes")

with open('/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson') as f:
    roi = json.load(f)
roi_coords = roi['features'][0]['geometry']['coordinates'][0]
polygon = ShapelyPolygon(roi_coords)
roi_mask = np.array([polygon.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
adata = adata[roi_mask].copy()
adata.obs['cell_type'] = adata.obs['cell_type'].cat.remove_unused_categories()
print(f"ROI subset: {adata.shape[0]:,} cells")

# stLearn spatial metadata requirements
adata.obs['imagecol'] = adata.obsm['spatial'][:, 0].astype(float)
adata.obs['imagerow'] = adata.obsm['spatial'][:, 1].astype(float)
adata.uns['spatial'] = {'BC_prime': {'scalefactors': {
    'spot_diameter_fullres':     10.0,
    'tissue_hires_scalef':        1.0,
    'fiducial_diameter_fullres':  1.0,
    'tissue_lowres_scalef':       1.0,
}}}

# ─────────────────────────────────────────────────────────────────
# 2. QC PLOTS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: QC plots")
print("=" * 60)

sc.pp.calculate_qc_metrics(adata, inplace=True)

# 2A — Transcript and gene count distributions
fig, axes = plt.subplots(1, 4, figsize=(20, 4), facecolor='white')
fig.suptitle('Cell-Level QC Metrics (post-ROI subset)', fontsize=14, fontweight='bold')

sns.histplot(adata.obs['total_counts'],      bins=60, ax=axes[0], color='steelblue')
axes[0].axvline(10, color='red', ls='--', label='min_counts=10')
axes[0].set_title('Total transcripts per cell')
axes[0].set_xlabel('Total counts')
axes[0].legend(fontsize=8)

sns.histplot(adata.obs['n_genes_by_counts'], bins=60, ax=axes[1], color='steelblue')
axes[1].set_title('Unique genes per cell')
axes[1].set_xlabel('N genes')

if 'cell_area' in adata.obs.columns:
    sns.histplot(adata.obs['cell_area'], bins=60, ax=axes[2], color='steelblue')
    axes[2].set_title('Cell area')
    axes[2].set_xlabel('Area (µm²)')
else:
    axes[2].text(0.5, 0.5, 'cell_area not available', ha='center', va='center',
                 transform=axes[2].transAxes)
    axes[2].set_title('Cell area')

# Scatter: total_counts vs n_genes coloured by cell type
pos_ct = adata.obs['cell_type'].values
for ct, col in custom_palette.items():
    mask = pos_ct == ct
    if mask.sum() > 0:
        axes[3].scatter(
            adata.obs.loc[mask, 'total_counts'],
            adata.obs.loc[mask, 'n_genes_by_counts'],
            s=0.3, alpha=0.4, c=col, label=ct, rasterized=True)
axes[3].set_xlabel('Total counts')
axes[3].set_ylabel('N genes')
axes[3].set_title('Counts vs Genes by cell type')
axes[3].legend(markerscale=6, fontsize=5, bbox_to_anchor=(1.01, 1), loc='upper left')

plt.tight_layout()
plt.savefig('results/qc/qc_distributions.png', dpi=500, bbox_inches='tight')
plt.close()
print("  Saved: results/qc/qc_distributions.png")

# 2B — Cell type composition bar chart
ct_counts = adata.obs['cell_type'].value_counts()
fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
colours = [custom_palette.get(ct, 'grey') for ct in ct_counts.index]
ax.bar(range(len(ct_counts)), ct_counts.values, color=colours)
ax.set_xticks(range(len(ct_counts)))
ax.set_xticklabels(ct_counts.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Number of cells')
ax.set_title('Cell type distribution (ROI)', fontsize=12, fontweight='bold')
for i, (ct, n) in enumerate(ct_counts.items()):
    ax.text(i, n + 50, f'{n/len(adata)*100:.1f}%', ha='center', fontsize=7)
plt.tight_layout()
plt.savefig('results/qc/celltype_distribution.png', dpi=500, bbox_inches='tight')
plt.close()
print("  Saved: results/qc/celltype_distribution.png")

# 2C — Spatial overview coloured by cell type (mirrors notebook cell 28)
xy = adata.obsm['spatial']
fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
for ct, col in custom_palette.items():
    mask = adata.obs['cell_type'] == ct
    if mask.sum() > 0:
        ax.scatter(xy[mask, 0], xy[mask, 1], s=0.3, c=col,
                   label=ct, alpha=0.7, rasterized=True)
ax.set_aspect('equal')
ax.set_title('Cell types — spatial overview', fontsize=12, fontweight='bold')
ax.set_xlabel('X (µm)')
ax.set_ylabel('Y (µm)')
ax.legend(markerscale=8, fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/qc/spatial_celltypes_overview.png', dpi=500, bbox_inches='tight')
plt.close()
print("  Saved: results/qc/spatial_celltypes_overview.png")

# Print summary stats
print(f"\n  Summary statistics:")
print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars}")
print(f"  Total counts  — mean: {adata.obs['total_counts'].mean():.1f}, "
      f"median: {adata.obs['total_counts'].median():.1f}")
print(f"  Genes/cell    — mean: {adata.obs['n_genes_by_counts'].mean():.1f}, "
      f"median: {adata.obs['n_genes_by_counts'].median():.1f}")

# ─────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Preprocessing")
print("=" * 60)

n_before = adata.n_obs
sc.pp.filter_cells(adata, min_counts=10)
sc.pp.filter_genes(adata, min_cells=3)
print(f"  Filtered: {n_before - adata.n_obs} cells removed (min_counts=10)")
print(f"  Remaining: {adata.n_obs:,} cells, {adata.n_vars} genes")

adata.var_names = adata.var_names.str.replace('_', '-', regex=False)
print(f"  Gene names sanitised (underscores → hyphens)")

adata.raw = adata
print("  Raw counts stored in adata.raw")

# Normalise only — no log1p per stLearn Xenium tutorial note
# (we want genes to be more separated for hotspot detection)
#st.pp.normalize_total(adata)
#print("  Normalised (no log1p — per stLearn Xenium tutorial)")

# PCA using st.em.run_pca so stLearn internals work correctly
st.em.run_pca(adata, n_comps=50, random_state=0)
print("  PCA complete (50 components)")

# Neighbours + leiden (kept for QC/reference even though CCI uses cell_type)
st.pp.neighbors(adata, n_neighbors=25, use_rep='X_pca', random_state=0)
sc.tl.leiden(adata, random_state=0)
n_leiden = adata.obs['leiden'].nunique()
print(f"  Leiden clustering: {n_leiden} clusters")

# ─────────────────────────────────────────────────────────────────
# 4. UMAP + CLUSTER VISUALISATIONS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: UMAP visualisations")
print("=" * 60)

sc.tl.umap(adata)

# 4A — UMAP coloured by leiden clusters
fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
sc.pl.umap(adata, color='leiden', ax=ax,
           title=f'leiden clusters (UMAP)  n={n_leiden}')
plt.tight_layout()
plt.savefig('results/umap/umap_leiden.png', dpi=500, bbox_inches='tight')
plt.close()

# 4B — UMAP coloured by cell type
fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
sc.pl.umap(adata, color='cell_type', ax=ax,
           title='Cell types (UMAP)',
           palette=list(custom_palette.values()))
plt.tight_layout()
plt.savefig('results/umap/umap_celltype.png', dpi=500, bbox_inches='tight')
plt.close()

# 4C — UMAP coloured by QC metrics (mirrors notebook cell 26)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
for ax, col, title in zip(axes,
    ['total_counts', 'n_genes_by_counts', 'leiden'],
    ['Total counts', 'N genes', 'leiden']):
    sc.pl.umap(adata, color=col, ax=ax, title=title)
plt.tight_layout()
plt.savefig('results/umap/umap_qc_metrics.png', dpi=500, bbox_inches='tight')
plt.close()
print("  Saved: results/umap/umap_leiden.png, umap_celltype.png, umap_qc_metrics.png")

# ─────────────────────────────────────────────────────────────────
# 5. SPATIAL PLOTS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Spatial cluster plots")
print("=" * 60)

xy = adata.obsm['spatial']

# 5A — leiden clusters spatially
fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
for label in adata.obs['leiden'].unique():
    mask = adata.obs['leiden'] == label
    ax.scatter(xy[mask, 0], xy[mask, 1], s=0.3, alpha=0.7,
               label=label, rasterized=True)
ax.set_aspect('equal')
ax.set_title('leiden clusters (spatial)')
ax.legend(markerscale=6, fontsize=6, bbox_to_anchor=(1.01,1), loc='upper left')
plt.tight_layout()
plt.savefig('results/spatial/spatial_leiden.png', dpi=500, bbox_inches='tight')
plt.close()

# 5B — Cell types spatially
fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
for ct, col in custom_palette.items():
    mask = adata.obs['cell_type'] == ct
    if mask.sum() > 0:
        ax.scatter(xy[mask, 0], xy[mask, 1], s=0.3, c=col,
                   label=ct, alpha=0.7, rasterized=True)
ax.set_aspect('equal')
ax.set_title('Cell types (spatial)')
ax.legend(markerscale=6, fontsize=6, bbox_to_anchor=(1.01,1), loc='upper left')
plt.tight_layout()
plt.savefig('results/spatial/spatial_celltype.png', dpi=500, bbox_inches='tight')
plt.close()

# 5C — leiden ↔ cell type overlap heatmap
overlap = pd.crosstab(adata.obs['leiden'], adata.obs['cell_type'], normalize='index')
overlap.to_csv('results/spatial/leiden_celltype_overlap.csv')
fig, ax = plt.subplots(
    figsize=(max(10, len(overlap.columns)), max(5, len(overlap) * 0.4)),
    facecolor='white')
sns.heatmap(overlap, cmap='Blues', ax=ax,
            cbar_kws={'label': 'Fraction of leiden cluster'})
ax.set_title('leiden cluster → cell type composition', fontsize=12, fontweight='bold')
ax.set_xlabel('Cell type')
ax.set_ylabel('leiden cluster')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('results/spatial/leiden_celltype_heatmap.png', dpi=500, bbox_inches='tight')
plt.close()

# 5D — Per-cell-type spatial density plots
n_types = len(custom_palette)
ncols = 4
nrows = int(np.ceil(n_types / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), facecolor='white')
axes = axes.flatten()
xy = adata.obsm['spatial']
for i, (ct, col) in enumerate(custom_palette.items()):
    ax = axes[i]
    ax.scatter(xy[:, 0], xy[:, 1], s=0.1, c='lightgrey', alpha=0.2, rasterized=True)
    mask = adata.obs['cell_type'] == ct
    if mask.sum() > 0:
        ax.scatter(xy[mask, 0], xy[mask, 1], s=0.5, c=col,
                   alpha=0.8, rasterized=True)
    ax.set_title(f'{ct}\n(n={mask.sum():,})', fontsize=8, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Per-cell-type spatial distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/spatial/spatial_per_celltype.png', dpi=500, bbox_inches='tight')
plt.close()
print("  Saved: spatial_leiden, spatial_celltype, leiden_celltype_heatmap, spatial_per_celltype")

# 5E — Key marker genes spatially (mirrors notebook cell 38)
marker_genes = ['EPCAM', 'CD3E', 'CD68', 'MKI67', 'PDGFRB', 'VIM']
valid_markers = [g for g in marker_genes if g in adata.var_names]
if valid_markers:
    fig, axes = plt.subplots(1, len(valid_markers),
                              figsize=(5 * len(valid_markers), 4.5), facecolor='white')
    if len(valid_markers) == 1:
        axes = [axes]
    for ax, gene in zip(axes, valid_markers):
        expr = (adata[:, gene].X.toarray().flatten()
                if hasattr(adata.X, 'toarray') else adata[:, gene].X.flatten())
        vmax = np.percentile(expr[expr > 0], 95) if (expr > 0).any() else expr.max()
        sc_plot = ax.scatter(xy[:, 0], xy[:, 1], c=expr, cmap='YlOrRd',
                             s=0.3, alpha=0.8, vmin=0, vmax=vmax, rasterized=True)
        plt.colorbar(sc_plot, ax=ax, shrink=0.6, label='Expr')
        ax.set_title(gene, fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    plt.suptitle('Marker gene expression (spatial)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/spatial/spatial_marker_genes.png', dpi=500, bbox_inches='tight')
    plt.close()
    print("  Saved: results/spatial/spatial_marker_genes.png")

# ─────────────────────────────────────────────────────────────────
# 6. GRIDDING + GRID VALIDATION PLOTS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Gridding")
print("=" * 60)

n_ = 40
print(f"  Grid resolution: {n_} x {n_} = {n_*n_:,} potential spots")
grid = st.tl.cci.grid(adata, n_row=n_, n_col=n_, use_label='cell_type')
grid_xy = np.column_stack([grid.obs['imagecol'], grid.obs['imagerow']])
print(f"  Grid spots (non-empty): {grid.shape[0]:,}")
print(f"  Grid genes: {grid.shape[1]}")

# 6A — Side-by-side: grid dominant cell type vs original cell scatter
#      (exact comparison recommended by the tutorial)
fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')

grid_xy = np.column_stack([grid.obs['imagecol'], grid.obs['imagerow']])
for ct, col in custom_palette.items():
    mask = grid.obs['cell_type'] == ct
    if mask.sum() > 0:
        axes[0].scatter(grid_xy[mask, 0], grid_xy[mask, 1],
                        s=10, c=col, label=ct, alpha=0.8, rasterized=True)
axes[0].set_aspect('equal')
axes[0].legend(markerscale=4, fontsize=6, bbox_to_anchor=(1.01,1), loc='upper left')

axes[0].set_title(f'Grid — dominant cell type per spot\n(n={grid.shape[0]:,} spots)',
                  fontsize=11, fontweight='bold')


for ct, col in custom_palette.items():
    mask = adata.obs['cell_type'] == ct
    if mask.sum() > 0:
        axes[1].scatter(xy[mask, 0], xy[mask, 1], s=0.3, c=col,
                        label=ct, alpha=0.7, rasterized=True)
axes[1].set_aspect('equal')
axes[1].legend(markerscale=6, fontsize=6, bbox_to_anchor=(1.01,1), loc='upper left')

axes[1].set_title(f'Original — single cell labels\n(n={adata.shape[0]:,} cells)',
                  fontsize=11, fontweight='bold')

plt.suptitle('Grid vs original: tissue structure comparison',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/grid/grid_vs_original_celltype.png', dpi=500, bbox_inches='tight')
plt.close()
print("  Saved: results/grid/grid_vs_original_celltype.png")

# 6B — Per-cell-type: grid proportion map vs original scatter
#      (mirrors tutorial groups loop exactly)
cell_types_present = list(adata.obs['cell_type'].cat.categories)
n_ct = len(cell_types_present)
ncols = 3
nrows = int(np.ceil(n_ct / ncols))

fig, axes = plt.subplots(nrows * 2, ncols,
                          figsize=(ncols * 6, nrows * 2 * 5),
                          facecolor='white')

for i, ct in enumerate(cell_types_present):
    col_idx = i % ncols
    row_top = (i // ncols) * 2      # proportion map row
    row_bot = row_top + 1           # original cells row

    ax_top = axes[row_top, col_idx]
    ax_bot = axes[row_bot, col_idx]
    colour = custom_palette.get(ct, 'grey')

    # Top: proportion per grid spot
    if ct in grid.uns.get('cell_type', {}):
        grid.obs['_ct_prop'] = grid.uns['cell_type'][ct].values
        vals = grid.obs['_ct_prop'].values
        sc_p = ax_top.scatter(grid_xy[:, 0], grid_xy[:, 1], c=vals,
                              cmap='Reds', s=5, vmin=0, vmax=1, rasterized=True)
        plt.colorbar(sc_p, ax=ax_top, shrink=0.5)
        ax_top.set_aspect('equal')
        ax_top.set_title(f'{ct}\nGrid proportion (max=1)', fontsize=8)
    else:
        ax_top.text(0.5, 0.5, f'{ct}\n(not in grid.uns)',
                    ha='center', va='center', transform=ax_top.transAxes, fontsize=8)

    # Bottom: original single cells
    colour = custom_palette.get(ct, 'grey')
    ax_bot.scatter(xy[:, 0], xy[:, 1], s=0.1, c='lightgrey', alpha=0.2, rasterized=True)
    mask_ct = adata.obs['cell_type'] == ct
    if mask_ct.sum() > 0:
        ax_bot.scatter(xy[mask_ct, 0], xy[mask_ct, 1], s=0.3, c=colour,
                       alpha=0.8, rasterized=True)
    ax_bot.set_aspect('equal')
    ax_bot.set_title(f'{ct}\nSingle cells', fontsize=8)

# Hide unused subplots
total_slots = nrows * ncols
for i in range(n_ct, total_slots):
    col_idx = i % ncols
    axes[(i // ncols) * 2,     col_idx].set_visible(False)
    axes[(i // ncols) * 2 + 1, col_idx].set_visible(False)

plt.suptitle('Per-cell-type: grid proportion vs original cells',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/grid/grid_per_celltype_comparison.png', dpi=500, bbox_inches='tight')
plt.close()
print("  Saved: results/grid/grid_per_celltype_comparison.png")

# 6C — Gene expression: grid vs original for key markers
#      (mirrors tutorial CXCL12 comparison)
check_genes = ['CXCL12', 'EPCAM', 'CD3E']
valid_check = [g for g in check_genes if g in grid.var_names]
if valid_check:
    fig, axes = plt.subplots(len(valid_check), 2,
                              figsize=(14, 5 * len(valid_check)), facecolor='white')
    if len(valid_check) == 1:
        axes = [axes]
    for ax_row, gene in zip(axes, valid_check):       
        if gene in grid.var_names:
            expr_g = (grid[:, gene].X.toarray().flatten()) if hasattr(grid.X, 'toarray') else grid[:, gene].X.flatten()
            ax_row[0].scatter(grid_xy[:, 0], grid_xy[:, 1], c=expr_g,
                              cmap='YlOrRd', s=3, rasterized=True)
        ax_row[0].set_aspect('equal')
        ax_row[0].set_title(f'{gene} — grid expression', fontsize=10)

        if gene in adata.var_names:
            expr_a = adata[:, gene].X.toarray().flatten() if hasattr(adata.X, 'toarray') else adata[:, gene].X.flatten()
            ax_row[1].scatter(xy[:, 0], xy[:, 1], c=expr_a,
                              cmap='YlOrRd', s=0.3, rasterized=True)
        ax_row[1].set_aspect('equal')
        ax_row[1].set_title(f'{gene} — cell expression', fontsize=10)
    plt.suptitle('Gene expression: grid vs original cells',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/grid/grid_gene_expression_comparison.png',
                dpi=500, bbox_inches='tight')
    plt.close()
    print("  Saved: results/grid/grid_gene_expression_comparison.png")

# 6D — Grid spot statistics
n_cells_per_spot = grid.obs.get('n_cells', None)
if n_cells_per_spot is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='white')
    sns.histplot(n_cells_per_spot, bins=40, ax=axes[0], color='steelblue')
    axes[0].set_title('Cells per grid spot')
    axes[0].set_xlabel('N cells')
    axes[0].axvline(n_cells_per_spot.mean(), color='red', ls='--',
                    label=f'mean={n_cells_per_spot.mean():.1f}')
    axes[0].legend()

    # Cell type entropy per spot (how mixed is each spot?)
    if 'cell_type' in grid.uns:
        ct_matrix = pd.DataFrame(grid.uns['cell_type'])
        # Shannon entropy: -sum(p * log(p+eps))
        entropy = -(ct_matrix * np.log(ct_matrix + 1e-10)).sum(axis=1)
        axes[1].scatter(grid.obs['imagecol'], grid.obs['imagerow'],
                        c=entropy, cmap='RdYlGn_r', s=3, alpha=0.8, rasterized=True)
        axes[1].set_title('Cell type entropy per spot\n(green=pure, red=mixed)')
        axes[1].set_aspect('equal')
        plt.colorbar(axes[1].collections[0], ax=axes[1], shrink=0.6, label='Entropy')
    plt.tight_layout()
    plt.savefig('results/grid/grid_spot_statistics.png', dpi=500, bbox_inches='tight')
    plt.close()
    print("  Saved: results/grid/grid_spot_statistics.png")

# ─────────────────────────────────────────────────────────────────
# 7. LR PERMUTATION TEST
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: LR permutation test")
print("=" * 60)

lrs = st.tl.cci.load_lrs(['connectomeDB2020_lit'], species='human')
print(f"  LR pairs loaded: {len(lrs)}")

print("  Running LR permutation test on grid...")
st.tl.cci.run(
    grid, lrs,
    min_spots=5,
    distance=250,
    n_pairs=1000,
    n_cpus=10,
)

st.tl.cci.adj_pvals(grid, correct_axis='spot',
                    pval_adj_cutoff=0.05, adj_method='fdr_bh')

lr_summary = grid.uns['lr_summary'].copy()
lr_sig = lr_summary[lr_summary['n_spots_sig'] > 0].sort_values(
    'n_spots_sig', ascending=False)
lr_summary.to_csv('results/lr/lr_pair_results_full.csv')
lr_sig.to_csv('results/lr/lr_pair_results_significant.csv')
print(f"  LR pairs tested: {len(lr_summary)}")
print(f"  Pairs with significant spots: {len(lr_sig)}")
if len(lr_sig) > 0:
    print(f"  Top 5:\n{lr_sig.head(5)[['n_spots_sig']].to_string()}")

top_lr_pairs = lr_sig.head(6).index.tolist()

# 7A — LR pair ranking (mirrors tutorial st.pl.lr_summary)
try:
    fig, ax = plt.subplots(figsize=(10, max(4, min(len(lr_sig), 30) * 0.35)),
                            facecolor='white')
    st.pl.lr_summary(grid, n_top=min(30, len(lr_sig)), ax=ax)
    plt.tight_layout()
    plt.savefig('results/lr/lr_pair_rankings.png', dpi=500, bbox_inches='tight')
    plt.close()
    print("  Saved: results/lr/lr_pair_rankings.png")
except Exception as e:
    print(f"  LR ranking plot failed: {e}")

# 7B — Spatial maps for top LR pairs: scores, -log10(p_adj), sig_scores
#      (mirrors tutorial best_lr stats loop exactly)
stats = ['lr_scores', '-log10(p_adjs)', 'lr_sig_scores']
# Replace the lr_result_plot loop with manual plotting:
for lr_pair in top_lr_pairs:
    try:
        lr_idx = list(grid.uns['lr_summary'].index).index(lr_pair)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
        for ax, stat in zip(axes, ['lr_scores', '-log10(p_adjs)', 'lr_sig_scores']):
            vals = grid.obsm[stat][:, lr_idx]
            sc = ax.scatter(grid.obs['imagecol'], grid.obs['imagerow'],
                           c=vals, cmap='Reds', s=10, alpha=0.8)
            plt.colorbar(sc, ax=ax, shrink=0.6)
            ax.set_title(stat, fontsize=9)
            ax.set_aspect('equal')
        plt.suptitle(f'LR pair: {lr_pair}', fontsize=11, fontweight='bold')
        plt.tight_layout()
        safe = lr_pair.replace('/', '_').replace(' ', '_')
        plt.savefig(f'results/lr/lr_spatial_{safe}.png', dpi=500, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  LR spatial plot failed for {lr_pair}: {e}")

print(f"  Saved spatial maps for {len(top_lr_pairs)} top LR pairs")

# ─────────────────────────────────────────────────────────────────
# 8. CELL-TYPE CCI
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Cell-type CCI")
print("=" * 60)

st.tl.cci.run_cci(
    grid, 'cell_type',
    min_spots=5,
    spot_mixtures=True,
    cell_prop_cutoff=0.1,
    sig_spots=True,
    n_perms=10,
    n_cpus=10,
)
print("  CCI complete")

# 8A — CCI diagnostic check: cell type frequency vs interaction count
#      (tutorial: "Should be little to no correlation")
try:
    fig = plt.figure(figsize=(16, 5), facecolor='white')
    st.pl.cci_check(grid, 'cell_type')
    plt.suptitle('CCI diagnostic: frequency vs interaction count\n'
                 '(little/no correlation = permutation controlled adequately)',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig('results/cci/cci_diagnostic_check.png', dpi=500, bbox_inches='tight')
    plt.close()
    print("  Saved: results/cci/cci_diagnostic_check.png")
except Exception as e:
    print(f"  CCI check plot failed: {e}")

# 8B — CCI heatmap across all cell type pairs
try:
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    st.pl.cci_map(grid, use_label='cell_type', ax=ax,
                  title='Cell-type CCI map (all LR pairs)')
    plt.tight_layout()
    plt.savefig('results/cci/cci_celltype_heatmap.png', dpi=500, bbox_inches='tight')
    plt.close()
    print("  Saved: results/cci/cci_celltype_heatmap.png")
except Exception as e:
    print(f"  CCI heatmap failed: {e}")

# 8C — CCI network plot (all pairs + per top LR)
#      (mirrors tutorial ccinet_plot section)
try:
    pos_net = st.pl.ccinet_plot(grid, 'cell_type',
                                 return_pos=True, min_counts=5)
    plt.suptitle('CCI network — all LR pairs', fontsize=11, fontweight='bold')
    plt.savefig('results/cci/cci_network_all.png', dpi=500, bbox_inches='tight')
    plt.close()
    print("  Saved: results/cci/cci_network_all.png")

    for lr_pair in top_lr_pairs[:3]:
        try:
            st.pl.ccinet_plot(grid, 'cell_type', lr_pair,
                              min_counts=2, figsize=(10, 7),
                              pos=pos_net)
            plt.suptitle(f'CCI network — {lr_pair}', fontsize=11)
            safe = lr_pair.replace('/', '_').replace(' ', '_')
            plt.savefig(f'results/cci/cci_network_{safe}.png',
                        dpi=500, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  CCI network for {lr_pair} failed: {e}")
except Exception as e:
    print(f"  CCI network plots failed: {e}")

# 8D — Interaction heatmaps (chord plot fallback — version-safe)
try:
    cci_key = 'per_lr_cci_cell_type'

    if cci_key in grid.uns:
        cell_types = list(grid.obs['cell_type'].cat.categories)
        n = len(cell_types)

        def make_cci_matrix(grid, cci_key, lr_pair=None, min_ints=2):
            matrix = np.zeros((n, n))
            pairs_to_use = (
                [lr_pair] if lr_pair is not None and lr_pair in grid.uns[cci_key]
                else list(grid.uns[cci_key].keys())
            )
            for pair in pairs_to_use:
                df = grid.uns[cci_key][pair]
                for i, ct_i in enumerate(cell_types):
                    for j, ct_j in enumerate(cell_types):
                        col = f"{ct_i}_{ct_j}"
                        if col in df.columns:
                            val = df[col].sum()
                            if val >= min_ints:
                                matrix[i][j] += val
            return matrix

        def save_cci_heatmap(matrix, cell_types, filepath, title):
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
            df_mat = pd.DataFrame(matrix, index=cell_types, columns=cell_types)
            sns.heatmap(df_mat, cmap='Blues', ax=ax, square=True,
                        cbar_kws={'label': 'CCI interaction count'},
                        linewidths=0.5, annot=True, fmt='.0f', annot_kws={'size': 7})
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Receiver cell type', fontsize=9)
            ax.set_ylabel('Sender cell type', fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
            plt.tight_layout()
            plt.savefig(filepath, dpi=500, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {filepath}")

        # All LR pairs combined
        matrix_all = make_cci_matrix(grid, cci_key, min_ints=2)
        save_cci_heatmap(matrix_all, cell_types,
                         'results/cci/cci_chord_all.png',
                         'CCI interactions — all LR pairs')

        # Per top LR pair
        for lr_pair in top_lr_pairs[:2]:
            matrix_lr = make_cci_matrix(grid, cci_key, lr_pair=lr_pair, min_ints=2)
            safe = lr_pair.replace('/', '_').replace(' ', '_')
            save_cci_heatmap(matrix_lr, cell_types,
                             f'results/cci/cci_chord_{safe}.png',
                             f'CCI interactions — {lr_pair}')

except Exception as e:
    print(f"  CCI interaction heatmaps failed: {e}")


# 8E — Per-LR-pair CCI map (fixed: no ax_or_none)
try:
    st.pl.lr_cci_map(
        adata=grid,
        use_label='cell_type',
        lrs=top_lr_pairs[:6],
        n_top_lrs=6,
        n_top_ccis=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig('results/cci/cci_lr_map.png', dpi=500, bbox_inches='tight')
    plt.close()
    print("  Saved: results/cci/cci_lr_map.png")
except TypeError as e:
    print(f"  lr_cci_map TypeError: {e} — trying minimal call")
    try:
        st.pl.lr_cci_map(grid, use_label='cell_type', show=False)
        plt.savefig('results/cci/cci_lr_map.png', dpi=500, bbox_inches='tight')
        plt.close()
        print("  Saved: results/cci/cci_lr_map.png (minimal call)")
    except Exception as e2:
        print(f"  lr_cci_map minimal call also failed: {e2}")
except Exception as e:
    print(f"  lr_cci_map failed: {e}")

# Save CCI results table
try:
    cci_key = 'per_lr_cci_cell_type'
    if cci_key in grid.uns:
        all_cci = []
        for lr_pair, df in grid.uns[cci_key].items():
            df = df.copy()
            df['lr_pair'] = lr_pair
            all_cci.append(df)
        if all_cci:
            pd.concat(all_cci, ignore_index=True).to_csv(
                'results/cci/cci_celltype_results.csv', index=False)
            print("  Saved: results/cci/cci_celltype_results.csv")
except Exception as e:
    print(f"  CCI results export failed: {e}")

# ─────────────────────────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Saving")
print("=" * 60)

for key in list(adata.obsm.keys()):
    val = adata.obsm[key]
    if not isinstance(val, (np.ndarray, pd.DataFrame)):
        try:
            adata.obsm[key] = np.array(val)
        except Exception:
            del adata.obsm[key]
            print(f"  Dropped non-serialisable obsm key: {key}")

if 'lrfeatures' in grid.uns:
    for key in grid.uns['lrfeatures']:
        val = grid.uns['lrfeatures'][key]
        if not isinstance(val, (str, np.ndarray, dict, list)):
            grid.uns['lrfeatures'][key] = str(val)
grid.write_h5ad('BC_prime_stlearn_grid.h5ad')
adata.write_h5ad('BC_prime_stlearn_cells.h5ad')
print("  Saved: BC_prime_stlearn_grid.h5ad")
print("  Saved: BC_prime_stlearn_cells.h5ad")

print("\n" + "=" * 60)
print("FINISHED")
print("=" * 60)
print("Outputs:")
print("  results/qc/         — QC distributions, cell type bar chart, spatial overview")
print("  results/umap/       — UMAP by leiden, cell type, QC metrics")
print("  results/spatial/    — spatial clusters, per-cell-type maps, marker genes")
print("  results/grid/       — grid vs original comparisons, gene expression, spot stats")
print("  results/lr/         — LR rankings, spatial maps for top pairs")
print("  results/cci/        — CCI heatmaps, network, chord, per-pair maps, diagnostic")
