#### stLearn ####
# Spatial transcriptomics analysis using stLearn
# Covers: SME normalisation, SME clustering, PSTS trajectory,
#         LR/CCC analysis, and spatial visualisation


import os
import warnings
import logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scanpy as sc        # retained ONLY for sc.tl.paga (PSTS dependency)
import anndata as ad
import stlearn as st
import json
from shapely.geometry import Point, Polygon as ShapelyPolygon

mpl.rcParams['savefig.facecolor'] = 'white'

path = "/scratch/users/k22026807/masters/project/benchmarking/stlearn/"
os.chdir(path)
os.makedirs('figures', exist_ok=True)

# ─────────────────────────────────────────────
# Colour palette (consistent with other scripts)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
adata = ad.read_h5ad(
    "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
)
adata.var_names_make_unique()

# ROI subset
with open('/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson') as f:
    roi = json.load(f)
roi_coords = roi['features'][0]['geometry']['coordinates'][0]
polygon = ShapelyPolygon(roi_coords)
roi_mask = np.array([polygon.contains(Point(x, y)) for x, y in adata.obsm['spatial']])
adata = adata[roi_mask].copy()
print(f"ROI cells: {adata.shape[0]}")

# stLearn requires imagecol/imagerow in obs for spatial plotters
adata.obs['imagecol'] = adata.obsm['spatial'][:, 0].astype(float)
adata.obs['imagerow'] = adata.obsm['spatial'][:, 1].astype(float)

# stLearn requires a scalefactors dict; set to 1.0 for micron-space Xenium data
if 'spatial' not in adata.uns:
    adata.uns['spatial'] = {}
if 'BC_prime' not in adata.uns['spatial']:
    adata.uns['spatial']['BC_prime'] = {
        'scalefactors': {
            'spot_diameter_fullres':     10.0,
            'tissue_hires_scalef':        1.0,
            'fiducial_diameter_fullres':  1.0,
            'tissue_lowres_scalef':       1.0,
        }
    }

# ─────────────────────────────────────────────
# 2. PREPROCESSING  (st.pp)
# ─────────────────────────────────────────────
print("Preprocessing...")
st.pp.filter_genes(adata, min_cells=3)
st.pp.normalize_total(adata)
st.pp.log1p(adata)
st.pp.scale(adata)
st.pp.reduce_dim(adata, n_comps=50)       # PCA via st.pp

# ─────────────────────────────────────────────
# 3. SME NORMALISATION  (st.pp)
#    Key stLearn contribution: smooths gene expression
#    using the spatial graph of physical neighbours
# ─────────────────────────────────────────────
print("Running SME normalisation...")
st.pp.neighbours(adata, n_neighbors=15, use_rep='X_pca')

st.pp.SME_normalize(adata, use_data='raw', weights='physical_distance',
                    platform='Xenium')
print("SME normalisation complete.")

# Re-run PCA on the SME-normalised matrix so clustering uses it
st.pp.reduce_dim(adata, n_comps=50, use_data='SME')

# ─────────────────────────────────────────────
# 4. SME CLUSTERING  (st.tl)
#    Leiden clustering on the SME-corrected PCA embedding
# ─────────────────────────────────────────────
print("Running SME clustering...")
st.pp.neighbours(adata, n_neighbors=15, use_rep='X_pca_SME')
st.tl.clustering.SME_clustering(adata, use_data='SME', method='leiden')

# Identify the cluster column created
cluster_key = 'X_pca_SME_leiden'
if cluster_key not in adata.obs.columns:
    candidates = [c for c in adata.obs.columns
                  if 'leiden' in c.lower() or 'SME' in c]
    cluster_key = candidates[0] if candidates else cluster_key
    print(f"Using cluster key: {cluster_key}")

adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
print(f"SME clusters found: {adata.obs[cluster_key].nunique()}")

# ─────────────────────────────────────────────
# 5. UMAP  (st.tl / st.pl)
# ─────────────────────────────────────────────
print("Computing UMAP...")
st.tl.umap(adata)

fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
st.pl.umap(adata, color=cluster_key, ax=ax, show=False,
           title='SME Clusters (UMAP)')
plt.tight_layout()
plt.savefig('figures/umap_sme_clusters.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
st.pl.umap(adata, color='cell_type', ax=ax, show=False,
           title='Cell Type (UMAP)',
           palette=list(custom_palette.values()))
plt.tight_layout()
plt.savefig('figures/umap_celltype.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ─────────────────────────────────────────────
# 6. SPATIAL CLUSTER PLOTS  (st.pl)
# ─────────────────────────────────────────────
print("Plotting SME clusters spatially...")

fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
st.pl.cluster_plot(adata, use_label=cluster_key, ax=ax, show=False,
                   title='SME Clusters (spatial)')
plt.tight_layout()
plt.savefig('figures/sme_clusters_spatial.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
st.pl.cluster_plot(adata, use_label='cell_type', ax=ax, show=False,
                   title='Cell Types (spatial)',
                   list_clusters=list(custom_palette.keys()),
                   colour=list(custom_palette.values()))
plt.tight_layout()
plt.savefig('figures/celltype_spatial.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ─────────────────────────────────────────────
# 7. CLUSTER–CELLTYPE OVERLAP
#    How SME clusters map onto annotated cell types
# ─────────────────────────────────────────────
print("Computing cluster–celltype overlap...")
overlap = pd.crosstab(adata.obs[cluster_key], adata.obs['cell_type'],
                      normalize='index')
overlap.to_csv('figures/cluster_celltype_overlap.csv')

fig, ax = plt.subplots(
    figsize=(max(10, len(overlap.columns)), max(6, len(overlap) * 0.5)),
    facecolor='white'
)
im = ax.imshow(overlap.values, aspect='auto', cmap='Blues')
ax.set_xticks(range(len(overlap.columns)))
ax.set_xticklabels(overlap.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(overlap.index)))
ax.set_yticklabels(overlap.index, fontsize=8)
plt.colorbar(im, ax=ax, label='Fraction of SME cluster')
ax.set_title('SME Cluster → Cell Type Composition', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/cluster_celltype_overlap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ─────────────────────────────────────────────
# 8. LR / CCI ANALYSIS  (st.tl.cci / st.pl)
#    Permutation-based spatial co-expression testing
#    for LR pairs within 200 um (paracrine range)
# ─────────────────────────────────────────────
print("Loading LR database...")
lrs = st.tl.cci.load_lrs(['connectomeDB2020_lit'], species='human')
print(f"LR pairs loaded: {len(lrs)}")

print("Running spatial LR scoring...")
st.tl.cci.run(adata, lrs,
              min_spots=5,
              distance=200,        # paracrine range, consistent with commot/spatialdm
              n_pairs=1000,
              n_cpus=10,
              save_path='figures/')

lr_summary = adata.uns['lr_summary'].copy()
lr_summary_sig = lr_summary[lr_summary['p_adj'] < 0.05].sort_values(
    'lr_scores', ascending=False)
lr_summary.to_csv('figures/lr_pair_results_full.csv')
lr_summary_sig.to_csv('figures/lr_pair_results_significant.csv')
print(f"Total LR pairs tested:    {len(lr_summary)}")
print(f"Significant (FDR < 0.05): {len(lr_summary_sig)}")
print("Top 10 pairs:\n", lr_summary_sig.head(10)[['lr_scores', 'p_adj']].to_string())

top_lr_pairs = lr_summary_sig.head(6).index.tolist()

# ── LR pair ranking plot (st.pl) ──
print("Plotting LR pair rankings...")
try:
    fig, ax = plt.subplots(
        figsize=(8, max(5, len(lr_summary_sig.head(20)) * 0.4)),
        facecolor='white'
    )
    st.pl.lr_summary(adata, n_top=20, ax=ax, show=False)
    plt.tight_layout()
    plt.savefig('figures/lr_pair_rankings.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
except Exception as e:
    print(f"LR ranking plot failed: {e}")

# ── Spatial maps for top LR pairs (st.pl) ──
print("Plotting spatial LR score maps...")
for lr_pair in top_lr_pairs:
    try:
        fig, ax = plt.subplots(figsize=(9, 8), facecolor='white')
        st.pl.lr_result_plot(adata, use_lr=lr_pair, ax=ax, show=False,
                             title=f'LR Score: {lr_pair}')
        plt.tight_layout()
        plt.savefig(f'figures/lr_spatial_{lr_pair.replace("/", "_")}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    except Exception as e:
        print(f"LR spatial plot failed for {lr_pair}: {e}")

# ─────────────────────────────────────────────
# 9. CELL-TYPE-SPECIFIC CCI  (st.tl.cci.run_cci / st.pl)
#    Tests each LR pair between every sender–receiver
#    cell type pair using permutation testing
# ─────────────────────────────────────────────
print("Running cell-type-specific CCI...")
st.tl.cci.run_cci(
    adata,
    'cell_type',
    min_spots=3,
    distance=200,
    n_pairs=1000,
    n_cpus=10,
    save_path='figures/'
)
print("CCI complete.")

# ── CCI heatmap across all cell type pairs (st.pl) ──
print("Plotting CCI heatmaps...")
try:
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    st.pl.cci_map(adata, use_label='cell_type', ax=ax, show=False,
                  title='Cell-Type CCI Map (all LR pairs)')
    plt.tight_layout()
    plt.savefig('figures/cci_celltype_heatmap.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
except Exception as e:
    print(f"CCI map failed: {e}")

# ── Per-LR-pair CCI plots for top pairs (st.pl) ──
for lr_pair in top_lr_pairs:
    try:
        fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
        st.pl.cci_map(adata, use_label='cell_type', use_lr=lr_pair,
                      ax=ax, show=False, title=f'CCI Map: {lr_pair}')
        plt.tight_layout()
        plt.savefig(f'figures/cci_map_{lr_pair.replace("/", "_")}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    except Exception as e:
        print(f"CCI map failed for {lr_pair}: {e}")

# Save raw CCI results table
try:
    cci_results = adata.uns.get('per_lr_results', {})
    if cci_results:
        all_cci = []
        for lr_pair, df in cci_results.items():
            df = df.copy()
            df['lr_pair'] = lr_pair
            all_cci.append(df)
        cci_combined = pd.concat(all_cci, ignore_index=True)
        cci_combined.to_csv('figures/cci_celltype_results.csv', index=False)
        print(f"CCI results saved: {len(cci_combined)} rows")
except Exception as e:
    print(f"CCI results export failed: {e}")

# ─────────────────────────────────────────────
# 10. PSTS TRAJECTORY  (st.tl.PSTS / st.pl)
#     Pseudo-time Spatial Trajectory: anchors pseudotime
#     in physical space via PAGA + spatial graph.
#
#     NOTE: sc.tl.paga is called here because st.tl.PSTS
#     requires a pre-computed PAGA graph in adata.uns['paga'].
#     This is a hard internal dependency of stLearn's PSTS
#     and cannot be replaced with a stLearn-native call.
# ─────────────────────────────────────────────
print("Running PSTS trajectory analysis...")
try:
    # Required pre-step: PAGA (stLearn hard dependency)
    sc.tl.paga(adata, groups=cluster_key)

    # Root cluster: most Epithelial- or Malignant-enriched SME cluster
    for ct in ['Epithelial cell', 'Malignant cell']:
        if ct in overlap.columns:
            root_cluster = str(overlap[ct].idxmax())
            print(f"PSTS root cluster ({ct}): {root_cluster}")
            break
    else:
        root_cluster = str(overlap.index[0])
        print(f"PSTS root cluster (fallback): {root_cluster}")

    st.tl.PSTS(adata,
               use_label=cluster_key,
               pseudo_time_output='pseudotime',
               n_neighbors=15,
               root_cluster=root_cluster)

    # Spatial pseudotime plot (st.pl)
    fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
    st.pl.cluster_plot(adata, use_label='pseudotime', ax=ax, show=False,
                       title='PSTS Pseudotime (spatial)')
    plt.tight_layout()
    plt.savefig('figures/psts_pseudotime_spatial.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

    # UMAP pseudotime (st.pl)
    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    st.pl.umap(adata, color='pseudotime', ax=ax, show=False,
               title='PSTS Pseudotime (UMAP)', cmap='viridis')
    plt.tight_layout()
    plt.savefig('figures/psts_pseudotime_umap.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

    # Pseudotime distribution per cell type (st.pl)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    st.pl.violin(adata, keys='pseudotime', groupby='cell_type',
                 ax=ax, show=False, rotation=45,
                 palette=list(custom_palette.values()))
    ax.set_title('PSTS Pseudotime Distribution by Cell Type',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/psts_pseudotime_per_celltype.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

    adata.obs[['pseudotime']].to_csv('figures/psts_pseudotime_values.csv')
    print("PSTS complete.")

except Exception as e:
    print(f"PSTS failed: {e}")

# ─────────────────────────────────────────────
# 11. GENE EXPRESSION SPATIAL PLOTS  (st.pl)
#     Visualise key marker genes in physical space
# ─────────────────────────────────────────────
print("Plotting marker gene spatial distributions...")
marker_genes = ['EPCAM', 'MKI67', 'PDGFRB', 'MUC1', 'CXCL9', 'FCGR2A']
valid_markers = [g for g in marker_genes if g in adata.var_names]

for gene in valid_markers:
    try:
        fig, ax = plt.subplots(figsize=(9, 8), facecolor='white')
        st.pl.gene_plot(adata, gene_symbols=gene, ax=ax, show=False,
                        title=f'{gene} expression (spatial)')
        plt.tight_layout()
        plt.savefig(f'figures/gene_spatial_{gene}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close()
    except Exception as e:
        print(f"Gene plot failed for {gene}: {e}")

# ─────────────────────────────────────────────
# 12. SAVE RESULTS
# ─────────────────────────────────────────────
print("Saving final AnnData...")

for key in list(adata.obsm.keys()):
    val = adata.obsm[key]
    if not isinstance(val, (np.ndarray, pd.DataFrame)):
        try:
            adata.obsm[key] = np.array(val)
        except Exception:
            del adata.obsm[key]
            print(f"Dropped non-serialisable obsm key: {key}")

adata.write_h5ad('BC_prime_stlearn.h5ad')
print("Finished.")