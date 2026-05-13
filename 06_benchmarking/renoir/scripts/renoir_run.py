### RENOIR CELL-CELL COMMUNICATION ANALYSIS ###
# Xenium single-cell data — ligand-target activity mapping
# Steps:
#   0. Prepare databases (LR pairs, LT pairs, cell type proportions)
#   1. Load + ROI subset
#   2. Compute neighbourhood scores
#   3. Downstream analysis (domains, pathways, DE)
#   4. Ligand ranking per domain
#   5. Sankey plot
#   6. Save

import os
import warnings
import logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import pickle
import json
import pyreadr
import scipy.sparse as sp
from shapely.geometry import Point, Polygon as ShapelyPolygon
import Renoir
from Renoir import downstream as rd

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
path = "/scratch/users/k22026807/masters/project/benchmarking/renoir/"
os.chdir(path)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('data',    exist_ok=True)
os.makedirs('logs',    exist_ok=True)

ADATA_PATH   = "/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad"
GEOJSON_PATH = "/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson"
LR_PAIRS_PATH     = "natmi_lr_pairs.csv"
NICHENET_RDS_PATH = "nichenet_lt_matrix.rds"
MSIG_PATH         = "msigdb/msig_human_WP_H_KEGG_new.csv"

# Prepared file paths (computed once, reloaded on rerun)
LT_PAIRS_PATH  = "data/lt_pairs.csv"
CT_PROP_PATH   = "data/celltype_proportions.csv"
ST_OUT_PATH    = "data/ST_roi.h5ad"
SC_OUT_PATH    = "data/SC_roi.h5ad"
NB_SCORES_PATH = "results/neighborhood_scores.h5ad"
EXPINS_PATH    = "data/expins.pkl"


N_TOP_TARGETS = 25   # top NicheNet targets per ligand
RADIUS        = 200   # paracrine signalling radius in µm
LEIDEN_RES    = 0.6   # communication domain resolution

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
# STEP 0A: PREPARE LT PAIRS FROM NICHENET
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 0A: Preparing LT pairs from NicheNet matrix")
print("=" * 60)

if not os.path.exists(LT_PAIRS_PATH):
    lr_df = pd.read_csv(LR_PAIRS_PATH)
    print(f"  LR pairs loaded: {len(lr_df)}")

    print("  Loading NicheNet RDS matrix (250MB)...")
    result = pyreadr.read_r(NICHENET_RDS_PATH)
    # Matrix is targets(rows) x ligands(columns)
    lt_matrix = result[None]
    print(f"  NicheNet matrix shape: {lt_matrix.shape} (targets x ligands)")

    # Load Xenium gene panel
    print("  Loading Xenium gene panel...")
    adata_tmp = ad.read_h5ad(ADATA_PATH)
    xenium_genes = set(adata_tmp.var_names)
    # Also try hyphenated versions since gene names may differ
    xenium_genes_hyph = set(g.replace('_', '-') for g in xenium_genes)
    xenium_genes_all  = xenium_genes | xenium_genes_hyph
    del adata_tmp

    known_ligands = set(lr_df['ligand'].str.strip())

    lt_pairs = []
    print(f"  Extracting top {N_TOP_TARGETS} targets per ligand...")
    for ligand in lt_matrix.columns:
        if ligand not in known_ligands:
            continue
        if ligand not in xenium_genes_all:
            continue
        scores = lt_matrix[ligand].sort_values(ascending=False)
        top_targets = [
            t for t in scores.index
            if t in xenium_genes_all and scores[t] > 0
        ][:N_TOP_TARGETS]
        for target in top_targets:
            lt_pairs.append({'ligand': ligand, 'target': target})

    lt_df = pd.DataFrame(lt_pairs).drop_duplicates()
    lt_df.to_csv(LT_PAIRS_PATH, index=False)
    print(f"  LT pairs saved: {len(lt_df)} pairs")
    print(f"  Unique ligands: {lt_df['ligand'].nunique()}")
    print(f"  Unique targets: {lt_df['target'].nunique()}")
else:
    lt_df = pd.read_csv(LT_PAIRS_PATH)
    print(f"  LT pairs loaded: {len(lt_df)} pairs")

# ─────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA + ROI SUBSET
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Loading data + ROI subset")
print("=" * 60)

if not os.path.exists(ST_OUT_PATH):
    adata = ad.read_h5ad(ADATA_PATH)
    adata.var_names_make_unique()
    print(f"  Full dataset: {adata.shape[0]:,} cells, {adata.shape[1]} genes")

    with open(GEOJSON_PATH) as f:
        roi = json.load(f)
    roi_coords = roi['features'][0]['geometry']['coordinates'][0]
    polygon = ShapelyPolygon(roi_coords)
    roi_mask = np.array([
        polygon.contains(Point(x, y))
        for x, y in adata.obsm['spatial']
    ])
    adata = adata[roi_mask].copy()
    adata.obs['cell_type'] = adata.obs['cell_type'].cat.remove_unused_categories()
    print(f"  ROI subset: {adata.shape[0]:,} cells")

    # Renoir requires obs column named exactly 'celltype'
    adata.obs['celltype'] = adata.obs['cell_type'].astype(str)

    # Renoir uses array_row/array_col for radius-based graph
    adata.obs['array_row'] = adata.obsm['spatial'][:, 0].astype(float)
    adata.obs['array_col'] = adata.obsm['spatial'][:, 1].astype(float)

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    adata.write_h5ad(ST_OUT_PATH)
    adata.write_h5ad(SC_OUT_PATH)
    print(f"  Saved ST: {ST_OUT_PATH}")
    print(f"  Saved SC: {SC_OUT_PATH}")
else:
    adata = ad.read_h5ad(ST_OUT_PATH)
    print(f"  Loaded ROI data: {adata.shape[0]:,} cells")

# ─────────────────────────────────────────────────────────────────
# STEP 0B: PREPARE CELL TYPE PROPORTIONS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 0B: Preparing cell type proportions")
print("=" * 60)

if not os.path.exists(CT_PROP_PATH):
    # One-hot encoding for single-cell data
    ct_onehot = pd.get_dummies(adata.obs['celltype'])
    ct_onehot.index = adata.obs_names
    ct_onehot = ct_onehot.astype(float)
    ct_onehot.to_csv(CT_PROP_PATH)
    print(f"  Saved: {ct_onehot.shape} → {CT_PROP_PATH}")
    print(f"  Cell types: {list(ct_onehot.columns)}")
else:
    ct_onehot = pd.read_csv(CT_PROP_PATH, index_col=0)
    print(f"  Loaded cell type proportions: {ct_onehot.shape}")

# ─────────────────────────────────────────────────────────────────
# STEP 0C: PRE-COMPUTE EXPINS (cell-type-specific expression)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 0C: Pre-computing cell-type-specific expression")
print("=" * 60)

if not os.path.exists(EXPINS_PATH):
    import scipy.sparse as sp_sci

    ST_tmp = sc.read_h5ad(ST_OUT_PATH)
    ct_tmp = pd.read_csv(CT_PROP_PATH, index_col=0)
    ct_tmp = ct_tmp.loc[ST_tmp.obs_names, :]
    celltypes_list = ct_tmp.columns.tolist()
    obs_names_list = ST_tmp.obs_names.tolist()

    X = ST_tmp.X
    expr = X.toarray().astype(np.float64) if hasattr(X, 'toarray') \
           else np.array(X).astype(np.float64)
    ct_arr     = ct_tmp.values.astype(np.float64)
    genes_list = ST_tmp.var_names.tolist()

    # Filter to only genes in LT pairs to save memory and disk
    lt_df_tmp  = pd.read_csv(LT_PAIRS_PATH)
    lt_genes   = set(lt_df_tmp['ligand'].tolist() + lt_df_tmp['target'].tolist())
    keep_idx   = [i for i, g in enumerate(genes_list) if g in lt_genes]
    keep_genes = [genes_list[i] for i in keep_idx]
    expr       = expr[:, keep_idx]
    print(f"  Filtered to {len(keep_genes)} LT-relevant genes (from {len(genes_list)})")
    print(f"  Building expins dict...")

    expins = {}
    expins['cells']     = obs_names_list
    expins['celltypes'] = celltypes_list
    for i, gene in enumerate(keep_genes):
        expins[gene] = sp_sci.csr_matrix(expr[:, i, np.newaxis] * ct_arr)
        if i % 200 == 0:
            print(f"  {i}/{len(keep_genes)} genes processed")

    with open(EXPINS_PATH, 'wb') as f:
        pickle.dump(expins, f)
    print(f"  Saved: {EXPINS_PATH}")
    del expr, ct_arr, ST_tmp, ct_tmp
else:
    print(f"  expins already exists: {EXPINS_PATH}")

# ─────────────────────────────────────────────────────────────────
# STEP 2: COMPUTE NEIGHBOURHOOD SCORES
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Computing neighbourhood scores")
print("=" * 60)

if not os.path.exists(NB_SCORES_PATH):
    print(f"  Radius: {RADIUS} µm")
    print(f"  LT pairs: {len(lt_df)}")
    print(f"  This step may take 2-5 hours for 53k cells...")

    neighborhood_scores = Renoir.compute_neighborhood_scores(
        SC_path=SC_OUT_PATH,
        ST_path=ST_OUT_PATH,
        pairs_path=LT_PAIRS_PATH,
        ligand_receptor_path=LR_PAIRS_PATH,
        celltype_proportions_path=CT_PROP_PATH,
        expins_path=EXPINS_PATH,
        single_cell=True,
        use_radius=True,
        radius=RADIUS,
        return_adata=True,
    )

    neighborhood_scores.write_h5ad(NB_SCORES_PATH)
    print(f"  Saved: {NB_SCORES_PATH}")
    print(f"  Shape: {neighborhood_scores.shape} (cells x LT pairs)")
else:
    neighborhood_scores = sc.read_h5ad(NB_SCORES_PATH)
    print(f"  Loaded: {neighborhood_scores.shape} (cells x LT pairs)")

neighborhood_scores.raw = neighborhood_scores.copy()

# ─────────────────────────────────────────────────────────────────
# STEP 3A: SPATIAL MAPS OF TOP LT PAIRS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3A: Spatial maps of top LT pairs")
print("=" * 60)

try:
    X = neighborhood_scores.X
    if hasattr(X, 'toarray'):
        mean_scores = np.array(X.mean(axis=0)).flatten()
    else:
        mean_scores = np.array(X.mean(axis=0)).flatten()

    mean_scores = pd.Series(mean_scores, index=neighborhood_scores.var_names)
    mean_scores = mean_scores.sort_values(ascending=False)
    top_lt_pairs = mean_scores.head(6).index.tolist()
    print(f"  Top LT pairs by mean activity: {top_lt_pairs}")

    xy = adata.obsm['spatial']

    for lt_pair in top_lt_pairs:
        try:
            fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
            vals = neighborhood_scores[:, lt_pair].X
            if hasattr(vals, 'toarray'):
                vals = vals.toarray().flatten()
            else:
                vals = np.array(vals).flatten()
            sc_plot = ax.scatter(
                xy[:, 0], xy[:, 1],
                c=vals, cmap='YlOrRd', s=0.3, alpha=0.8, rasterized=True
            )
            plt.colorbar(sc_plot, ax=ax, shrink=0.6, label='Activity score')
            ax.set_title(f'LT activity: {lt_pair}', fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
            plt.tight_layout()
            safe = lt_pair.replace(':', '_').replace('/', '_')
            plt.savefig(f'figures/lt_spatial_{safe}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  Spatial plot failed for {lt_pair}: {e}")

    print(f"  Saved spatial maps for top {len(top_lt_pairs)} LT pairs")
except Exception as e:
    print(f"  Step 3A failed: {e}")
    top_lt_pairs = neighborhood_scores.var_names[:6].tolist()

# ─────────────────────────────────────────────────────────────────
# STEP 3B: PATHWAY CLUSTERS + COMMUNICATION DOMAINS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3B: Pathway clusters + communication domains")
print("=" * 60)

pcs = None
neighbscore_copy = neighborhood_scores

try:
    if MSIG_PATH and os.path.exists(MSIG_PATH):
        msig = Renoir.get_msig('custom', path=MSIG_PATH)
    else:
        msig = Renoir.get_msig('human')
    print(f"  MSigDB pathways loaded")

    pathways = Renoir.create_cluster(
        neighborhood_scores,
        msig,
        method=None,
        pathway_thresh=10,
        restrict_to_KHW=True,
    )
    print(f"  Pathway clusters: {len(pathways)}")

    try:
        all_clusters = Renoir.create_cluster(
            neighborhood_scores,
            msig,
            method='dhc',
            ltclust_thresh=20,
            restrict_to_KHW=True,
        )
        de_novo = {k: v for k, v in all_clusters.items() if k.startswith('cluster_')}
        print(f"  De novo clusters: {len(de_novo)}")
        combined_clusters = all_clusters
    except Exception as e:
        print(f"  De novo clustering failed: {e} — using pathway clusters only")
        combined_clusters = pathways

    de_novo_only = {k: v for k, v in combined_clusters.items()
                if k.startswith('cluster_')}
    print(f"  Using {len(de_novo_only)} de novo clusters for downstream analysis")

    neighbscore_copy, pcs = Renoir.downstream_analysis(
        neighborhood_scores,
        ltpair_clusters=de_novo_only,
        resolution=LEIDEN_RES,
        n_markers=20,
        n_top=20,
        pdf_path=None,
        return_cluster=True,
        return_pcs=True,
    )

    n_domains = neighbscore_copy.obs['leiden'].nunique()
    print(f"  Communication domains: {n_domains}")

    neighborhood_scores.obs['leiden'] = neighbscore_copy.obs['leiden']
    neighborhood_scores.uns = neighbscore_copy.uns

    neighbscore_copy.obs[['leiden']].to_csv('results/communication_domains.csv')
    print("  Saved: results/communication_domains.csv")

except Exception as e:
    print(f"  Pathway/domain analysis failed: {e}")

# ─────────────────────────────────────────────────────────────────
# STEP 3C: SPATIAL MAP OF COMMUNICATION DOMAINS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3C: Spatial map of communication domains")
print("=" * 60)

try:
    if 'leiden' in neighbscore_copy.obs.columns:
        n_domains = neighbscore_copy.obs['leiden'].nunique()
        cmap = matplotlib.colormaps.get_cmap('tab10')

        fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
        xy = adata.obsm['spatial']
        for i, domain in enumerate(sorted(neighbscore_copy.obs['leiden'].unique())):
            mask = neighbscore_copy.obs['leiden'] == domain
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                s=0.3, c=[cmap(int(domain) % 10)],
                label=f'Domain {domain}', alpha=0.7, rasterized=True
            )
        ax.set_aspect('equal')
        ax.set_title(f'Communication domains (n={n_domains})',
                     fontsize=12, fontweight='bold')
        ax.legend(markerscale=6, fontsize=7,
                  bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('figures/communication_domains_spatial.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/communication_domains_spatial.png")
except Exception as e:
    print(f"  Domain spatial plot failed: {e}")

# ─────────────────────────────────────────────────────────────────
# STEP 3D: PATHWAY ACTIVITY SPATIAL MAPS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3D: Pathway activity spatial maps")
print("=" * 60)

try:
    if pcs is not None and pcs.shape[1] > 0:
        pcs_X = pcs.X.toarray() if hasattr(pcs.X, 'toarray') else np.array(pcs.X)
        pathway_var = pd.Series(
            np.var(pcs_X, axis=0),
            index=pcs.var_names
        ).sort_values(ascending=False)
        top_pathways = pathway_var.head(6).index.tolist()
        print(f"  Top pathways by spatial variance: {top_pathways}")

        n_pw = len(top_pathways)
        ncols = 3
        nrows = int(np.ceil(n_pw / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 6, nrows * 5),
                                  facecolor='white')
        axes = axes.flatten()
        xy = adata.obsm['spatial']

        for i, pathway in enumerate(top_pathways):
            ax = axes[i]
            vals = pcs_X[:, pcs.var_names.get_loc(pathway)]
            sc_plot = ax.scatter(
                xy[:, 0], xy[:, 1],
                c=vals, cmap='YlGnBu_r', s=0.3, alpha=0.8, rasterized=True
            )
            plt.colorbar(sc_plot, ax=ax, shrink=0.6)
            ax.set_title(pathway.replace('_', ' ')[:40], fontsize=7, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Pathway activity (spatial)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/pathway_activity_spatial.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/pathway_activity_spatial.png")
except Exception as e:
    print(f"  Pathway spatial plots failed: {e}")

# ─────────────────────────────────────────────────────────────────
# STEP 3E: DIFFERENTIAL LT PAIRS PER DOMAIN
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3E: Differential LT pairs per domain")
print("=" * 60)

try:
    if 'leiden' in neighborhood_scores.obs.columns:
        n_domains = neighborhood_scores.obs['leiden'].nunique()
        sc.tl.rank_genes_groups(
            neighborhood_scores, 'leiden', method='wilcoxon')

        fig, ax = plt.subplots(
            figsize=(max(12, n_domains * 2), 10), facecolor='white')
        sc.pl.rank_genes_groups_heatmap(
            neighborhood_scores,
            n_genes=8,
            groupby='leiden',
            show_gene_labels=True,
            min_logfoldchange=0.3,
            dendrogram=False,
            swap_axes=True,
            standard_scale='var',
            cmap='viridis',
            ax=ax,
            show=False,
        )
        plt.tight_layout()
        plt.savefig('figures/de_lt_pairs_heatmap.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/de_lt_pairs_heatmap.png")

        de_names  = neighborhood_scores.uns['rank_genes_groups']['names']
        de_scores = neighborhood_scores.uns['rank_genes_groups']['scores']
        de_rows = []
        for domain in de_names.dtype.names:
            for gene, score in zip(de_names[domain][:20],
                                   de_scores[domain][:20]):
                de_rows.append({'domain': domain,
                                'lt_pair': gene,
                                'score': score})
        pd.DataFrame(de_rows).to_csv('results/de_lt_pairs.csv', index=False)
        print("  Saved: results/de_lt_pairs.csv")

        top_lt_pairs = [
            de_names[domain][0]
            for domain in de_names.dtype.names
            if len(de_names[domain]) > 0
        ]
        top_lt_pairs = list(dict.fromkeys(top_lt_pairs))[:6]

except Exception as e:
    print(f"  DE analysis failed: {e}")

# ─────────────────────────────────────────────────────────────────
# STEP 4: LIGAND RANKING PER DOMAIN
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Ligand ranking per domain")
print("=" * 60)

try:
    if 'leiden' in neighborhood_scores.obs.columns:
        lr_df = pd.read_csv(LR_PAIRS_PATH)
        for domain_id in sorted(neighborhood_scores.obs['leiden'].unique()):
            try:
                fig = Renoir.ligand_ranking(
                    neighborhood_scores,
                    neighborhood_scores,
                    neighborhood_scores,
                    lr_df,
                    LT_PAIRS_PATH,
                    str(domain_id),
                    receptor_exp=0.05,
                    markers={'top': 50},
                    domain_celltypes=['top', 5],
                    celltype_colors={'auto': True},
                )
                fig.set_size_inches(20, 10)
                plt.tight_layout()
                plt.savefig(
                    f'figures/ligand_ranking_domain_{domain_id}.png',
                    dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved: figures/ligand_ranking_domain_{domain_id}.png")
            except Exception as e:
                print(f"  Ligand ranking failed for domain {domain_id}: {e}")
except Exception as e:
    print(f"  Ligand ranking step failed: {e}")

# ─────────────────────────────────────────────────────────────────
# STEP 5: SANKEY PLOT
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Sankey plot")
print("=" * 60)

try:
    if ('leiden' in neighborhood_scores.obs.columns and
            'rank_genes_groups' in neighborhood_scores.uns):

        de_names = neighborhood_scores.uns['rank_genes_groups']['names']
        top_per_domain = {
            group: list(de_names[group][:5])
            for group in de_names.dtype.names
        }
        ltpairs_for_sankey = sorted({
            pair for pairs in top_per_domain.values() for pair in pairs
        })
        print(f"  Sankey LT pairs: {len(ltpairs_for_sankey)}")

        rd.sankeyPlot(
            neighborhood_scores,
            neighborhood_scores,
            ltpairs_for_sankey,
            n_celltype=5,
            clusters='All',
            title='Ligand-target: ligand → target → cell type → domain',
            labelsize=8,
            labelcolor='#000000',
        )
        plt.savefig('figures/sankey_lt_domains.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/sankey_lt_domains.png")
except Exception as e:
    print(f"  Sankey plot failed: {e}")

# ─────────────────────────────────────────────────────────────────
# STEP 6: SAVE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving")
print("=" * 60)

neighborhood_scores.write_h5ad(NB_SCORES_PATH)
print(f"  Saved: {NB_SCORES_PATH}")

if pcs is not None:
    pcs.write_h5ad('results/pathway_pcs.h5ad')
    print("  Saved: results/pathway_pcs.h5ad")

print("\n" + "=" * 60)
print("FINISHED")
print("=" * 60)
print("Outputs:")
print("  data/                              — prepared input files")
print("  results/neighborhood_scores.h5ad  — core output")
print("  results/communication_domains.csv — domain labels")
print("  results/de_lt_pairs.csv           — DE LT pairs per domain")
print("  figures/lt_spatial_*              — spatial LT activity maps")
print("  figures/communication_domains_spatial.png")
print("  figures/pathway_activity_spatial.png")
print("  figures/de_lt_pairs_heatmap.png")
print("  figures/ligand_ranking_domain_*.png")
print("  figures/sankey_lt_domains.png")
