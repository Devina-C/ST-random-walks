#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings
import logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
path = "/scratch/users/k22026807/masters/project/spatial_discovery/"
os.chdir(path)
os.makedirs(path+'/figures', exist_ok=True)
import matplotlib as mpl
from matplotlib.colors import ListedColormap
#from sbf import visualise_crop, load_sd
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import glob
from natsort import natsorted
import ast
import gseapy
import squidpy as sq
from tools import SVG, spagcn_svg, spagcn_domain, cellphoneDB_ccc #, mebocost_ccc
from tools import cellcharter_domain, tangram_imputation, liana_ccc, nichecompass_ccc
from tools import plot_space, plot_area, plot_abundance, EA_top_terms, plot_EA
from tools import spateo_similar_gene_pattern
import tifffile as tiff
import xarray as xr
import dask.array as da
from tabulate import tabulate
from PIL import Image
from skimage.transform import estimate_transform, warp, AffineTransform
import spatialdata as sd
import networkx as nx
import json
from shapely.geometry import Point, Polygon as ShapelyPolygon

mpl.rcParams['savefig.facecolor'] = 'white'
sc.set_figure_params(transparent=False, facecolor='white', dpi=300)

custom_palette = {
    # Myeloid lineage — strong warm orange-red
    "Myeloid cell": "#e6550d",  # vivid burnt orange

    # T cells — deep blue-purple
    "T cell": "#5b5bd6",  # saturated indigo-blue

    # NK cells — warm earthy brown-red
    "NK cell": "#a63603",  # strong reddish brown

    # B cells — vibrant violet
    "B cell": "#984ea3",  # deep purple

    # Dendritic cells — teal with higher saturation
    "Plasmacytoid dendritic cell": "#20b2aa",  # vivid teal

    # Stromal cells — golden & sky-blue for contrast
    "Fibroblast": "#d8b365",  # golden sand
    "Pericyte": "#67a9cf",    # medium sky blue

    # Endothelial cells — bright lime-green
    "Endothelial cell": "#66c2a5",  # saturated green-teal

    # Epithelial lineage — neutral grey with depth
    "Epithelial cell": "#636363",  # darker neutral grey

    # Others — strong accents for special types
    "Megakaryocyte": "#fb9a99",  # bright pink
    "Mast cell": "#ffd92f",      # saturated yellow
    "Malignant cell": "#999999"  # medium grey for neutral tone
}


# ================================================
# BC prime
# ================================================

sdata = sd.read_zarr("../xenium_output/BC_prime.zarr")
adata_ct = sc.read("../celltyping/celltype_output/BC_prime/refined_annotations.h5ad")

with open('/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson') as f:
    roi = json.load(f)
roi_coords = roi['features'][0]['geometry']['coordinates'][0]
polygon = ShapelyPolygon(roi_coords)
roi_mask = np.array([polygon.contains(Point(x, y)) for x, y in adata_ct.obsm['spatial']])
adata_ct = adata_ct[roi_mask].copy()
print(f"ROI cells: {adata_ct.shape[0]}")

try:
    adata_ct.obs['x_global_px'] = adata_ct.obsm['spatial'][:,0]
    adata_ct.obs['y_global_px'] = adata_ct.obsm['spatial'][:,1]
except KeyError:
    adata_ct.obs['x_global_px'] = adata_ct.obsm['X_spatial'][:,0]
    adata_ct.obs['y_global_px'] = adata_ct.obsm['X_spatial'][:,1]

print(f"Number of nodes (cells): {adata_ct.shape[0]}")
print(f"Number of node features (genes): {adata_ct.shape[1]}")

# ================================================
# Define colours palette for cell-types
# ================================================
cell_types = sorted(adata_ct.obs['cell_type'].unique())
color_dict = custom_palette
colormap = ListedColormap(list(custom_palette.values()), name="cell_types")

# ================================================
# Analysis cell space displaying
# ================================================
plot_space(adata_ct, custom_palette)

# ================================================
# Analysis cell size
# ================================================
plot_area(adata_ct, color_dict=custom_palette)

# ================================================
# Analysis cell abundance
# ================================================
plot_abundance(adata_ct, color_dict=custom_palette)

# ================================================
# DEG and Enrichment analysis
# ================================================
sc.tl.rank_genes_groups(adata_ct, groupby='cell_type', method='t-test')
de_results = pd.DataFrame(adata_ct.uns['rank_genes_groups']['names'])

cell_type = 'T cell'
top_10_genes = de_results[:100][cell_type]
top_10 = list(top_10_genes)
sc.pl.dotplot(adata_ct, var_names=top_10, groupby='cell_type',
              dot_max=0.8, show=True, save='top10')

# GO_Biological_Process_2023, KEGG_2021_Human
database = 'WikiPathways_2024_Human'
enr_res = gseapy.enrichr(gene_list=list(de_results[:1000][cell_type]),
                         organism='Human', gene_sets=database, cutoff=0.5)
results_df = enr_res.res2d.copy()
results_df.sort_values(by='Adjusted P-value', inplace=True)
plot_EA(results_df, database)

# Get top 10 terms
top_terms = results_df.head(10).copy()
EA_top_terms(top_terms, database)

# ========================
# Neighborhood & Centrality Analysis
# ========================
#print("Calculating action radius...")
#mean_area = adata_ct.obs['cell_area'].mean()
#R = np.sqrt(mean_area / np.pi)
action_radius = 50 
print(f"Action radius set to: {action_radius}")

#build spatial graph
sq.gr.spatial_neighbors(adata_ct, coord_type="generic", radius=action_radius)

# centrality 
# build networkx graph from the spatial connectivity matrix squidpy already made
g = nx.from_scipy_sparse_array(adata_ct.obsp['spatial_connectivities'])

# degree centrality (local)
degree_cent = nx.degree_centrality(g)
# clustering coefficient (local)
clustering = nx.clustering(g)

adata_ct.obs['degree_centrality'] = pd.Series(degree_cent).values
adata_ct.obs['clustering_coefficient'] = pd.Series(clustering).values

# average per cell type for plotting
cent_df = adata_ct.obs.groupby('cell_type')[['degree_centrality', 'clustering_coefficient']].mean()

# plot
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
cent_df['degree_centrality'].plot(kind='bar', ax=axes[0], color=[custom_palette.get(ct, 'grey') for ct in cent_df.index])
axes[0].set_title('Mean Degree Centrality by Cell Type')
axes[0].set_ylabel('Degree Centrality')
axes[0].tick_params(axis='x', rotation=45)

cent_df['clustering_coefficient'].plot(kind='bar', ax=axes[1], color=[custom_palette.get(ct, 'grey') for ct in cent_df.index])
axes[1].set_title('Mean Clustering Coefficient by Cell Type')
axes[1].set_ylabel('Clustering Coefficient')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/centrality_score.png', facecolor='white', dpi=300)
plt.close()

# ========================
# Neighborhood Enrichment and colocalisation
# ========================
sq.gr.nhood_enrichment(adata_ct, cluster_key="cell_type")
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
sq.pl.nhood_enrichment(adata_ct, cluster_key="cell_type", figsize=(8, 8),
                       title="Neighborhood enrichment", ax=ax[0], palette=colormap)
adata_plot = sc.pp.subsample(adata_ct, fraction=0.2, copy=True)
sq.pl.spatial_scatter(adata_plot, color="cell_type", shape=None, size=1,
                      ax=ax[1], dpi=300, palette=colormap)

plt.savefig('figures/neighborhood_enrichment.png', facecolor='white',
            format='png', dpi=300)

sq.gr.interaction_matrix(adata_ct, cluster_key="cell_type")
sq.pl.interaction_matrix(adata_ct, cluster_key="cell_type",
                         method="average", save='interaction_matrix.png', format='png', dpi=300, facecolor='white')

adata_sub = sc.pp.subsample(adata_ct, fraction=0.1, copy=True)
sq.gr.spatial_neighbors(adata_sub, coord_type="generic", radius=action_radius)
sq.gr.co_occurrence(adata_sub, cluster_key="cell_type", interval=10, n_splits=50)
sq.pl.co_occurrence(adata_sub, cluster_key="cell_type",
                    clusters=cell_type, save='co_occurrence.png')

# ================================================
# Ripley’s statistics
# ================================================
mode = "L"
sq.gr.ripley(adata_sub, cluster_key="cell_type", mode=mode, max_dist=50)
sq.pl.ripley(adata_sub, cluster_key="cell_type", mode=mode, save='ripley')

# ================================================
# Spatial Variable Gene (SVG)
# ================================================
adata_ct.layers['counts'] = adata_ct.X.copy()
try:
    moran_set = SVG(adata_ct, method='Moran')
    print(f"Moran SVGs: {len(moran_set)}")
except Exception as e:
    print(f"Moran failed: {e}")
    moran_set = set()

try:
    spatialDE_set = SVG(adata_ct, method='SpatialDE')
    print(f"SpatialDE SVGs: {len(spatialDE_set)}")
except Exception as e:
    print(f"SpatialDE failed: {e}")
    spatialDE_set = set()

try:
    spanve_set = SVG(adata_ct, method='Spanve')
    print(f"Spanve SVGs: {len(spanve_set)}")
except Exception as e:
    print(f"Spanve failed: {e}")
    spanve_set = set()

try:
    spagft_set = SVG(adata_ct, method='SpaGFT')
    print(f"SpaGFT SVGs: {len(spagft_set)}")
except Exception as e:
    print(f"SpaGFT failed: {e}")
    spagft_set = set()

# spateo_set, _ = SVG(adata_ct, method='spateo')
# save the new zarr
#sdata.write("BC_prime_SVG_updated.zarr")

print(spagft_set & moran_set & spatialDE_set)
completed = [s for s in [moran_set, spanve_set, spatialDE_set, spagft_set] if len(s) > 0]
if len(completed) >= 2:
    consensus_svg = list(set.intersection(*completed))
else:
    consensus_svg = list(completed[0]) if completed else []

pd.Series(consensus_svg).to_csv('figures/consensus_SVGs.csv', index=False)

# visualise top SVGs spatially
if consensus_svg:
    sq.pl.spatial_scatter(adata_ct, color=consensus_svg[:6], shape=None, save='consensus_SVG_spatial.png')
else:
    print("No consensus SVGs found, skipping spatial scatter.")


# report SVG results at different stringency levels
print("Moran SVGs:", sorted(moran_set)[:20])
print("Spanve SVGs:", sorted(spanve_set)[:20])
print("SpaGFT SVGs:", sorted(spagft_set)[:20])
print("Moran ∩ Spanve:", moran_set & spanve_set)
print("Moran ∩ SpaGFT:", moran_set & spagft_set)
print("Spanve ∩ SpaGFT:", spanve_set & spagft_set)
print(f"All 4 methods: {len(spagft_set & moran_set & spatialDE_set & spanve_set)}")
print(f"Any 3 methods: {len((spagft_set & moran_set & spatialDE_set) | (spagft_set & moran_set & spanve_set) | (spagft_set & spatialDE_set & spanve_set) | (moran_set & spatialDE_set & spanve_set))}")
print(f"Moran + SpatialDE (classic pair): {len(moran_set & spatialDE_set)}")

#sim_MUC1 = spateo_similar_gene_pattern(adata_ct, 'MUC1')
#sim_EPCAM = spateo_similar_gene_pattern(adata_ct, 'EPCAM')

# ========================
# Overlay Gene Expression on Morphology
# ========================
print("Plotting gene expression overlays...")

gene_sets = [
    (["EPCAM", "MKI67", "PDGFRB"], 'genes.png'),
    (["MUC1", "GABBR2", "EGLN3"], 'genes_simMUC1.png'),
    (["EPCAM", "S100A1", "TRPS1"], 'genes_simEPCAM.png'),
    (["EPCAM", "CTNND1"], 'genes_.png')
]

for target_genes, filename in gene_sets:
    valid_genes = [g for g in target_genes if g in adata_ct.var_names]
    if len(valid_genes) == 0:
        print(f"Skipping {filename} - none of these genes are in your dataset.")
        continue
    try:
        fig, ax = plt.subplots(1, len(valid_genes), figsize=(7 * len(valid_genes), 7))
        if len(valid_genes) == 1: 
            ax = [ax] 
            
        for compt, name in enumerate(valid_genes):
            sdata.pl.render_images("morphology_focus").pl.render_shapes(
                "cell_circles", color=name, table_name="table", use_raw=False, size=5
            ).pl.show(
                title=f"{name} expression over Morphology image",
                coordinate_systems="global", ax=ax[compt])
                
        plt.savefig(f'figures/{filename}', format='png', dpi=300,  bbox_inches='tight', facecolor='white')
        plt.close()
    except Exception as e:
        print(f"Gene overlay failed for {filename}: {e}")
        
# ================================================
# Niche/Domain identification
# ================================================
# SpaGCN (can do more with histology, https://www.sc-best-practices.org/spatial/domains.html)
# subset to crop region for domain identification
adata_crop = adata_ct.copy()
print(f"SpaGCN/CellCharter cells: {adata_crop.shape[0]}")

try:
    df = spagcn_svg(adata_crop)
    spagcn_domain(adata_crop, num_cluster=7)
except Exception as e:
    print(f"SpaGCN failed: {e}")

# CellCharter
try:
    cellcharter_domain(adata_crop)
except Exception as e:
    print(f"CellCharter failed: {e}")
# ================================================
# Gene imputation with the scAtlas and the bulk
# ================================================
# Tangram
#adata_sc = sc.read_h5ad("/Users/k2481276/Documents/METABRIC/breast_cancer.h5ad")
#xenium_table = pd.read_csv('gene_list_10X.csv')
#tg_map, adata_ensembl = tangram_imputation(adata_ct, adata_sc, xenium_table)

# ================================================
# Cell-cell communications
# ================================================
# liana
#source = 'Fibroblast'
#target = 'Myeloid cell'
#liana_ccc(adata_ct, source, target)

# squidpy (CellPhoneDB) also incude omnipath data
try:
    res = sq.gr.ligrec(
        adata_ct, 
        n_perms=100, 
        cluster_key="cell_type", 
        copy=True,
        use_raw=False, 
        transmitter_params={"categories": "ligand"}, 
        receiver_params={"categories": "receptor"})
    sq.pl.ligrec(res, 
        alpha=0.005, 
        remove_empty_interactions=True,
        save='squidpy_global_discovery.png'
        )
except Exception as e:
    print(f"ligrec failed: {e}")
# sq.pl.ligrec(res, source_groups=source, target_groups=target, alpha=0.005, 
#              remove_empty_interactions=True, save='squidpy_ccc')

# CellPhoneDB (from repo)
#genes_list = ["CXCL9", "FCGR2A"]
#cellphoneDB_ccc(adata_ct, source, target, genes_list)

# Mebocost
#mebo_obj, commu_res = mebocost_ccc(adata_ct)

# NicheCompass
#os.makedirs("nichecompass_data", exist_ok=True)

# dummy column 
adata_ct.obs['region'] = 'BC_prime_slide'

#nc = nichecompass_ccc(species = "human",n_neighbors = 4, conv_layer_encoder = "gcnconv",
#                 active_gp_thresh_ratio = 0.01, n_epochs = 400, lr = 0.001, 
#                cell_type_key = "cell_type", sample_key = "region", spot_size = 5, 
#               latent_leiden_resolution = 0.4, model_folder_path = "nichecompass_data/model",
#                 figure_folder_path = "nichecompass_data/figures")

#try:
#    print("Training NicheCompass...")
#    nc.training(adata_ct)
#    nc.plot()

#    print("Calculating enriched gene programs...")
#    enriched_gp_summary_df, gp_summary_df, df = nc.ccc()

#    enriched_gp_summary_df.to_csv("nichecompass_data/enriched_gp_summary.csv")

#    top_5_gps = enriched_gp_summary_df.head(5).index.tolist()

#    print(f"Plotting the top 5 most significant GPs: {top_5_gps}")

#    for gp in top_5_gps:
#        try:
#            nc.plot_ccc(gp_name=gp)
#        except Exception as e:
#            print(f"Warning: could not plot GP '{gp}'. Error: {e}")

#except Exception as e:
#    print(f"NicheCompass failed to run. Error: {e}")

# ================================================
# Networks analysis
# ================================================

# ================================================
# DeepSpot
# ================================================

# ================================================
# TDA with cell-type
# ================================================

# ================================================
# MuSpan
# ================================================

# ================================================
# BENTO
# ================================================

# ================================================
# SpaGFT
# ================================================

# ================================================
# MESA
# ================================================
