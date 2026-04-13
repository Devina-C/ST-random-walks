#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings
import logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
path = "/scratch/users/k22026807/masters/project/spatial_discovery/"
os.chdir(path)
os.makedirs(path+'/figures/squidpy_subsample', exist_ok=True)
from matplotlib.colors import ListedColormap
#from sbf import visualise_crop, load_sd
import pandas as pd
import numpy as np
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
print("Calculating dynamic action radius...")
mean_area = adata_ct.obs['cell_area'].mean()
R = np.sqrt(mean_area / np.pi)
action_radius = 10 * R
print(f"Action radius set to: {action_radius:.2f}")

print("Building spatial graph using biological radius...")
sq.gr.spatial_neighbors(adata_ct, coord_type="generic", radius=action_radius)

print("Calculating centrality scores...")
sq.gr.centrality_scores(adata_ct, cluster_key="cell_type")
sq.pl.centrality_scores(adata_ct, cluster_key="cell_type",
                        figsize=(16, 5), save='squidpy_subsample/centrality_score.png')


# ================================================
# Subsample for GLOBAL spatial statistics
# ================================================

print("Subsampling for global spatial statistics...")
adata_sub = sc.pp.subsample(adata_ct, n_obs=100_000, copy=True, random_state=0)

print(f"Original cells: {adata_ct.n_obs}")
print(f"Subsampled cells: {adata_sub.n_obs}")

print("Filtering empty genes from subset...")
sc.pp.filter_genes(adata_sub, min_cells=1)

print("Rebuilding spatial graph for the subset...")
sq.gr.spatial_neighbors(adata_sub, coord_type="generic", radius=action_radius)

# ========================
# Neighborhood Enrichment and colocalisation
# ========================
sq.gr.nhood_enrichment(adata_ct, cluster_key="cell_type")
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
sq.pl.nhood_enrichment(adata_ct, cluster_key="cell_type", figsize=(8, 8),
                       title="Neighborhood enrichment", ax=ax[0], palette=colormap)
sq.pl.spatial_scatter(adata_ct, color="cell_type", shape=None, size=10,
                      ax=ax[1], dpi=600, palette=colormap)
plt.savefig('figures/squidpy_subsample/neighborhood_enrichment_call_type.png',
            format='png', dpi=600)

sq.gr.interaction_matrix(adata_ct, cluster_key="cell_type")
sq.pl.interaction_matrix(adata_ct, cluster_key="cell_type",
                         method="average", save='squidpy_subsample/interaction_matrix')

sq.gr.co_occurrence(adata_sub, cluster_key="cell_type")
sq.pl.co_occurrence(adata_sub, cluster_key="cell_type",
                    clusters=cell_type, save='squidpy_subsample/co_occurrence')

# ================================================
# Ripley’s statistics
# ================================================
mode = "L"
sq.gr.ripley(adata_sub, cluster_key="cell_type", mode=mode)
sq.pl.ripley(adata_sub, cluster_key="cell_type", mode=mode, save='squidpy_subsample/ripley')

# ================================================
# Spatial Variable Gene (SVG)
# ================================================

if 'counts' not in adata_sub.layers:
    adata_sub.layers['counts'] = adata_sub.X.copy()

moran_set = SVG(adata_sub, method='Moran')
spatialDE_set = SVG(adata_sub, method='SpatialDE')
spanve_set = SVG(adata_sub, method='Spanve')
spagft_set = SVG(adata_sub, method='SpaGFT')
# spateo_set, _ = SVG(adata_ct, method='spateo')

# save the new zarr
sdata.write("BC_prime_squidpy_subsample.zarr")

print(spagft_set & moran_set & spatialDE_set)

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
    # 1. Only keep the genes that actually exist in your dataset
    valid_genes = [g for g in target_genes if g in adata_ct.var_names]
    
    # 2. If none of them exist, skip to the next image
    if len(valid_genes) == 0:
        print(f"Skipping {filename} - none of these genes are in your dataset.")
        continue
        
    # 3. Create exactly the right number of subplots
    fig, ax = plt.subplots(1, len(valid_genes), figsize=(7 * len(valid_genes), 7))
    if len(valid_genes) == 1: 
        ax = [ax] # Fixes an indexing bug if only 1 gene is found
        
    # 4. Plot them safely!
    for compt, name in enumerate(valid_genes):
        sdata.pl.render_images("morphology_focus").pl.render_shapes(
            "cell_circles", color=name, table_name="table", use_raw=False, size=5
        ).pl.show(
            title=f"{name} expression over Morphology image",
            coordinate_systems="global", ax=ax[compt])
            
    plt.savefig(f'figures/squidpy_subsample/{filename}', format='png', dpi=600)
    plt.close()
        
# ================================================
# Niche/Domain identification
# ================================================
# SpaGCN (can do more with histology, https://www.sc-best-practices.org/spatial/domains.html)
df = spagcn_svg(adata_ct)
spagcn_domain(adata_ct, num_cluster=7)

# CellCharter
cellcharter_domain(adata_ct)

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
res = sq.gr.ligrec(
    adata_ct, 
    n_perms=1000, 
    cluster_key="cell_type", 
    copy=True,
    use_raw=False, 
    transmitter_params={"categories": "ligand"}, 
    receiver_params={"categories": "receptor"})

sq.pl.ligrec(res, 
    alpha=0.005, 
    remove_empty_interactions=True,
    save='squidpy_subsample/squidpy_global_discovery.png'
    )
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
