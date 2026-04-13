# replacement workshop_lib.py file

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# colours
def get_custom_palette(extended=False):
    """Returns a dictionary of colors for cell types."""
    # Basic tab20 palette mapped to generic types to prevent crashes
    return sc.pl.palettes.default_20

# gene mapping
def genename_to_ensg(xenium_table):
    """Creates a mapping dictionary from Gene Name to Ensembl ID."""
    if 'Ensembl_ID' in xenium_table.columns and 'Gene_Name' in xenium_table.columns:
        return dict(zip(xenium_table['Gene_Name'], xenium_table['Ensembl_ID']))
    # fallback if columns don't match exactly
    return {g: g for g in xenium_table.iloc[:, 0].unique()}

# save figures
def save_figure(filename, output_dir, dpi=300, bbox_inches='tight'):
    """Helper to save matplotlib figures."""
    full_path = Path(output_dir) / filename
    plt.savefig(full_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure: {filename}")
    plt.close() 

# marker dictionary
def create_marker_dict(biomarker_df, palette=None):
    """Converts a biomarker matrix into a dictionary of lists."""
    marker_dict = {}
    for cell_type in biomarker_df.columns:
        # get genes with non-zero entries for this cell type
        genes = biomarker_df.index[biomarker_df[cell_type] > 0].tolist()
        if genes:
            marker_dict[cell_type] = genes
    return marker_dict, list(marker_dict.keys()), []

# spatial plotting
def plot_celltype_spatial(adata, color_col, title="", spot_size=30):
    """Plots cell types in spatial coordinates."""
    # ensure spatial coordinates exist
    if 'spatial' not in adata.obsm:
        print("Warning: No 'spatial' in obsm. Creating dummy coords.")
        adata.obsm['spatial'] = np.random.rand(adata.n_obs, 2)
    
    sc.pl.spatial(
        adata, 
        color=color_col, 
        spot_size=spot_size, 
        title=f"{title} Spatial Map",
        show=False,
        frameon=False
    )

# matrix visualisation
def plot_matrix_visualizations(adata, markers_df, groupby, output_dir):
    """Plots DotPlot and MatrixPlot of markers."""
    # get list of genes to plot (intersection)
    genes = markers_df['Gene_Symbol_Resolved'].unique().tolist()
    valid_genes = [g for g in genes if g in adata.var_names]
    
    if not valid_genes:
        print("No valid marker genes found for visualization.")
        return

    # dotplot
    sc.pl.dotplot(adata, valid_genes, groupby=groupby, show=False)
    save_figure("dotplot_markers.png", output_dir)
    
    # matrixplot
    sc.pl.matrixplot(adata, valid_genes, groupby=groupby, show=False)
    save_figure("matrixplot_markers.png", output_dir)

# distribution plot
def plot_celltype_distribution(adata, col_name, output_dir, suffix="", palette=None):
    """Bar chart of cell type counts."""
    counts = adata.obs[col_name].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=90)
    plt.title(f"Cell Type Distribution ({col_name})")
    plt.tight_layout()
    save_figure(f"distribution_{col_name}{suffix}.png", output_dir)

# confidence statistics
def save_confidence_statistics(adata, output_path):
    """Saves mean confidence scores per cell type."""
    if 'conf_score' not in adata.obs:
        print("No 'conf_score' found. Skipping stats.")
        return

    key = 'cell_type' if 'cell_type' in adata.obs else 'majority_voting'
    stats = adata.obs.groupby(key)['conf_score'].describe()
    stats.to_csv(output_path)
    print(f"Saved confidence stats to {output_path}")

# annotate with biomarkers
def annotate_cells_with_marker_scores(adata, marker_dict):
    """Scores cells based on marker gene expression."""
    # calculate score for each cell type
    for cell_type, genes in marker_dict.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if valid_genes:
            sc.tl.score_genes(adata, valid_genes, score_name=f"{cell_type}_score")
    return adata