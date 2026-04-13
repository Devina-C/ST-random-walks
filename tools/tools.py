#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tangram.utils import eval_metric
import mygene
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import squidpy as sq
import random
import torch
import os
import NaiveDE
# from mebocost import mebocost
from spagcn import calculate_adj_matrix, search_l, SpaGCN, search_radius
from spagcn import prefilter_genes, prefilter_specialgenes, refine
from spagcn import find_neighbor_clusters, rank_genes_groups, search_res
from spagft import determine_frequency_ratio, detect_svg
from spanve import Spanve
import spatialDE
import tangram as tg
from tangram.mapping_utils import pp_adatas, map_cells_to_space
from tangram.plot_utils import plot_cell_annotation_sc, plot_training_scores
from tangram.plot_utils import plot_genes_sc
from tangram.utils import project_genes, compare_spatial_geneexp
from matplotlib import gridspec
from sklearn.preprocessing import MinMaxScaler
#from nichecompass.models import NicheCompass
#from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
#                                compute_communication_gp_network,
#                                visualize_communication_gp_network,
#                                create_new_color_dict,
#                                extract_gp_dict_from_mebocost_ms_interactions,
#                                extract_gp_dict_from_nichenet_lrt_interactions,
#                                extract_gp_dict_from_omnipath_lr_interactions,
#                                filter_and_combine_gp_dict_gps_v2,
#                                generate_enriched_gp_info_plots)
import liana as li
from liana.method import cellphonedb
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
import ktplotspy as kpy
from plotnine import facet_wrap
import omnipath as op
# from mebocost import mebocost
import scvi
import cellcharter as cc
from pypath.utils import mapping
# import spateo as st
from typing import List, Tuple, Union
import multiprocessing
from tqdm import tqdm
from functools import partial
import scipy as sp
from statsmodels.stats.multitest import multipletests
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse


def palette(key = 0):
    """
    Returns a color palette based on the specified key.

    Args:
        key (int, optional): Determines the palette to return. If `key` is 0, returns a 
            custom RGBA color palette. Otherwise, returns a dictionary of standard color 
            names mapped to integer keys. Default is 0.

    Returns:
        dict: A dictionary where each key corresponds to a color. If `key` is 0, 
            the values are RGBA tuples representing specific custom colors:
            - 0: (0.604, 0.678, 0.749, 1)  # '#9AADBF'
            - 1: (0.827, 0.725, 0.624, 1)  # '#D3B99F'
            - 2: (0.757, 0.467, 0.404, 1)  # '#C17767'
            - 3: (0.427, 0.596, 0.729, 1)  # '#6D98BA'
            
            If `key` is not 0, the values are standard color names as strings:
            - 0: 'red'
            - 1: 'blue'
            - 2: 'green'
            - 3: 'orange'
            - 4: 'gold'
            - ...
            - 29: 'khaki'
    """
    
    if key == 0:
        color1 = (0.604, 0.678, 0.749, 1) # '#9AADBF'
        color2 = (0.827, 0.725, 0.624, 1) # '#D3B99F'
        color3 = (0.757, 0.467, 0.404, 1) # '#C17767'
        color4 = (0.427, 0.596, 0.729, 1) # '#6D98BA'
        colors = {0:color1 , 1:color2, 2:color3, 3:color4}
        
        return colors
    
    else:    
        add_colors = {0:'teal', 1:'orange', 2:'green', 3:'red', 4:'gold', 5:'purple', 6:'grey', 7:'pink',\
                 8:'navy', 9:'springgreen', 10:'salmon', 11:'skyblue', 12:'tan', 13:'sienna',\
                 14:'turquoise', 15:'aqua', 16:'chartreuse', 17:'crimson', 18:'fuchsia', 19:'beige',\
                 20:'yellow', 21:'blue', 22:'olivedrab', 23:'deeppink', 24:'maroon', 25:'mistyrose',\
                 26:'seagreen', 27:'darkorange', 28:'mediumpurple', 29:'khaki'}
            
        return add_colors


def plot_space(adata, color_dict):
    """
    Plots spatial cell-type distribution using a scatterplot.
    
    This function generates a spatial plot of cells based on their global 
    x and y pixel coordinates (`x_global_px`, `y_global_px`) stored in the 
    `.obs` attribute of an AnnData object. Each point is colored by its 
    corresponding cell type using a user-provided color palette.
    
    Args:
        adata (AnnData): Annotated data matrix (from the `anndata` package), 
            where `adata.obs` must contain `x_global_px`, `y_global_px`, and 
            `cell_type` columns.
        color_dict (dict): Dictionary mapping cell type labels to color values. 
            Used to define the `palette` in the seaborn scatterplot.
    
    Returns:
        None: Displays a matplotlib figure but does not return any value.
    """

    plt.figure(figsize=(6, 5))
    g = sns.scatterplot(x="x_global_px", y="y_global_px", s=20, marker='.',
                        data=adata.obs, hue='cell_type', palette=color_dict)
    sns.move_legend(g, "upper right", bbox_to_anchor=(1, 1))
    handles, labels = g.get_legend_handles_labels()
    for h in handles:
        h.set_markersize(5)
    plt.legend(
        handles, labels,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),  # adjust slightly inward
        fontsize=4,
        title='Cell Type',
        title_fontsize=6,
        ncol=1,                       # keeps it compact
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        labelspacing=0.3)
    g.invert_yaxis()
    g.set_ylabel("")
    g.set_xlabel("")
    plt.tight_layout()
    plt.savefig('figures/space.png', format='png', dpi=600)
    plt.close()


def plot_area(sample, color_dict):
    """
    Plots distribution of cell areas by cell type.

    This function generates overlaid histograms (with optional KDE curves) 
    showing the distribution of cell areas (`cell_area`) for each cell type 
    present in the sample. Each histogram is colored according to the provided 
    `color_dict`.
  
    Args:
        sample (AnnData): Annotated data matrix (from the `anndata` package), 
            where `sample.obs` must contain `cell_type` and `cell_area` columns.
        color_dict (dict): Dictionary mapping cell type labels to color values. 
            Used to color the histograms corresponding to each cell type.
  
    Returns:
        None: Displays a matplotlib figure but does not return any value.
    """

    plt.figure(figsize=(10, 6))
    unique_cell_types = sorted(sample.obs['cell_type'].unique())

    for k in unique_cell_types:
        temp = sample[sample.obs['cell_type'] == k].obs['cell_area']
        sns.histplot(temp, kde=True, stat="count", label=k,
                     alpha=0.5, color=color_dict[k])

    plt.xlim([0, 300])
    plt.legend()
    plt.savefig('figures/area.png', format='png', dpi=600)
    plt.close()


def plot_abundance(adata, color_dict):
    """
    Plots the percentage abundance of each cell type as a stacked bar chart.

    This function calculates the relative abundance (in percent) of each cell 
    type in the dataset and visualizes it as a single stacked bar. Each segment 
    of the bar represents a different cell type, colored using the provided 
    `color_dict`.

    Args:
        adata (AnnData): Annotated data matrix (from the `anndata` package), 
            where `adata.obs` must contain a `cell_type` column.
        color_dict (dict): Dictionary mapping cell type labels to color values. 
            Used to color each segment of the stacked bar.

    Returns:
        None: Displays a matplotlib figure but does not return any value.
    """

    cell_counts = adata.obs['cell_type'].value_counts().sort_index()
    percentages = cell_counts / cell_counts.sum() * 100

    # Plot setup
    colors = [color_dict[col] for col in percentages.index]
    barWidth = 0.85

    # Stacked bar: just one bar at position 0
    plt.figure(figsize=(6, 6))
    bottoms = 0

    for i, (label, value) in enumerate(percentages.items()):
        plt.bar(0, value, bottom=bottoms, color=colors[i], edgecolor='white',
                width=barWidth, label=label)
        bottoms += value

    # Customization
    plt.xticks([0], ['All cells'])
    plt.ylabel("Percentage of Cell Types")
    plt.title("Cell Type Distribution (Single Group)")
    plt.legend(title="Cell Type", loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('figures/abundance.png', format='png', dpi=600)
    plt.close()


def parse_overlap(overlap_str):
    """
    Parses a string fraction and returns its float value.

    Converts a string representing a fraction (e.g., "3/5") into a float by 
    dividing the numerator by the denominator. If the input is invalid or 
    cannot be parsed, the function returns 0.

    Args:
        overlap_str (str): A string in the format "numerator/denominator".

    Returns:
        float: The result of the division as a float. Returns 0 if parsing fails.
    """

    try:
        num, denom = map(int, str(overlap_str).split('/'))
        return num / denom
    except:
        return 0


def plot_EA(results_df, database):
    """
    Plots the top enrichment analysis (EA) terms as a horizontal bar chart.

    This function visualizes the top 10 enriched terms from an enrichment 
    analysis results DataFrame by plotting the negative log10 of the adjusted 
    p-values. The bars are ordered top-down with the most significant terms at 
    the top. The plot is saved as a PNG file named after the database.

    Args:
        results_df (pandas.DataFrame): DataFrame containing enrichment results, 
            with at least the columns `'Term'` and `'Adjusted P-value'`.
        database (str): Name of the enrichment database. Used as the plot title 
            and the filename for the saved image (e.g., `"KEGG.png"`).

    Returns:
        None: Saves the plot to disk but does not return any value.
    """

    # Plot manually for full control
    plt.figure(figsize=(12, 6))
    plt.barh(y=results_df['Term'].values[:10][::-1],  # Reverse for top-down
             width=-np.log10(results_df['Adjusted P-value'].values[:10][::-1]),
             color='salmon',
             edgecolor='black')
    plt.xlabel(r'$-\log_{10}$ (Adjusted P-value)', fontsize=12)
    plt.title(database, fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/' + database + ".png", dpi=300)
    plt.close()


def EA_top_terms(top_terms, database):
    """
    Plots a dot plot of top enrichment analysis (EA) terms.

    This function creates a dot plot summarizing the top enriched terms from 
    an enrichment analysis. Each term is represented by a dot whose x-position 
    corresponds to the statistical significance (`-log10(Adjusted P-value)`), 
    size reflects the overlap ratio, and color represents the combined score. 
    The plot is saved as a PNG file.

    Args:
        top_terms (pandas.DataFrame): DataFrame containing enrichment terms with 
            at least the following columns:
            - `'Adjusted P-value'`: The corrected p-value for each term.
            - `'Term'`: The name of each enriched term.
            - `'Overlap'`: String fractions (e.g., "5/100") indicating gene overlap.
            - `'Combined Score'`: Score summarizing enrichment significance and magnitude.
        database (str): Name of the enrichment database. Used in the plot title 
            and output filename.

    Returns:
        None: Saves the plot to disk but does not return any value.
    """

    top_terms['-log10(padj)'] = -np.log10(top_terms['Adjusted P-value'])
    top_terms['Overlap_ratio'] = top_terms['Overlap'].apply(parse_overlap)

    plt.figure(figsize=(12, 5))
    scatter = sns.scatterplot(
        data=top_terms,
        x='-log10(padj)',
        y='Term',
        size='Overlap_ratio',
        sizes=(100, 1000),
        hue='Combined Score',
        palette='Reds',
        edgecolor='black',
        legend='brief')

    plt.title(f"{database}", fontsize=14)
    plt.xlabel(r"$-\log_{10}$ (Adjusted P-value)", fontsize=12)
    plt.ylabel("Pathway", fontsize=12)

    # Improve legend layout
    h, l = scatter.get_legend_handles_labels()
    # Automatically separate size and hue legends
    scatter.legend(
        title=None,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        frameon=True)

    plt.tight_layout()
    plt.savefig("figures/GSEA_dotplot_{database}_clean.png", dpi=300)
    plt.close()


def genename_to_ensg_mygeneinfo(gene_names):
    """
    Maps gene symbols to Ensembl gene IDs using MyGene.info.

    This function queries the MyGene.info API to retrieve corresponding Ensembl 
    gene identifiers (`ENSG...`) for a list of human gene symbols.

    Args:
        gene_names (list of str): A list of gene symbols (e.g., `["TP53", "BRCA1"]`).

    Returns:
        dict: A dictionary mapping each input gene symbol to its corresponding 
        Ensembl gene ID (as a string). If a gene has no match, the value is `None`.
    """

    mg = mygene.MyGeneInfo()
    results = mg.querymany(gene_names, scopes='symbol',
                           fields='ensembl.gene', species='human', returnall=True)
    gene_to_ensg = {}

    for res in results['out']:
        gene = res['query']
        ensembl = res.get('ensembl')
        if isinstance(ensembl, list):
            gene_to_ensg[gene] = ensembl[0]['gene']
        elif isinstance(ensembl, dict):
            gene_to_ensg[gene] = ensembl['gene']
        else:
            gene_to_ensg[gene] = None

    return gene_to_ensg


def genename_to_ensg(xenium_table):
    """
    Creates a mapping from gene symbols to Ensembl gene IDs from a DataFrame.

    This function builds a dictionary that maps gene names (`gene_name`) to 
    Ensembl gene IDs (`gene_id`) using columns from a given `xenium_table` DataFrame.

    Args:
        xenium_table (pandas.DataFrame): A DataFrame containing at least the columns 
            `'gene_name'` and `'gene_id'`.

    Returns:
        dict: A dictionary where keys are gene names (str) and values are Ensembl 
        gene IDs (str).
    """

    genes = list(xenium_table['gene_name'])
    ensg = list(xenium_table['gene_id'])

    gene_to_ensg = {}

    for k in range(len(genes)):
        gene_to_ensg[genes[k]] = ensg[k]

    return gene_to_ensg


def plot_auc(df_all_genes, test_genes=None):
    """
    Plots auc curve which is used to evaluate model performance.
    
    Args:
        df_all_genes (Pandas dataframe): returned by compare_spatial_geneexp(adata_ge, adata_sp); 
        test_genes (list): list of test genes, if not given, test_genes will be set to genes where 'is_training' field is False

    Returns:
        None
    """

    test_genes = list(df_all_genes[df_all_genes['is_training'] == False].index)
    metric_dict, ((pol_xs, pol_ys), (xs, ys)) = eval_metric(
        df_all_genes, test_genes)

    plt.figure()
    plt.figure(figsize=(6, 5))

    plt.plot(pol_xs, pol_ys, c='r')
    sns.scatterplot(x=xs, y=ys, alpha=0.5, edgecolors='face',
                    size=0.5, legend=False)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect(.5)
    plt.xlabel('score')
    plt.ylabel('spatial sparsity')
    plt.tick_params(axis='both', labelsize=8)
    plt.title('Prediction on test transcriptome')

    textstr = 'auc_score={}'.format(np.round(metric_dict['auc_score'], 3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    # place a text box in upper left in axes coords
    plt.text(0.03, 0.1, textstr, fontsize=11,
             verticalalignment='top', bbox=props)
    plt.savefig("figures/tangram_auc.png", dpi=300)
    plt.close()


# function from spateo fixed to solve the following issue:
# TypeError: Setting a MultiIndex dtype to anything other than object is not supported
def bin_adata(
    adata: AnnData,
    bin_size: int = 1,
    layer: str = "spatial",
) -> AnnData:
    """Aggregate cell-based adata by bin size. Cells within a bin would be
    aggregated together as one cell.

    Args:
        adata: the input adata.
        bin_size: the size of square to bin adata.

    Returns:
        Aggregated adata.
    """
    adata = adata.copy()
    adata.obsm[layer] = (adata.obsm[layer] // bin_size).astype(np.int32)

    df = (
        pd.DataFrame(adata.X.toarray(), columns=adata.var_names)
        if issparse(adata.X)
        else pd.DataFrame(adata.X, columns=adata.var_names)
    )
    df[["x", "y"]] = adata.obsm[layer]
    
    # Group and reset index to avoid MultiIndex
    df2 = df.groupby(by=["x", "y"]).sum().reset_index()

    # Create obs names from x and y
    
    obs_names = df2.apply(lambda row: f"{row['x']}_{row['y']}", axis=1)

    # Extract expression matrix
    expr_matrix = df2.drop(columns=["x", "y"]).values
    var_names = df2.drop(columns=["x", "y"]).columns

    # Create new AnnData object
    a = AnnData(expr_matrix, var=pd.DataFrame(index=var_names))
    a.uns["__type"] = "UMI"
    a.obs_names = obs_names
    a.obsm[layer] = df2[["x", "y"]].values.astype(np.float64)

    return a


def bin_scale_adata_get_distance(
    adata: AnnData,
    bin_size: int = 1,
    bin_layer: str = "spatial",
    distance_layer: str = "spatial",
    cell_distance_method: str = "geodesic",
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
    n_neighbors: int = 30,
) -> Tuple[AnnData, csr_matrix]:
    """Bin (based on spatial information), scale adata object and calculate the distance matrix based on the specified
    method (either geodesic or euclidean).

    Args:
        adata: AnnData object.
        bin_size: Bin size for mergeing cells.
        bin_layer: Data in this layer will be binned according to the spatial information.
        distance_layer: The data of this layer would be used to calculate distance
        cell_distance_method: The method for calculating distance between two cells, either geodesic or euclidean.
        min_dis_cutoff: Cells/Bins whose min distance to 30th neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30th neighbors are larger than this cutoff would be filtered.
        n_neighbors: The number of nearest neighbors that will be considered for calculating spatial distance.

    Returns:
        bin_scale_adata: Bin, scaled anndata object.
        M: The scipy sparse matrix of the calculated distance of nearest neighbors.
    """
    bin_scale_adata = bin_adata(adata, bin_size, layer=bin_layer)
    bin_scale_adata = bin_scale_adata[:, np.sum(bin_scale_adata.X, axis=0) > 0]
    bin_scale_adata = st.svg.scale_to(bin_scale_adata)
    if cell_distance_method == "geodesic":
        bin_scale_adata = st.svg.utils.cal_geodesic_distance(
            bin_scale_adata,
            min_dis_cutoff=min_dis_cutoff,
            max_dis_cutoff=max_dis_cutoff,
            layer=distance_layer,
            n_neighbors=n_neighbors,
        )
    elif cell_distance_method == "euclidean":
        bin_scale_adata = st.svg.utils.cal_euclidean_distance(
            bin_scale_adata, min_dis_cutoff=min_dis_cutoff, max_dis_cutoff=max_dis_cutoff, layer=distance_layer
        )

    M = bin_scale_adata.obsp["distance"]

    return bin_scale_adata, M


def cal_wass_dist_bs(
    adata: AnnData,
    bin_size: int = 1,
    bin_layer: str = "spatial",
    cell_distance_method: str = "geodesic",
    distance_layer: str = "spatial",
    n_neighbors: int = 30,
    numItermax: int = 1000000,
    gene_set: Union[List, np.ndarray] = None,
    target: Union[List, np.ndarray, str] = [],
    processes: int = 1,
    bootstrap: int = 100,
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
    rank_p: bool = True,
    bin_num: int = 100,
    larger_or_small: str = "larger",
) -> Tuple[pd.DataFrame, AnnData]:
    """Computing Wasserstein distance for an AnnData to identify spatially variable genes.

    Args:
        adata: AnnData object.
        bin_size: Bin size for mergeing cells.
        bin_layer: Data in this layer will be binned according to the spatial information.
        cell_distance_method: The method for calculating distance between two cells, either geodesic or euclidean.
        distance_layer: The data of this layer would be used to calculate distance
        n_neighbors: The number of neighbors for calculating spatial distance.
        numItermax: The maximum number of iterations before stopping the optimization algorithm if it has not converged.
        gene_set: Gene set that will be used to compute Wasserstein distances, default is for all genes.
        target: The target gene expression distribution or the target gene name.
        processes: The process number for parallel computing
        bootstrap: Bootstrap number for permutation to calculate p-value
        min_dis_cutoff: Cells/Bins whose min distance to 30th neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30th neighbors are larger than this cutoff would be filtered.
        rank_p: Whether to calculate p value in ranking manner.
        bin_num: Classy genes into bin_num groups according to mean Wasserstein distance from bootstrap.
        larger_or_small: In what direction to get p value. Larger means the right tail area of the null distribution.

    Returns:
        w_df: A dataframe storing information related to the Wasserstein distances.
        bin_scale_adata: Binned AnnData object
    """

    bin_scale_adata, M = bin_scale_adata_get_distance(
        adata,
        bin_size=bin_size,
        bin_layer=bin_layer,
        distance_layer=distance_layer,
        min_dis_cutoff=min_dis_cutoff,
        max_dis_cutoff=max_dis_cutoff,
        cell_distance_method=cell_distance_method,
        n_neighbors=n_neighbors,
    )

    if gene_set is None:
        gene_set = list(bin_scale_adata.var_names)

    if isinstance(target, (list, np.ndarray)):
        b = target
    if isinstance(target, str):
        b = (
            bin_scale_adata[:, target].X.toarray().flatten()
            if issparse(bin_scale_adata[:, target].X)
            else bin_scale_adata[:, target].X.flatten()
        )
        b = np.array(b, dtype=np.float64)
        b = b / np.sum(b)
        
    genes, ws, pos_rs = st.svg.cal_wass_dis_for_genes((M, bin_scale_adata), (0, gene_set, b, numItermax))
    w_df_ori = pd.DataFrame({"gene_id": genes, "Wasserstein_distance": ws, "positive_ratio": pos_rs})

    pool = multiprocessing.Pool(processes=processes)

    inputs = [(i, gene_set, b, numItermax) for i in range(1, bootstrap + 1)]
    res = []
    for result in tqdm(
        pool.imap_unordered(partial(st.svg.cal_wass_dis_for_genes, (M, bin_scale_adata)), inputs), total=len(inputs)
    ):
        res.append(result)

    genes, ws, pos_rs = zip(*res)
    genes = [g for i in genes for g in i]
    ws = [g for i in ws for g in i]
    pos_rs = [g for i in pos_rs for g in i]
    w_df = pd.DataFrame({"gene_id": genes, "Wasserstein_distance": ws, "positive_ratio": pos_rs})
    mean_std_df = pd.DataFrame(
        {
            "mean": w_df.groupby("gene_id")["Wasserstein_distance"].mean().to_list(),
            "std": w_df.groupby("gene_id")["Wasserstein_distance"].std().to_list(),
        },
        index=w_df.groupby("gene_id")["Wasserstein_distance"].mean().index,
    )
    w_df = pd.concat([w_df_ori.set_index("gene_id"), mean_std_df], axis=1)
    w_df["zscore"] = (w_df["Wasserstein_distance"] - w_df["mean"]) / w_df["std"]

    w_df = w_df.replace(np.inf, 0).replace(np.nan, 0)

    # find p-value
    if larger_or_small == "larger":
        w_df["pvalue"] = sp.stats.norm.sf(w_df["zscore"])
    elif larger_or_small == "small":
        w_df["pvalue"] = 1 - sp.stats.norm.sf(w_df["zscore"])

    w_df["adj_pvalue"] = multipletests(w_df["pvalue"])[1]

    w_df["fc"] = w_df["Wasserstein_distance"] / w_df["mean"]
    w_df["log2fc"] = np.log2(w_df["fc"])
    w_df["-log10adjp"] = -np.log10(w_df["adj_pvalue"])

    # rank p
    if rank_p:
        w_df["rank_p"], each_bin_ws = st.svg.cal_rank_p(genes, ws, w_df, bin_num=bin_num)
        w_df.loc[w_df["positive_ratio"] == 0, "rank_p"] = 1.0
        w_df["adj_rank_p"] = multipletests(w_df["rank_p"])[1]

    w_df = w_df.replace(np.inf, 0).replace(np.nan, 0)
    return w_df, bin_scale_adata  # , each_bin_ws


def cal_wass_dis_target_on_genes(
    adata: AnnData,
    bin_size: int = 1,
    bin_layer: str = "spatial",
    distance_layer: str = "spatial",
    cell_distance_method: str = "geodesic",
    n_neighbors: int = 30,
    numItermax: int = 1000000,
    target_genes: Union[List, np.ndarray] = None,
    gene_set: Union[List, np.ndarray] = None,
    processes: int = 1,
    bootstrap: int = 0,
    top_n: int = 100,
    min_dis_cutoff: float = 2.0,
    max_dis_cutoff: float = 6.0,
) -> Tuple[dict, AnnData]:
    """Find genes in gene_set that have similar distribution to each target_genes.

    Args:
        adata: AnnData object.
        bin_size: Bin size for mergeing cells.
        bin_layer: Data in this layer will be binned according to the spatial information.
        distance_layer: The data of this layer would be used to calculate distance
        cell_distance_method: The method for calculating distance between two cells, either geodesic or euclidean.
        n_neighbors: The number of neighbors for calculating spatial distance.
        numItermax: The maximum number of iterations before stopping the optimization algorithm if it has not converged.
        target_genes: The list of the target genes.
        gene_set: Gene set that will be used to compute Wasserstein distances, default is for all genes.
        processes: The process number for parallel computing.
        bootstrap: Number of bootstraps.
        top_n: Number of top genes to select.
        min_dis_cutoff: Cells/Bins whose min distance to 30th neighbors are larger than this cutoff would be filtered.
        max_dis_cutoff: Cells/Bins whose max distance to 30th neighbors are larger than this cutoff would be filtered.

    Returns:
        w_genes: The dictionary of the Wasserstein distance. Each key corresponds to a gene name while the corresponding
            value the pandas DataFrame of the Wasserstein distance related information.
        bin_scale_adata: binned, scaled anndata object.
    """
    bin_scale_adata, M = bin_scale_adata_get_distance(
        adata,
        bin_size=bin_size,
        bin_layer=bin_layer,
        distance_layer=distance_layer,
        min_dis_cutoff=min_dis_cutoff,
        max_dis_cutoff=max_dis_cutoff,
        cell_distance_method=cell_distance_method,
        n_neighbors=n_neighbors,
    )

    if gene_set is None:
        gene_set = bin_scale_adata.var_names

    if issparse(bin_scale_adata.X):
        df = pd.DataFrame(bin_scale_adata.X.toarray(), columns=bin_scale_adata.var_names)
    else:
        df = pd.DataFrame(bin_scale_adata.X, columns=bin_scale_adata.var_names)

    w_genes = {}
    for gene in target_genes:
        b = np.array(df.loc[:, gene], dtype=np.float64) / np.array(df.loc[:, gene], dtype=np.float64).sum(0)
        genes, ws, pos_rs = st.svg.cal_wass_dis_for_genes((M, bin_scale_adata), (0, gene_set, b, numItermax))
        w_genes[gene] = pd.DataFrame({"gene_id": genes, "Wasserstein_distance": ws, "positive_ratio": pos_rs})

    if bootstrap == 0:
        return w_genes, bin_scale_adata

    for gene in target_genes:
        tmp = w_genes[gene]
        gene_set = tmp[tmp["positive_ratio"] > 0].sort_values(by="Wasserstein_distance")["gene_id"].head(top_n + 1)
        w_df, _ = cal_wass_dist_bs(
            adata,
            gene_set=gene_set,
            target=gene,
            bin_size=bin_size,
            bin_layer=bin_layer,
            distance_layer=distance_layer,
            min_dis_cutoff=min_dis_cutoff,
            max_dis_cutoff=max_dis_cutoff,
            cell_distance_method=cell_distance_method,
            bootstrap=bootstrap,
            processes=processes,
            larger_or_small="small",
            rank_p=False,
        )

        w_genes[gene] = w_df
    return w_genes, bin_scale_adata


def SVG(adata, method='Moran', top=100):
    """
    Identifies spatially variable genes (SVGs) using various spatial methods.

    This function detects the top spatially variable genes from a spatial 
    transcriptomics dataset (`adata`) using one of several supported methods: 
    `"Moran"`, `"SpatialDE"`, `"Spanve"`, `"SpaGFT"`, `"spateo"`.

    Args:
        adata (AnnData): Annotated data matrix (from the `anndata` package), 
            containing spatial coordinates and gene expression data.
        method (str, optional): The method used for SVG detection. 
            One of: `"Moran"`, `"SpatialDE"`, `"Spanve"`, or `"SpaGFT"`. 
            Defaults to `"Moran"`.
        top (int, optional): Number of top-ranked SVGs to return. Defaults to 100.

    Returns:
        set: A set of gene names identified as the top spatially variable genes 
        according to the selected method.

    Raises:
        ValueError: If an unsupported method name is provided.

    Notes:
        - `"Moran"` uses Squidpy's Moran's I for spatial autocorrelation.
        - `"SpatialDE"` applies the SpatialDE pipeline with normalization and 
          regression of total counts using `NaiveDE`.
        - `"Spanve"` uses the Spanve model to select genes based on entropy.
        - `"SpaGFT"` identifies spatially variable genes using graph Fourier 
          transform techniques, requiring spatial coordinates in `adata.obsm['spatial']`.
        - `"spateo"` identifies spatially variable genes using optimal transport 
          method by comparing the distribution of genes to a uniform distribution,
          requiring spatial coordinates in `adata.obsm['spatial']`.
    """

    if method == 'Moran':
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
        sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100, n_jobs=4)
        moran_set = set((adata.uns["moranI"].head(top)).index)

        return moran_set

    if method == 'SpatialDE':
        adata.layers["normalized"] = adata.X
        if issparse(adata.layers['counts']):
            counts = pd.DataFrame(adata.layers['counts'].todense(), index=list(
                adata.obs_names), columns=list(adata.var_names))
        else: 
            counts = pd.DataFrame(adata.layers['counts'], index=list(
                adata.obs_names), columns=list(adata.var_names))
        adata.layers["counts"] = counts.values
        total_counts = sc.get.obs_df(adata, keys=["total_counts"])
        norm_expr = NaiveDE.stabilize(counts.T).T
        resid_expr = NaiveDE.regress_out(
            total_counts, norm_expr.T, "np.log(total_counts)").T
        df = spatialDE.run(adata.obsm["spatial"], resid_expr)
        top100 = df.sort_values("qval").head(top)[["g", "l", "qval"]]
        spatialDE_set = set(top100['g'])

        return spatialDE_set

    if method == 'Spanve':
        adata.X = adata.layers['counts']
        spanve = Spanve(adata)
        spanve.fit(verbose=True)
        df = spanve.result_df
        spanve_set = set(df.sort_values(by='ent')[:top].index)

        return spanve_set

    if method == 'SpaGFT':
        adata.X = adata.layers['normalized']
        adata.obs.loc[:, ['array_row', 'array_col']] = adata.obsm['spatial']
        (ratio_low, ratio_high) = determine_frequency_ratio(
            adata, ratio_neighbors=1)

        df = detect_svg(adata, spatial_info=['array_row', 'array_col'],
                        ratio_low_freq=ratio_low, ratio_high_freq=ratio_high,
                        ratio_neighbors=1, filter_peaks=True, S=6)
        spagft_set = set(df.sort_values(
            by='gft_score', ascending=False)[:top].index)

        return spagft_set
    
    if method == 'spateo':
        # downsampling cells
        adata_downsampled = st.svg.downsampling(adata, downsampling=2000)
        
        # Identify SVGs by comparing to uniform distribution
        w_df, _ = cal_wass_dist_bs(adata_downsampled, bin_size=1, processes=7, 
                                      n_neighbors=8, bin_num=100, 
                                      min_dis_cutoff=500, max_dis_cutoff=1000, 
                                      bootstrap=100)
        
        # Select out significant SVGs
        st.svg.add_pos_ratio_to_adata(adata, layer='counts')
        w_df['raw_pos_rate'] = adata.var['raw_pos_rate']
        sig_df = w_df[(w_df['log2fc'] >= 1) & (w_df['rank_p'] <= 0.05) & (
            w_df['raw_pos_rate'] >= 0.05) & (w_df['adj_pvalue'] <= 0.05)]
        
        spateo_set = set(sig_df.index)
        
        return spateo_set, sig_df
    
    
def spateo_similar_gene_pattern(adata, gene):
    """
    Identify genes with spatial expression patterns similar to a target gene.

    This function uses spatial Wasserstein distance to find genes whose spatial 
    expression distributions resemble that of a given gene. It performs 
    downsampling before calculating pairwise distances.

    Args:
        adata (AnnData): The input spatial transcriptomics data in AnnData format.
        gene (str): The target gene name whose spatial expression pattern is used 
            as reference.

    Returns:
        pandas.DataFrame: A DataFrame containing genes sorted by ascending 
        Wasserstein distance to the target gene. Lower values indicate more 
        similar spatial patterns.
    """
    
    adata_downsampled = st.svg.downsampling(adata, downsampling=500)
    adata_gene_w, _ = cal_wass_dis_target_on_genes(adata_downsampled, n_neighbors=8, 
                                                   target_genes=[gene], 
                                                   processes=7, bootstrap=100, 
                                                   min_dis_cutoff=500, max_dis_cutoff=1000)
    res = adata_gene_w[gene].sort_values(by='Wasserstein_distance')
    
    return res
    


def spagcn_svg(adata, random_seed=100):
    """
    Run SpaGCN on spatial transcriptomics data to identify spatial domains 
    and detect differentially expressed (DE) genes between spatial clusters.

    This function performs the following steps:
    1. Sets the random seed for reproducibility.
    2. Computes the spatial adjacency matrix.
    3. Tunes the spatial smoothing parameter `l`.
    4. Trains a SpaGCN model to predict spatial domains (clusters).
    5. Identifies DE genes between each cluster and its neighboring clusters.
    6. Returns a DataFrame of gene-level statistics and metadata.

    Args:
        adata (anndata.AnnData): 
            Annotated data matrix containing spatial transcriptomics data. 
            Must include `.obsm['spatial']` with spatial coordinates.
        random_seed (int, optional): 
            Random seed for reproducibility. Default is 100.

    Returns:
        pandas.DataFrame: 
            A DataFrame containing gene-level statistics including adjusted 
            p-values for differential expression and gene metadata. If no 
            DE genes are detected, a placeholder DataFrame with random p-values 
            is returned.

    Raises:
        None explicitly, but catches and suppresses RuntimeError, TypeError, 
        and NameError during DE gene identification.

    Notes:
        - The function modifies `adata.obs["pred"]` in place to store predicted clusters.
        - Requires spatial clustering to be feasible (e.g., sufficient spatial structure).
        - Assumes the presence of spatial coordinates in `adata.obsm['spatial']`.
    """
    
    random_seed = random_seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    p = 0.5
    adj = calculate_adj_matrix(x=adata.obsm["spatial"][:, 0],
                               y=adata.obsm["spatial"][:, 1], histology=False)
    l = search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    
    clf = SpaGCN()
    clf.set_l(l)
    clf.train(adata, adj, init_spa=True, init="leiden", n_neighbors=10, res=0.5,
              tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob = clf.predict()
    adata.obs["pred"] = y_pred
    de_genes_all = list()
    n_clusters = len(adata.obs["pred"].unique())
    
    # identify DE genes
    for target in range(n_clusters):
        print(f"target: {target}")
        start, end = np.quantile(adj[adj != 0], q=0.001), np.quantile(
            adj[adj != 0], q=0.1)
        r = search_radius(target_cluster=target, cell_id=adata.obs.index.tolist(),
                          x=adata.obsm["spatial"][:,
                                                  0], y=adata.obsm["spatial"][:, 1],
                          pred=adata.obs["pred"].tolist(), start=start, end=end,
                          num_min=10, num_max=14, max_run=100)

        try:
            nbr_domains = find_neighbor_clusters(target_cluster=target,
                                                 cell_id=adata.obs.index.tolist(), x=adata.obsm["spatial"][:, 0],
                                                 y=adata.obsm["spatial"][:, 1], pred=adata.obs["pred"].tolist(
                                                 ),
                                                 radius=r, ratio=0)

            de_genes_info = rank_genes_groups(input_adata=adata, target_cluster=target,
                                              nbr_list=nbr_domains, label_col="pred",
                                              adj_nbr=True, log=True)
            de_genes_all.append(de_genes_info)
        except (RuntimeError, TypeError, NameError):
            pass

    if len(de_genes_all) == 0:
        df = adata.var
        df['pvals_adj'] = np.random.random(adata.n_vars)
    else:
        df_res = pd.concat(de_genes_all)
        df_res = df_res.groupby(["genes"]).min()
        df_res = df_res.loc[adata.var_names]
        df = pd.concat([df_res, adata.var], axis=1)
        
    return df


def spagcn_domain(adata, num_cluster = 5, random_seed=100):
    """
    Run SpaGCN to identify and visualize spatial domains in spatial transcriptomics data.

    This function performs preprocessing, adjacency matrix computation, parameter tuning, 
    clustering, prediction refinement, and spatial visualization. It adds predicted and 
    refined domain labels to the AnnData object and generates a spatial scatter plot.

    Args:
        adata (anndata.AnnData): 
            Annotated data matrix containing spatial transcriptomics data. 
            Must include `.obsm['spatial']` with spatial coordinates.
        num_cluster (int, optional): 
            The number of spatial domains (clusters) to identify. Default is 5.
        random_seed (int, optional): 
            Random seed for reproducibility. Default is 100.

    Returns:
        None: 
            The function modifies the `adata` object in place by adding predicted 
            and refined domain labels under `adata.obs["pred"]` and `adata.obs["refined_pred"]`. 
            A spatial plot is saved to `"figures/spagcn_domain.png"`.

    Notes:
        - The function uses `SpaGCN` for spatial domain identification.
        - Colors for domain plotting are predefined and truncated if fewer than 20 clusters are used.
        - The spatial plot is saved with high resolution (600 dpi).
        - Requires the following `adata` fields: `.obsm['spatial']`, `.obs`, and index.
        - Saves no return value but modifies `adata` and creates a plot.

    """

    prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    prefilter_specialgenes(adata)
    p = 0.5
    adj = calculate_adj_matrix(x=adata.obsm["spatial"][:, 0],
                               y=adata.obsm["spatial"][:, 1], histology=False)
    l = search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    x_array = adata.obsm["spatial"][:, 0]
    y_array = adata.obsm["spatial"][:, 1]

    n_clusters = num_cluster
    r_seed = t_seed = n_seed = random_seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)

    res = search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3,
                         lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
    clf = SpaGCN()
    clf.set_l(l)

    # Run
    clf.train(adata, adj, init_spa=True, init="leiden",
              res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob = clf.predict()
    adata.obs["pred"] = y_pred
    adata.obs["pred"] = adata.obs["pred"].astype('category')

    adj_2d = calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    refined_pred = refine(sample_id=adata.obs.index.tolist(
    ), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"] = refined_pred
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
    # Save results

    # Set colors used
    plot_color = ["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", 
                  "#D1D1D1", "#6D1A9C", "#15821E", "#3A84E6", "#997273", 
                  "#787878", "#DB4C6C", "#9E7A7A", "#554236", "#AF5F3C", 
                  "#93796C", "#F9BD3F", "#DAB370", "#877F6C", "#268785"]
    # Plot spatial domains
    domains = "pred"
    num_celltype = len(adata.obs[domains].unique())
    adata.uns[domains+"_colors"] = list(plot_color[:num_celltype])
    sq.pl.spatial_scatter(adata, color=domains, shape=None, size=10, dpi=600)
    plt.savefig("figures/spagcn_domain.png", dpi=600)
    plt.close()
    
    #Plot refined spatial domains
    domains="refined_pred"
    num_celltype=len(adata.obs[domains].unique())
    adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
    sq.pl.spatial_scatter(adata, color=domains, shape=None, size=10, dpi=600)
    plt.savefig("figures/spagcn_domain_refined.png", dpi=600)
    plt.close()
    
    return None


def cellcharter_domain(adata):
    """
    Run CellCharter pipeline to identify spatial domains using scVI-based latent space 
    and graph-based clustering on spatial transcriptomics data.

    This function performs the following steps:
    1. Sets up and trains an scVI model on raw counts.
    2. Computes a low-dimensional latent representation (`X_scVI`).
    3. Builds spatial neighbor graphs using Delaunay triangulation.
    4. Aggregates neighborhood features across multiple spatial layers.
    5. Automatically determines the optimal number of clusters using ClusterAutoK.
    6. Assigns cluster labels and visualizes spatial domains.

    Args:
        adata (anndata.AnnData): 
            Annotated data matrix with spatial transcriptomics counts in `.layers['counts']` 
            and spatial coordinates in `.obsm['spatial']`.

    Returns:
        None: 
            The function modifies `adata` in place:
            - Adds latent representation in `adata.obsm['X_scVI']`.
            - Adds spatial domain features in `adata.obsm['X_cellcharter']`.
            - Stores predicted cluster labels in `adata.obs['cluster_cellcharter']`.
            - Saves a spatial scatter plot as `"cellcharter.png"` in the working directory.

    Notes:
        - The function uses `scvi-tools`, `scanpy`, `squidpy`, and `cellcharter`.
        - Requires internet or local cache to download pretrained weights if applicable.
        - The clustering stability plot is shown but not saved unless handled externally.
        - Spatial plot uses Matplotlib's default output location unless changed via `rcParams`.

    Raises:
        None explicitly. Assumes that:
        - `adata.layers['counts']` exists and contains raw UMI counts.
        - `adata.obsm['spatial']` contains valid coordinates.
        - Required packages (`scvi-tools`, `squidpy`, `cellcharter`) are installed.
    """
    
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(adata)
    model.train(early_stopping=True, enable_progress_bar=True)
    adata.obsm['X_scVI'] = model.get_latent_representation(
        adata).astype(np.float32)
    
    sq.gr.spatial_neighbors(adata, coord_type='generic',
                            delaunay=True, percentile=99)
    cc.gr.aggregate_neighbors(
        adata, n_layers=3, use_rep='X_scVI', out_key='X_cellcharter')
    autok = cc.tl.ClusterAutoK(n_clusters=(
        2, 12), max_runs=10, convergence_tol=0.001)
    autok.fit(adata, use_rep='X_cellcharter')
    cc.pl.autok_stability(autok)
    adata.obs['cluster_cellcharter'] = autok.predict(
        adata, use_rep='X_cellcharter')
    sq.pl.spatial_scatter(adata, library_id="spatial", color='cluster_cellcharter',
                          shape=None, size=2,  palette='Set2', save="cellcharter_domain.png", dpi=600)
    
    return None


def tangram_imputation(adata, adata_sc, xenium_table, top_genes=100):
    """
    Perform spatial gene expression imputation using Tangram by mapping single-cell RNA-seq 
    data onto spatial transcriptomics data.

    This function prepares and filters input data, performs gene matching via Ensemble IDs, 
    selects highly variable genes, trains a Tangram model, and evaluates mapping performance. 
    It projects cell-type annotations and gene expression predictions onto spatial coordinates 
    and visualizes several diagnostic plots.

    Args:
        adata (anndata.AnnData): 
            Spatial transcriptomics dataset with gene expression and spatial coordinates. 
            Spatial coordinates should be in `adata.obs[['x_global_px', 'y_global_px']]`.
        adata_sc (anndata.AnnData): 
            Single-cell RNA-seq dataset with raw counts in `.X` and a categorical 
            cell type annotation in `adata_sc.obs['cell_type']`.
        xenium_table (pandas.DataFrame): 
            A table mapping gene symbols to Ensembl IDs, used to harmonize gene names 
            across modalities.

    Returns:
        None: 
            The function modifies `adata` in place by:
            - Mapping cell types to spatial locations via Tangram.
            - Projecting gene expression data onto the spatial tissue.
            - Creating and displaying various plots (training diagnostics, cell type maps, 
              AUC scores, and spatial gene visualizations).

    Notes:
        - Genes with infinite mean expression in `adata_sc` are removed before training.
        - Only the top 100 highly variable genes in the spatial data are used for training.
        - Remaining genes are used for downstream validation and visualization.
        - Spatial plots use `x_global_px` and `y_global_px` as spatial axes.

    Raises:
        None explicitly, but assumes:
        - `adata_sc.obs['cell_type']` exists.
        - `xenium_table` includes valid gene name mappings.
        - Required libraries (`scanpy`, `tangram`, etc.) are available and properly configured.

    Side Effects:
        - Plots are generated inline or saved, depending on the plot functions used.
        - Spatial and projected data are modified in `adata`.
    """
    
    # Check if data are raw counts (approximate)
    print(adata_sc.obs.cell_type.value_counts())
    
    # Normalize total counts per cell
    sc.pp.normalize_total(adata_sc)
    means = np.array(adata_sc.X.mean(axis=0)).flatten()
    inf_genes = np.where(np.isinf(means))[0]
    print(f"Detected {len(inf_genes)} genes with infinite mean.")
    if len(inf_genes) > 0:
        adata_sc = adata_sc[:, ~np.isinf(means)].copy()
    
    # use the EnsembleID
    gene_to_ensg = genename_to_ensg(xenium_table)
    adata_ensembl = adata.copy()
    adata_ensembl.var_names = adata_ensembl.var_names.map(gene_to_ensg)
    
    # Select top 100 highly variable genes
    sc.pp.highly_variable_genes(adata_ensembl, n_top_genes=top_genes, flavor="cell_ranger")
    genes = adata_ensembl.var[adata_ensembl.var.highly_variable].index.tolist()
    pp_adatas(adata_sc, adata_ensembl, genes=genes)
    
    # Run mapping using clusters to reduce memory
    tg_map = map_cells_to_space(adata_sc, adata_ensembl, device='cpu', num_epochs=500)
    
    # Project Cell Type Annotations to Spatial Data
    tg.utils.project_cell_annotations(tg_map, adata_ensembl, annotation="cell_type")
    annotation_list = list(pd.unique(adata_sc.obs['cell_type']))
    plot_cell_annotation_sc(adata_ensembl, annotation_list, x='x_global_px',
                            y='y_global_px', spot_size=5, scale_factor=1, perc=0.1)
    plt.savefig('figures/tangram_cell_annotation.png', format='png', dpi=600)
    plt.close()
    
    plot_training_scores(tg_map)
    plt.savefig('figures/tangram_training.png', format='png', dpi=600)
    plt.close()
    
    # Check accuracy on the validation set (remaining genes)
    ad_ge = project_genes(adata_map=tg_map, adata_sc=adata_sc)
    df_all_genes = compare_spatial_geneexp(ad_ge, adata_ensembl, adata_sc)
    plot_auc(df_all_genes)
    
    # Visualisation of 10 random genes from this remaining set
    random_genes_vis = random.sample(
        list(df_all_genes[df_all_genes['is_training'] == False].index), 10)
    plot_genes_sc(random_genes_vis, adata_measured=adata_ensembl, adata_predicted=ad_ge,
                  spot_size=10, scale_factor=0.1, perc=0.02, return_figure=True)
    plt.savefig('figures/tangram_genes_sc.png', format='png', dpi=600)
    plt.close()
    
    return tg_map, adata_ensembl


def liana_ccc(adata, source, target):
    """
    Perform and visualize cell-cell communication (CCC) analysis using LIANA and CellPhoneDB.

    This function uses LIANA to run the CellPhoneDB method for ligand-receptor inference,
    followed by several visualizations of predicted interactions between specific cell types. 
    It also performs consensus-based ranking of ligand-receptor pairs across multiple methods.

    Args:
        adata (anndata.AnnData): 
            Annotated data matrix with cell type labels in `adata.obs['cell_type']` 
            and normalized gene expression data in `.X`.

    Returns:
        None: 
            The function modifies `adata` in place by adding:
            - `'cpdb_res'`: results from CellPhoneDB stored in `adata.uns`.
            - `'liana_res'`: consensus-based ligand-receptor rankings in `adata.uns`.

    Notes:
        - CellPhoneDB is run using LIANA with a minimum expression proportion of 0.1.
        - Dotplot and tileplot visualizations are generated to highlight interactions 
          from 'endothelial cell' to 'macrophage'.
        - Tileplots display interaction strength and supporting proportions.
        - Consensus ranking aggregates results from multiple CCC tools, stored in `adata.uns['liana_res']`.
        - Circle plots highlight ligand-receptor pairs with high specificity 
          (`specificity_rank <= 0.05`) based on magnitude scores.

    Raises:
        None explicitly, but assumes:
        - `adata.obs['cell_type']` exists and is categorical.
        - Required libraries (`liana`, `pandas`, etc.) are installed and functioning.

    Side Effects:
        - Modifies the `adata.uns` dictionary with CCC results.
        - Generates and displays plots inline (not saved by default).
    """

    cellphonedb(adata, groupby='cell_type', resource_name='cellphonedb', expr_prop=0.1, use_raw=False, verbose=True, key_added='cpdb_res')
    adata.uns['cpdb_res'].head()
    
    dotplot = li.pl.dotplot(adata=adata, colour='lr_means', size='cellphone_pvals',
                  inverse_size=True, source_labels=[source],
                  target_labels=[target], uns_key='cpdb_res')
    dotplot.save('figures/liana_dotplot' + source + '_' + target + '.png', dpi=600)

    tileplot = li.pl.tileplot(adata, fill='means', label='props', label_fun=lambda x: f'{x:.2f}',
                   top_n=10, orderby='cellphone_pvals', orderby_ascending=True,
                   source_labels=[source], target_labels=[target],
                   uns_key='cpdb_res', source_title='Ligand',
                   target_title='Receptor')
    tileplot.save('figures/liana_tileplot' + source + '_' + target + '.png', dpi=600)
    
    li.mt.rank_aggregate(adata, groupby="cell_type", return_all_lrs=True,
                         use_raw=False, resource_name='consensus', expr_prop=0.1,
                         verbose=True)
    
    li.pl.circle_plot(adata, groupby='cell_type', score_key='magnitude_rank',
                      inverse_score=True, source_labels=source,
                      filter_fun=lambda x: x['specificity_rank'] <= 0.05,
                      pivot_mode='counts', edge_arrow_size=5, edge_width_scale=(0.5,2),
                      node_label_size=5, node_label_offset=(0,-0.2))
    plt.savefig('figures/liana_circle_plot' + source +'.png', format='png', dpi=600)
    plt.close()

    return None
    

def cellphoneDB_ccc(adata, source, target, genes_list=None):
    """
    Perform cell-cell communication (CCC) analysis using CellPhoneDB and visualize 
    key ligand-receptor interactions between selected cell types.

    This function:
    1. Extracts metadata from `adata` and saves it as a CellPhoneDB-formatted metadata file.
    2. Runs CellPhoneDB's statistical analysis on the input AnnData object.
    3. Generates visualizations including a heatmap of significant interactions and 
       dot plots for selected ligand-receptor pairs.

    Args:
        adata (anndata.AnnData): 
            Annotated data matrix containing gene expression in `.X` and cell type 
            annotations in `adata.obs['cell_type']`. Gene names must be Ensembl IDs.

    Returns:
        None:
            The function saves metadata and CellPhoneDB outputs to disk, and displays
            heatmaps and interaction plots using `kpy.plot_cpdb`. Modifies no data in-place.

    Notes:
        - The CellPhoneDB database zip file must be available at `'CellphoneDB_v5/cellphonedb.zip'`.
        - Output files are written to `'results/cellphoneDB/'`.
        - The metadata file `'metadata_cell_type.csv'` is saved to the working directory.
        - Visualizations focus on interactions between 'endothelial cell' and 'macrophage'.
        - Includes focused plots for selected genes: `"THY1"` and `"NOTCH2"`.

    Raises:
        None explicitly, but assumes:
        - `adata.obs['cell_type']` exists.
        - Gene names in `adata` match the format expected by CellPhoneDB (`counts_data='ensembl'`).
        - Required modules (`cpdb_statistical_analysis_method`, `kpy`, `pandas`) are available.

    Side Effects:
        - Writes metadata and results files to disk.
        - Generates and displays several plots using `kpy.plot_cpdb_heatmap` and `kpy.plot_cpdb`.
    """
    
    cpdb_file_path = 'CellphoneDB_v5/cellphonedb.zip'
    metadata_cell_type = pd.DataFrame(dtype=object)
    metadata_cell_type['cell_type'] = adata.obs['cell_type'].astype(object)
    metadata_cell_type['Cell'] = adata.obs_names
    metadata_cell_type[['Cell', 'cell_type']].to_csv(
        'metadata_cell_type.csv', index=False)
    meta_file_path = 'metadata_cell_type.csv'
    out_path = 'results/cellphoneDB'
    
    cpdb_results = cpdb_statistical_analysis_method.call(cpdb_file_path=cpdb_file_path,
        meta_file_path=meta_file_path, counts_file_path=adata, counts_data='hgnc_symbol',
        score_interactions=True, output_path=out_path, separator='|', threads=5,
        threshold=0.1, result_precision=3)
    
    # Heatmap
    kpy.plot_cpdb_heatmap(pvals=cpdb_results['pvalues'], degs_analysis=False,
                          figsize=(5, 5), title="Sum of significant interactions")
    plt.savefig('figures/cellphonedb_plot1' + source + '_' + target + '.png', 
                format='png', dpi=600)
    plt.close()
    print(cpdb_results['pvalues'][cpdb_results['pvalues']
          [source+'|'+target] < 1][['gene_a', 'gene_b']])
    
    if genes_list != None:
        plot = kpy.plot_cpdb(adata, cell_type1=source, cell_type2=target,
                      means=cpdb_results['means'], pvals=cpdb_results['pvalues'],
                      celltype_key="cell_type", genes=genes_list, figsize=(10, 3),
                      title="Interactions between\nPV and trophoblast", max_size=3,
                      highlight_size=0.75, degs_analysis=False, standard_scale=True,
                      interaction_scores=cpdb_results['interaction_scores'],
                      scale_alpha_by_interaction_scores=True)
        plot.save('figures/cellphonedb_plot2' + source + '_' + target + '.png', dpi=600)
        
        p = kpy.plot_cpdb(adata, cell_type1=source, cell_type2=target,
                          means=cpdb_results['means'], pvals=cpdb_results['pvalues'],
                          celltype_key="cell_type", genes=genes_list, figsize=(12, 8),
                          title="Interactions between PV and trophoblast\ns grouped by classification",
                          max_size=6, highlight_size=0.75, degs_analysis=False, standard_scale=True)
        p + facet_wrap("~ classification", ncol=1)
        p.save('figures/cellphonedb_plot3' + source + '_' + target + '.png', dpi=600)
        
    return None


# def mebocost_ccc(adata):
#     """
#     Perform metabolic cell-cell communication (mCCC) analysis using MeboCost.

#     This function initializes a MeboCost object, estimates metabolite expression and 
#     enzyme-sensor co-expression, then infers metabolic communication between cell types 
#     based on statistical testing. The results include a metabolite activity matrix and 
#     a list of significant interactions.

#     Args:
#         adata (anndata.AnnData): 
#             Annotated data matrix with gene expression and cell type information. 
#             Must include `adata.obs['cell_type']` for group-wise inference.

#     Returns:
#         Tuple:
#             - mebo_obj (MeboCost): 
#                 The fitted MeboCost object containing metabolic expression estimates and internal data.
#             - commu_res (pandas.DataFrame): 
#                 A DataFrame of significant metabolic communications between cell types 
#                 based on enzyme-sensor co-expression and statistical testing.

#     Notes:
#         - MeboCost configuration is loaded from `'mebocost.conf'`.
#         - Only metabolites with expression above the `cutoff_prop` (default 0.25) are considered.
#         - The analysis uses 1000 permutations for statistical testing (FDR corrected).
#         - Interaction detection is based on co-expression of enzymes and sensors 
#           (Receptors, Transporters, and Nuclear Receptors).
#         - Detected results are printed and the full MeboCost object is saved 
#           to `'mebocost_cell_commu.pk'`.

#     Raises:
#         None explicitly, but assumes:
#         - The configuration file `mebocost.conf` is present and correctly formatted.
#         - Required packages (`mebocost`, `pandas`, etc.) are installed.
#         - `adata.obs['cell_type']` exists and is a valid categorical annotation.

#     Side Effects:
#         - Saves the MeboCost object to disk as `'mebocost_cell_commu.pk'`.
#         - Prints summary statistics to the console (number of detected interactions).
#     """
    
#     mebo_obj = mebocost.create_obj(adata = adata, group_col = ['cell_type'],
#                         met_est = 'mebocost', config_path = 'mebocost.conf',
#                         exp_mat=None, cell_ann=None, species='human',
#                         met_pred=None, met_enzyme=None, met_sensor=None,
#                         met_ann=None, scFEA_ann=None, compass_met_ann=None,
#                         compass_rxn_ann=None, cutoff_exp='auto', cutoff_met='auto',
#                         cutoff_prop=0.25, sensor_type=['Receptor', 'Transporter', 'Nuclear Receptor'],
#                         thread=8)
#     mebo_obj._load_config_()
#     mebo_obj.estimator()
#     met_mat = pd.DataFrame(mebo_obj.met_mat.toarray(), index=mebo_obj.met_mat_indexer,
#                            columns=mebo_obj.met_mat_columns)
#     met_mat.head()
    
#     commu_res = mebo_obj.infer_commu(n_shuffle=1000, seed=12345, Return=True,
#                                      thread=None, save_permuation=False, min_cell_number=1,
#                                      pval_method='permutation_test_fdr', pval_cutoff=0.05)
#     print('Number of mCCC detected by enzyme and sensor co-expression: ',
#           commu_res.shape[0])
#     mebocost.save_obj(obj=mebo_obj, path='mebocost_cell_commu.pk')
    
#     return mebo_obj, commu_res


def omnipath_info():
    """
    Retrieve and filter ligand-receptor interaction data from OmniPath.

    This function downloads intercellular communication interactions from the OmniPath database,
    filters for ligand–receptor pairs, and extracts relevant metadata. The filtered ligand-receptor
    network is also saved as a CSV file (`omnipath_lr_network.csv`).

    Returns:
        pandas.DataFrame:
            A DataFrame containing filtered ligand–receptor interaction data with the following columns:
            - `genesymbol_intercell_source`: Gene symbol of the ligand.
            - `genesymbol_intercell_target`: Gene symbol of the receptor.
            - `is_stimulation`: Boolean indicating whether the interaction is stimulatory.
            - `is_inhibition`: Boolean indicating whether the interaction is inhibitory.
            - `curation_effort`: A score reflecting manual curation effort.
            - `consensus_score_intercell_source`: Confidence score for the ligand.
            - `consensus_score_intercell_target`: Confidence score for the receptor.
            - `n_sources`: Total number of sources supporting the interaction.
            - `n_primary_sources`: Number of primary (original) sources.

    Notes:
        - Only interactions from the following OmniPath datasets are included:
          `'omnipath'`, `'pathwayextra'`, and `'ligrecextra'`.
        - The function specifically filters for interactions where the source is a ligand 
          and the target is a receptor.
        - The resulting network is saved to `'omnipath_lr_network.csv'` in the working directory.

    Raises:
        None explicitly, but assumes:
        - The `omnipath` Python package (`pypath`) is installed and accessible as `op`.
        - Internet access is available to fetch the data from the OmniPath web service.
    """
    
    intercell = op.interactions.import_intercell_network(
        include=['omnipath', 'pathwayextra', 'ligrecextra'])

    intercell_filtered = intercell[(intercell['category_intercell_source'] == 'ligand') &
                                   (intercell['category_intercell_target'] == 'receptor')]
    intercell_filtered.to_csv('omnipath_lr_network.csv')

    omnipath_data = intercell_filtered[['genesymbol_intercell_source',
                                        'genesymbol_intercell_target',
                                        'is_stimulation', 'is_inhibition',
                                        'curation_effort',
                                        'consensus_score_intercell_source',
                                        'consensus_score_intercell_target',
                                        'n_sources',
                                        'n_primary_sources']]
    
    return omnipath_data


def prot_to_gene(protein):
    """
    Convert a UniProt protein identifier to a gene symbol.

    This function maps a given protein name or UniProt ID to its corresponding gene symbol
    using a name mapping utility (e.g., from the OmniPath `pypath` package).

    Args:
        protein (str): 
            A UniProt protein ID or name to be mapped.

    Returns:
        str or None:
            The corresponding gene symbol if found; otherwise, returns `None`.
    """
    
    gene = mapping.map_name(protein, 'uniprot', 'genesymbol')
    
    return gene


class nichecompass_ccc:
    """
    A class wrapper for training and interpreting the NicheCompass model 
    for spatially-resolved gene program analysis in single-cell or spatial transcriptomics data.

    This class integrates prior biological knowledge (OmniPath, NicheNet, MeBoCoSt) and trains 
    a graph-based model to discover spatial gene programs and niches.

    Attributes:
        species (str): The species name (e.g., 'human', 'mouse').
        n_neighbors (int): Number of neighbors for spatial graph construction.
        conv_layer_encoder (str): Graph convolution layer type (e.g., 'gatv2conv').
        active_gp_thresh_ratio (float): Threshold for selecting active gene programs.
        n_epochs (int): Number of training epochs for sparse model.
        lr (float): Learning rate for optimization.
        cell_type_key (str): Key in AnnData `.obs` for cell type annotations.
        sample_key (str): Key in AnnData `.obs` for sample ID annotations.
        spot_size (float): Spot size for spatial plots.
        latent_leiden_resolution (float): Resolution for Leiden clustering in latent space.
        model_folder_path (str): Path to save trained model and AnnData.
        figure_folder_path (str): Path to save generated figures.
    """
    
    def __init__(self, species, n_neighbors, conv_layer_encoder, 
                 active_gp_thresh_ratio, n_epochs, lr, cell_type_key, 
                 sample_key, spot_size, latent_leiden_resolution, 
                 model_folder_path, figure_folder_path):
        
        self.species = species
        self.n_neighbors = n_neighbors
        self.conv_layer_encoder = conv_layer_encoder  # "gatv2conv" if enough compute and memory
        self.active_gp_thresh_ratio = active_gp_thresh_ratio
        self.n_epochs = n_epochs
        self.lr = lr
        self.cell_type_key = cell_type_key
        self.sample_key = sample_key
        self.spot_size = spot_size
        self.latent_leiden_resolution = latent_leiden_resolution
        self.model_folder_path = model_folder_path
        self.figure_folder_path = figure_folder_path
        
        self.spatial_key = "spatial"
        self.counts_key = "counts"
        self.adj_key = "spatial_connectivities"
        self.gp_names_key = "nichecompass_gp_names"
        self.active_gp_names_key = "nichecompass_active_gp_names"
        self.gp_targets_mask_key = "nichecompass_gp_targets"
        self.gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
        self.gp_sources_mask_key = "nichecompass_gp_sources"
        self.gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
        self.latent_key = "nichecompass_latent"
        self.n_epochs_all_gps = 25
        self.lambda_edge_recon = 500000.
        self.lambda_gene_expr_recon = 300.
        self.lambda_l1_masked = 0.  # prior GP  regularization
        self.lambda_l1_addon = 30.  # de novo GP regularization
        self.edge_batch_size = 1024  # increase if more memory available or decrease to save memory
        self.n_sampled_neighbors = 4
        self.use_cuda_if_available = True
        self.latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
        self.differential_gp_test_results_key = "nichecompass_differential_gp_test_results"
        self.artifacts_folder_path = "nichecompass_data/artifacts"
        self.save_figs = True
        self.save_file = True
        self.selected_cats = None
        self.comparison_cats = "rest"
        
        os.makedirs(self.model_folder_path, exist_ok=True)
        os.makedirs(self.figure_folder_path, exist_ok=True)
        
        
    def omnipath(self):
        """
        Loads ligand-receptor interaction data from OmniPath, filters for ligand-to-receptor 
        interactions, saves the filtered network, and constructs a gene program dictionary.
    
        The method imports the intercellular network data from OmniPath including additional 
        resources, filters interactions where the source is a ligand and the target is a receptor, 
        and then generates gene programs based on these filtered interactions. Optionally, 
        gene count distributions are plotted and saved.
    
        Returns:
            dict: A dictionary of gene programs extracted from OmniPath ligand-receptor interactions, 
                  with gene programs as keys and corresponding gene sets as values.
        """

        intercell = op.interactions.import_intercell_network(
            include=['omnipath', 'pathwayextra', 'ligrecextra'])
        intercell_filtered = intercell[(intercell['category_intercell_source'] == 'ligand') &
                                       (intercell['category_intercell_target'] == 'receptor')]
        intercell_filtered.to_csv('omnipath_lr_network.csv')

        omnipath_lr_network_file_path = "omnipath_lr_network.csv"
        omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
            species=self.species,
            load_from_disk=True, save_to_disk=True, 
            lr_network_file_path=omnipath_lr_network_file_path,
            plot_gp_gene_count_distributions=True,
            gp_gene_count_distributions_save_path=f"{self.figure_folder_path}/omnipath_gp_gene_count_distributions.svg")
        
        return omnipath_gp_dict
    
    
    def nichenet(self):
        """
        Loads ligand-receptor and ligand-target interaction data from NicheNet v2 
        and constructs a gene program dictionary.
    
        This method uses curated NicheNet v2 resources to extract ligand-receptor 
        and ligand-target interactions, and generates gene programs representing 
        signaling cascades. The resulting gene programs are optionally filtered 
        and saved to disk.
    
        Returns:
            dict: A dictionary of gene programs, where each key represents a ligand (gene program name) 
                  and the value is a set of associated target genes inferred from NicheNet data.
        """
        
        nichenet_lr_network_file_path = "nichenetv2/lr_network_human_21122021.csv"
        nichenet_ligand_target_matrix_file_path = "nichenetv2/ligand_target_matrix_nsga2r_final.csv"
        nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
            species=self.species, version="v2", keep_target_genes_ratio=1., 
            max_n_target_genes_per_gp=250, load_from_disk=True, save_to_disk=True,
            lr_network_file_path=nichenet_lr_network_file_path,
            ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
            plot_gp_gene_count_distributions=True)
        
        return nichenet_gp_dict
        
        
    def mebocostr(self):
        """
        Loads enzyme-sensor metabolite interaction data from MeBoCoSt and constructs 
        a gene program dictionary.
    
        This method reads interaction data from a specified directory, processes 
        the data based on the selected species, and generates gene programs 
        representing enzyme-sensor regulatory modules.
    
        Returns:
            dict: A dictionary where keys are gene program names and values are associated gene sets 
                  extracted from MeBoCoSt metabolite interaction data.
        """
        
        mebocost_enzyme_sensor_interactions_folder_path = "metabolites"
        mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
            dir_path=mebocost_enzyme_sensor_interactions_folder_path,
            species=self.species, plot_gp_gene_count_distributions=True)
        
        return mebocost_gp_dict
    
    
    def combination_dict(self):
        """
        Combines gene program dictionaries from multiple sources (OmniPath, NicheNet, and MeBoCoSt),
        and filters out overlapping or redundant gene programs.
    
        This method:
            - Calls internal methods to retrieve gene program dictionaries from OmniPath, NicheNet, and MeBoCoSt.
            - Merges and filters these dictionaries using `filter_and_combine_gp_dict_gps_v2()`.
    
        Returns:
            dict: A combined and filtered gene program dictionary suitable for model training.
        """
        
        omnipath_gp_dict = self.omnipath()
        nichenet_gp_dict = self.nichenet()
        mebocost_gp_dict = self.mebocostr()
        gp_dicts = [omnipath_gp_dict, nichenet_gp_dict, mebocost_gp_dict]
        combined_gp_dict = filter_and_combine_gp_dict_gps_v2(gp_dicts, verbose=True)
        
        return combined_gp_dict
            
    
    def training(self, adata):
        """
        Trains the NicheCompass model using spatial transcriptomics data and a combined gene program dictionary.
    
        This method performs the following steps:
            - Computes spatial neighbors for the input data.
            - Symmetrizes the spatial adjacency matrix.
            - Adds gene programs (GPs) to the AnnData object from a predefined dictionary.
            - Initializes the NicheCompass model.
            - Trains the model on gene expression and spatial adjacency data.
            - Computes a latent neighbor graph and UMAP embedding.
            - Saves the trained model and AnnData object to disk.
    
        Args:
            adata (AnnData): Annotated data object containing spatial transcriptomics data,
                             including coordinates in `adata.obsm[self.spatial_key]`.
    
        Returns:
            None
    
        Saves:
            - Trained model and modified `AnnData` object to `self.model_folder_path/adata_NicheCompass.h5ad`.
    
        Notes:
            - Requires configuration variables like `self.adj_key`, `self.counts_key`, `self.gp_names_key`,
              `self.latent_key`, `self.lr`, `self.n_epochs`, etc. to be pre-defined.
            - The training process uses GPU if `self.use_cuda_if_available` is set to True and CUDA is available.
        """
        
        # Compute spatial neighborhood
        sq.gr.spatial_neighbors(adata, coord_type="generic", 
                                spatial_key=self.spatial_key, n_neighs=self.n_neighbors)

        # Make adjacency matrix symmetric
        combined_gp_dict = self.combination_dict()
        adata.obsp[self.adj_key] = (adata.obsp[self.adj_key].maximum(adata.obsp[self.adj_key].T))
        add_gps_from_gp_dict_to_adata(gp_dict=combined_gp_dict, adata=adata,
                                      gp_targets_mask_key=self.gp_targets_mask_key,
                                      gp_targets_categories_mask_key=self.gp_targets_categories_mask_key,
                                      gp_sources_mask_key=self.gp_sources_mask_key,
                                      gp_sources_categories_mask_key=self.gp_sources_categories_mask_key,
                                      gp_names_key=self.gp_names_key, min_genes_per_gp=2,  min_source_genes_per_gp=1,
                                      min_target_genes_per_gp=1, max_genes_per_gp=None, max_source_genes_per_gp=None,
                                      max_target_genes_per_gp=None)

        # Initialize model
        self.model = NicheCompass(adata, counts_key=self.counts_key, adj_key=self.adj_key,
                             gp_names_key=self.gp_names_key, active_gp_names_key=self.active_gp_names_key,
                             gp_targets_mask_key=self.gp_targets_mask_key,
                             gp_targets_categories_mask_key=self.gp_targets_categories_mask_key,
                             gp_sources_mask_key=self.gp_sources_mask_key,
                             gp_sources_categories_mask_key=self.gp_sources_categories_mask_key,
                             latent_key=self.latent_key, conv_layer_encoder=self.conv_layer_encoder,
                             active_gp_thresh_ratio=self.active_gp_thresh_ratio)

        # Train model
        self.model.train(n_epochs=self.n_epochs, n_epochs_all_gps=self.n_epochs_all_gps,
                    lr=self.lr, lambda_edge_recon=self.lambda_edge_recon,
                    lambda_gene_expr_recon=self.lambda_gene_expr_recon,
                    lambda_l1_masked=self.lambda_l1_masked, 
                    edge_batch_size=self.edge_batch_size,
                    n_sampled_neighbors=self.n_sampled_neighbors,
                    use_cuda_if_available=self.use_cuda_if_available, verbose=False)

        # Compute latent neighbor graph
        sc.pp.neighbors(self.model.adata, use_rep=self.latent_key, key_added=self.latent_key)
        sc.tl.umap(self.model.adata, neighbors_key=self.latent_key)
        self.model.save(dir_path=self.model_folder_path, overwrite=True, save_adata=True,
                   adata_file_name="adata_NicheCompass.h5ad")

        self.samples = self.model.adata.obs[self.sample_key].unique().tolist()
        
        return None
    

    def plot(self):
        """
        Generate and save visualizations of cell types and niches in both latent and physical space,
        and plot the cell type composition of niches.
    
        This method performs the following:
            - Visualizes cell type annotations in latent (UMAP) and physical (spatial) space.
            - Computes Leiden clustering in latent space to define niches.
            - Visualizes niche annotations in latent and physical space.
            - Plots stacked bar charts showing the distribution of cell types within each niche.
            - Saves all generated plots to `self.figure_folder_path` if `self.save_figs` is True.
    
        Returns:
            None
    
        Saves:
            - Cell types in latent/physical space: 
                `<figure_folder_path>/cell_types_latent_physical_space.svg`
            - Niches in latent/physical space:
                `<figure_folder_path>/res_<latent_leiden_resolution>_niches_latent_physical_space.svg`
            - Niche cell type composition:
                `<figure_folder_path>/res_<latent_leiden_resolution>_niche_composition.svg`
    
        Notes:
            - Colors are generated using `create_new_color_dict()` for both cell types and niches.
            - Plots are displayed using `matplotlib.pyplot.show()`.
            - Requires pre-initialized `self.model`, `self.samples`, `self.cell_type_key`, `self.latent_cluster_key`, etc.
        """
        
        cell_type_colors = create_new_color_dict(
            adata=self.model.adata, cat_key=self.cell_type_key)

        # Create plot of cell type annotations in physical and latent space
        groups = None
        file_path = f"{self.figure_folder_path}/cell_types_latent_physical_space.svg"

        fig = plt.figure(figsize=(12, 14))
        title = fig.suptitle(t="Cell Types in Latent and Physical Space", y=0.96,
                             x=0.55, fontsize=20)
        spec1 = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[
                                  1], height_ratios=[3, 2])
        spec2 = gridspec.GridSpec(ncols=len(self.samples), nrows=2, width_ratios=[1] * len(self.samples),
                                  height_ratios=[3, 2])
        axs = []
        axs.append(fig.add_subplot(spec1[0]))
        sc.pl.umap(adata=self.model.adata, color=[self.cell_type_key], groups=groups, palette=cell_type_colors,
                   title="Cell Types in Latent Space", ax=axs[0], show=False)
        for idx, sample in enumerate(self.samples):
            axs.append(fig.add_subplot(spec2[len(self.samples) + idx]))
            sc.pl.spatial(adata=self.model.adata[self.model.adata.obs[self.sample_key] == sample],
                          color=[self.cell_type_key], groups=groups, palette=cell_type_colors,
                          spot_size=self.spot_size, title=f"Cell Types in Physical Space (Sample: {sample})",
                legend_loc=None, ax=axs[idx+1], show=False)

        # Create and position shared legend
        handles, labels = axs[0].get_legend_handles_labels()
        lgd = fig.legend(handles, labels, loc="center left",
                         bbox_to_anchor=(0.98, 0.5))
        axs[0].get_legend().remove()

        # Adjust, save and display plot
        plt.subplots_adjust(wspace=0.2, hspace=0.25)
        if self.save_figs:
            fig.savefig(file_path, bbox_extra_artists=(
                lgd, title), bbox_inches="tight")
        plt.show()
            
        # Compute latent Leiden clustering
        sc.tl.leiden(adata=self.model.adata, resolution=self.latent_leiden_resolution,
                     key_added=self.latent_cluster_key, neighbors_key=self.latent_key)
        self.latent_cluster_colors = create_new_color_dict(
            adata=self.model.adata, cat_key=self.latent_cluster_key)

        # Create plot of latent cluster / niche annotations in physical and latent space
        # set this to a specific cluster for easy visualization, e.g. ["17"]
        file_path = f"{self.figure_folder_path}/res_{self.latent_leiden_resolution}_niches_latent_physical_space.svg"

        fig = plt.figure(figsize=(12, 14))
        title = fig.suptitle(t="NicheCompass Niches in Latent and Physical Space",
                             y=0.96, x=0.55, fontsize=20)
        spec1 = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[
                                  1], height_ratios=[3, 2])
        spec2 = gridspec.GridSpec(ncols=len(self.samples), nrows=2,
                                  width_ratios=[1] * len(self.samples), height_ratios=[3, 2])
        axs = []
        axs.append(fig.add_subplot(spec1[0]))
        sc.pl.umap(adata=self.model.adata, color=[self.latent_cluster_key], groups=groups,
                   palette=self.latent_cluster_colors, title="Niches in Latent Space",
                   ax=axs[0], show=False)
        for idx, sample in enumerate(self.samples):
            axs.append(fig.add_subplot(spec2[len(self.samples) + idx]))
            sc.pl.spatial(adata=self.model.adata[self.model.adata.obs[self.sample_key] == sample],
                          color=[self.latent_cluster_key], groups=groups, 
                          palette=self.latent_cluster_colors, spot_size=self.spot_size, 
                          title="Niches in Physical Space (Sample: {sample})",
                          legend_loc=None, ax=axs[idx+1], show=False)

        # Create and position shared legend
        handles, labels = axs[0].get_legend_handles_labels()
        lgd = fig.legend(handles, labels, loc="center left",
                         bbox_to_anchor=(0.98, 0.5))
        axs[0].get_legend().remove()

        # Adjust, save and display plot
        plt.subplots_adjust(wspace=0.2, hspace=0.25)
        if self.save_figs:
            fig.savefig(file_path, bbox_extra_artists=(lgd, title), bbox_inches="tight")
        plt.show()

        file_path = f"{self.figure_folder_path}/res_{self.latent_leiden_resolution}_niche_composition.svg"

        df_counts = (self.model.adata.obs.groupby([self.latent_cluster_key, self.cell_type_key])
                     .size().unstack())
        df_counts.plot(kind="bar", stacked=True, figsize=(10, 10))
        legend = plt.legend(bbox_to_anchor=(1, 1), loc="upper left", prop={'size': 10})
        legend.set_title("Cell Type Annotations", prop={'size': 10})
        plt.title("Cell Type Composition of Niches")
        plt.xlabel("Niche")
        plt.ylabel("Cell Counts")
        if self.save_figs:
            plt.savefig(file_path, bbox_extra_artists=(legend,), bbox_inches="tight")
            
        return None
            
            
    def ccc(self):
        """
        Performs differential analysis of gene programs (GPs) across niches, visualizes enriched GPs, 
        and saves summaries of enriched gene programs.
    
        The method:
          - Retrieves active gene programs from the model.
          - Runs differential gene program testing between selected and comparison categories.
          - Visualizes the activity of enriched gene programs as a heatmap.
          - Saves a summary CSV file of enriched gene programs if `save_file` is enabled.
    
        Returns:
            tuple:
                enriched_gp_summary_df (pd.DataFrame): Summary of enriched gene programs after filtering and sorting.
                gp_summary_df (pd.DataFrame): Full gene program summary from the model.
                df (pd.DataFrame): Mean GP activities per niche used for heatmap visualization.
        """
        
        # Check number of active GPs
        active_gps = self.model.get_active_gps()
        print(f"Number of total gene programs: {len(self.model.adata.uns[self.gp_names_key])}.")
        print(f"Number of active gene programs: {len(active_gps)}.")

        # Display example active GPs
        gp_summary_df = self.model.get_gp_summary()
        gp_summary_df[gp_summary_df["gp_active"] == True].head()

        # Set parameters for differential gp testing
        self.log_bayes_factor_thresh = 2.3
    
        file_path = "{figure_folder_path}/og_bayes_factor_{log_bayes_factor_thresh}_niches_enriched_gps_heatmap.svg"

        # Run differential gp testing
        enriched_gps = self.model.run_differential_gp_tests(
            cat_key=self.latent_cluster_key, selected_cats=self.selected_cats,
            comparison_cats=self.comparison_cats, log_bayes_factor_thresh=self.log_bayes_factor_thresh)

        # Results are stored in a df in the adata object
        self.model.adata.uns[self.differential_gp_test_results_key]

        # Visualize GP activities of enriched GPs across niches
        df = self.model.adata.obs[[self.latent_cluster_key] +
                             enriched_gps].groupby(self.latent_cluster_key).mean()

        scaler = MinMaxScaler()
        normalized_columns = scaler.fit_transform(df)
        normalized_df = pd.DataFrame(normalized_columns, columns=df.columns)
        normalized_df.index = df.index

        plt.figure(figsize=(16, 8))  # Set the figure size
        sns.heatmap(normalized_df, cmap='viridis', annot=False, linewidths=0)
        plt.xticks(rotation=45, fontsize=8, ha="right")
        plt.xlabel("Gene Programs", fontsize=16)
        plt.savefig(f"{self.figure_folder_path}/enriched_gps_heatmap.svg",
                    bbox_inches="tight")

        # Store gene program summary of enriched gene programs
        
        file_path = f"{self.figure_folder_path}/" \
            f"/log_bayes_factor_{self.log_bayes_factor_thresh}_" \
            "niche_enriched_gps_summary.csv"

        gp_summary_cols = ["gp_name", "n_source_genes", "n_non_zero_source_genes",
                           "n_target_genes", "n_non_zero_target_genes", "gp_source_genes",
                           "gp_target_genes", "gp_source_genes_importances",
                           "gp_target_genes_importances"]

        enriched_gp_summary_df = gp_summary_df[gp_summary_df["gp_name"].isin(
            enriched_gps)]
        cat_dtype = pd.CategoricalDtype(categories=enriched_gps, ordered=True)
        enriched_gp_summary_df.loc[:, "gp_name"] = enriched_gp_summary_df["gp_name"].astype(
            cat_dtype)
        enriched_gp_summary_df = enriched_gp_summary_df.sort_values(by="gp_name")
        enriched_gp_summary_df = enriched_gp_summary_df[gp_summary_cols]

        if self.save_file:
            enriched_gp_summary_df.to_csv(f"{file_path}")
            
        plot_label = f"log_bayes_factor_{self.log_bayes_factor_thresh}_cluster_" \
                 f"{self.selected_cats[0] if self.selected_cats else 'None'}_vs_rest"

        generate_enriched_gp_info_plots(
            plot_label=plot_label,
            model=self.model,
            sample_key=self.sample_key,
            differential_gp_test_results_key=self.differential_gp_test_results_key,
            cat_key=self.latent_cluster_key,
            cat_palette=self.latent_cluster_colors,
            n_top_enriched_gp_start_idx=0,
            n_top_enriched_gp_end_idx=10,
            feature_spaces=self.samples,
            n_top_genes_per_gp=3,
            save_figs=self.save_figs,
            figure_folder_path=f"{self.figure_folder_path}/",
            spot_size=self.spot_size)

        return enriched_gp_summary_df, gp_summary_df, df
    
    
    def plot_ccc(self, gp_name):
        """
        Generates and visualizes enriched gene program (GP) plots and communication GP networks for a given gene program.
    
        Args:
            gp_name (str): Name of the gene program to visualize in the communication network.
    
        Returns:
            None
        """
    
        network_df = compute_communication_gp_network(
            gp_list=[gp_name],
            model=self.model,
            group_key=self.latent_cluster_key,
            n_neighbors=self.n_neighbors)
    
        visualize_communication_gp_network(
            adata=self.model.adata,
            network_df=network_df,
            figsize=(10, 7),
            cat_colors=self.latent_cluster_colors,
            edge_type_colors=["#1f77b4"],
            cat_key=self.latent_cluster_key,
            save=True,
            save_path=f"{self.figure_folder_path}/gp_network_{gp_name}.svg")
        
        return None