#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 
# Metabric: Cell typing
#
# =============================================================================
import os
path = "/scratch/users/k22026807/masters/project/celltyping/"
os.chdir(path)
import logging
import warnings
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import spatialdata as sd
import celltypist
from celltypist import annotate
from pathlib import Path
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from workshop_lib import (
    annotate_cells_with_marker_scores,
    genename_to_ensg,
    save_figure,
    create_marker_dict,
    plot_matrix_visualizations,
    plot_celltype_distribution,
    save_confidence_statistics,
    plot_celltype_spatial,
    get_custom_palette
    )

# =============================================================================
# Configuration
# =============================================================================
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

# Set plotting parameters globally
plt.rcParams['font.size'] = 5
plt.rcParams['axes.labelsize'] = 5
plt.rcParams['xtick.labelsize'] = 3
plt.rcParams['ytick.labelsize'] = 3

ZARR_DIR = "../xenium_output/BC_prime.zarr"
MODEL_NAME = 'breast_cancer_combined_cell_atlas'
TenX = "gene_list_10X.csv"
biomarkers_csv = "breast_cancer_biomarkers.csv"
biomarkers_matrix_csv = "biomarker_cell_type_matrix.csv"

# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================
def calculate_optimal_threshold(confidence_scores, percentile=25):
    """
    Calculate threshold based on distribution.
    
    Parameters:
    -----------
    confidence_scores : array-like
        Confidence scores to analyze
    percentile : int
        Percentile to use as threshold (default: 25th percentile)
    
    Returns:
    --------
    float : Optimal threshold value
    """
    threshold = np.percentile(confidence_scores, percentile)
    print(f"Calculated threshold at {percentile}th percentile: {threshold:.3f}")
    
    return threshold


def create_marker_dict_with_validation(biomarker_df, adata, palette):
    """
    Create marker dictionary with validation of available genes.
    
    Parameters:
    -----------
    biomarker_df : DataFrame
        Matrix of biomarkers (genes x cell types)
    adata : AnnData
        Annotated data object to check gene availability
    palette : dict
        Color palette for cell types
    
    Returns:
    --------
    dict : Marker dictionary
    list : Cell types with all markers present
    list : Cell types with missing markers
    """
    marker_dict = {}
    missing_markers = []
    complete_cell_types = []
    
    for cell_type in biomarker_df.columns:
        markers = biomarker_df[biomarker_df[cell_type] > 0].index.tolist()
        available_markers = [m for m in markers if m in adata.var_names]
        missing = set(markers) - set(available_markers)
        
        if missing:
            missing_markers.append((cell_type, missing))
            print(f"⚠️  {cell_type}: {len(missing)} missing markers")
        else:
            complete_cell_types.append(cell_type)
        
        marker_dict[cell_type] = available_markers
    
    print(f"\n{len(complete_cell_types)} cell types with complete marker sets")
    print(f"⚠️  {len(missing_markers)} cell types with missing markers")
    
    return marker_dict, complete_cell_types, missing_markers


def refine_with_neighborhood_consensus(
    adata,
    umap_key='X_umap',
    cell_type_key='cell_type',
    confidence_key='conf_score',
    low_confidence_threshold=0.5,
    n_neighbors=15,
    consensus_threshold=0.6
):
    """
    Refine low-confidence cell type assignments using neighborhood consensus
    in UMAP space.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object with UMAP coordinates
    umap_key : str
        Key for UMAP coordinates in adata.obsm
    cell_type_key : str
        Column name for cell type annotations
    confidence_key : str
        Column name for confidence scores
    low_confidence_threshold : float
        Threshold below which to apply neighborhood refinement
    n_neighbors : int
        Number of neighbors to consider
    consensus_threshold : float
        Minimum fraction of neighbors agreeing for reassignment
        
    Returns:
    --------
    AnnData : Updated annotated data object
    """
    print("\n[Neighborhood Refinement] Starting...")
    
    # Get low confidence cells
    if confidence_key not in adata.obs.columns:
        print(f"Warning: {confidence_key} not found, using default confidence of 0.5")
        adata.obs[confidence_key] = 0.5
    
    low_conf_mask = adata.obs[confidence_key] < low_confidence_threshold
    n_low_conf = low_conf_mask.sum()
    print(f"Found {n_low_conf} low-confidence cells ({n_low_conf/len(adata)*100:.1f}%)")
    
    if n_low_conf == 0:
        print("No low-confidence cells to refine")
        adata.obs['cell_type_refined'] = adata.obs[cell_type_key].copy()
        adata.obs['refinement_method'] = 'original'
        adata.obs['neighborhood_consensus'] = 1.0
        return adata
    
    # Fit KNN on UMAP space
    if umap_key not in adata.obsm:
        print(f"Error: {umap_key} not found in adata.obsm")
        adata.obs['cell_type_refined'] = adata.obs[cell_type_key].copy()
        adata.obs['refinement_method'] = 'original'
        adata.obs['neighborhood_consensus'] = 0.0
        return adata
    
    umap_coords = adata.obsm[umap_key]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean').fit(umap_coords)
    distances, indices = nbrs.kneighbors(umap_coords)
    
    # Create new columns for refined annotations
    adata.obs['cell_type_refined'] = adata.obs[cell_type_key].copy()
    adata.obs['refinement_method'] = 'original'
    adata.obs['neighborhood_consensus'] = 0.0
    
    reassigned_count = 0
    reassignment_details = []
    
    # For each low-confidence cell
    for idx in np.where(low_conf_mask)[0]:
        # Get neighbors (excluding self)
        neighbor_indices = indices[idx, 1:]
        neighbor_cells = adata.obs.iloc[neighbor_indices]
        
        # Get neighbor cell types and their confidence scores
        neighbor_types = neighbor_cells[cell_type_key]
        neighbor_confidences = neighbor_cells[confidence_key]
        
        # Weight by confidence (high-confidence neighbors matter more)
        weighted_votes = {}
        for cell_type, conf in zip(neighbor_types, neighbor_confidences):
            weight = max(conf, 0.1)  # Minimum weight to avoid zero
            weighted_votes[cell_type] = weighted_votes.get(cell_type, 0) + weight
        
        # Find consensus cell type
        if weighted_votes:
            total_weight = sum(weighted_votes.values())
            consensus_type = max(weighted_votes, key=weighted_votes.get)
            consensus_score = weighted_votes[consensus_type] / total_weight
            
            adata.obs.loc[adata.obs.index[idx], 'neighborhood_consensus'] = consensus_score
            
            # Reassign if consensus is strong enough
            original_type = adata.obs.loc[adata.obs.index[idx], cell_type_key]
            if consensus_score >= consensus_threshold and consensus_type != original_type:
                adata.obs.loc[adata.obs.index[idx], 'cell_type_refined'] = consensus_type
                adata.obs.loc[adata.obs.index[idx], 'refinement_method'] = 'neighborhood'
                reassigned_count += 1
                reassignment_details.append({
                    'original': original_type,
                    'refined': consensus_type,
                    'consensus': consensus_score
                })
    
    print(f"Reassigned {reassigned_count} cells based on neighborhood consensus")
    if n_low_conf > 0:
        print(f"({reassigned_count/n_low_conf*100:.1f}% of low-confidence cells)")
    
    # Summary of reassignments
    if reassignment_details:
        reassign_df = pd.DataFrame(reassignment_details)
        print("\n  Reassignment summary:")
        for orig in reassign_df['original'].unique():
            subset = reassign_df[reassign_df['original'] == orig]
            print(f"{orig}:")
            for refined, count in subset['refined'].value_counts().items():
                print(f"  → {refined}: {count}")
    
    return adata


def integrate_celltypist_and_biomarker_annotations(
    adata,
    biomarker_adata,
    celltypist_conf_threshold=0.7,
    biomarker_conf_threshold=0.5,
    use_neighborhood_refinement=True,
    n_neighbors=15,
    consensus_threshold=0.6
):
    """
    Integrate CellTypist and biomarker annotations intelligently.
    
    Strategy:
    1. High CellTypist confidence → Keep CellTypist
    2. High biomarker confidence → Use biomarker
    3. Low both → Use neighborhood consensus in UMAP space
    
    Parameters:
    -----------
    adata : AnnData
        Main annotated data with CellTypist annotations
    biomarker_adata : AnnData
        Annotated data with biomarker scores
    celltypist_conf_threshold : float
        Confidence threshold for CellTypist annotations
    biomarker_conf_threshold : float
        Confidence threshold for biomarker annotations
    use_neighborhood_refinement : bool
        Whether to use neighborhood consensus for low-confidence cells
    n_neighbors : int
        Number of neighbors for consensus
    consensus_threshold : float
        Minimum consensus score for reassignment
        
    Returns:
    --------
    AnnData : Integrated annotated data object
    """
    print("\n[Integration] Combining CellTypist and biomarker annotations...")
    
    adata_integrated = adata.copy()
    
    # Get confidence scores
    if 'conf_score' in adata.obs.columns:
        celltypist_conf = adata.obs['conf_score']
    elif 'majority_voting_score' in adata.obs.columns:
        celltypist_conf = adata.obs['majority_voting_score']
    else:
        print("Warning: No confidence scores found, using default value of 0.75")
        celltypist_conf = pd.Series(0.75, index=adata.obs.index)
    
    # Calculate biomarker confidence (max score across all cell types)
    biomarker_cols = [col for col in biomarker_adata.obs.columns if col.endswith('_score')]
    if biomarker_cols:
        biomarker_conf = biomarker_adata.obs[biomarker_cols].max(axis=1)
    else:
        print("Warning: No biomarker scores found, using default value of 0.0")
        biomarker_conf = pd.Series(0, index=biomarker_adata.obs.index)
    
    # ===================================================================
    # FIX: Convert categorical to regular string to avoid category errors
    # ===================================================================
    if pd.api.types.is_categorical_dtype(adata.obs['majority_voting']):
        adata_integrated.obs['cell_type_final'] = adata.obs['majority_voting'].astype(str).copy()
    else:
        adata_integrated.obs['cell_type_final'] = adata.obs['majority_voting'].copy()
    
    adata_integrated.obs['annotation_source'] = 'celltypist'
    adata_integrated.obs['celltypist_confidence'] = celltypist_conf.values
    adata_integrated.obs['biomarker_confidence'] = biomarker_conf.values
    
    # Decision logic
    high_celltypist = celltypist_conf >= celltypist_conf_threshold
    high_biomarker = biomarker_conf >= biomarker_conf_threshold
    
    # Use biomarker when it has high confidence and CellTypist doesn't
    use_biomarker_mask = high_biomarker & ~high_celltypist
    
    if 'cell_type' in biomarker_adata.obs.columns:
        # Convert biomarker cell types to string as well
        if pd.api.types.is_categorical_dtype(biomarker_adata.obs['cell_type']):
            biomarker_cell_types = biomarker_adata.obs['cell_type'].astype(str)
        else:
            biomarker_cell_types = biomarker_adata.obs['cell_type']
        
        # Assign biomarker annotations where mask is True
        adata_integrated.obs.loc[use_biomarker_mask, 'cell_type_final'] = \
            biomarker_cell_types.loc[use_biomarker_mask].values
        adata_integrated.obs.loc[use_biomarker_mask, 'annotation_source'] = 'biomarker'
    
    print("\n Annotation decision breakdown:")
    print(f"CellTypist (high conf): {high_celltypist.sum()} ({high_celltypist.sum()/len(adata)*100:.1f}%)")
    print(f"Biomarker override: {use_biomarker_mask.sum()} ({use_biomarker_mask.sum()/len(adata)*100:.1f}%)")
    print(f"Low confidence both: {(~high_celltypist & ~high_biomarker).sum()} ({(~high_celltypist & ~high_biomarker).sum()/len(adata)*100:.1f}%)")
    
    # Apply neighborhood refinement for low-confidence cells
    if use_neighborhood_refinement:
        # Combined confidence score
        adata_integrated.obs['combined_confidence'] = np.maximum(
            celltypist_conf.values, 
            biomarker_conf.values
        )
        
        adata_integrated = refine_with_neighborhood_consensus(
            adata_integrated,
            cell_type_key='cell_type_final',
            confidence_key='combined_confidence',
            low_confidence_threshold=max(celltypist_conf_threshold, biomarker_conf_threshold),
            n_neighbors=n_neighbors,
            consensus_threshold=consensus_threshold
        )
        
        # Update final annotations and source
        neighborhood_refined = adata_integrated.obs['refinement_method'] == 'neighborhood'
        
        # Make sure cell_type_refined is also string type
        if 'cell_type_refined' in adata_integrated.obs.columns:
            if pd.api.types.is_categorical_dtype(adata_integrated.obs['cell_type_refined']):
                adata_integrated.obs['cell_type_refined'] = \
                    adata_integrated.obs['cell_type_refined'].astype(str)
            
            adata_integrated.obs.loc[neighborhood_refined, 'cell_type_final'] = \
                adata_integrated.obs.loc[neighborhood_refined, 'cell_type_refined'].values
            adata_integrated.obs.loc[neighborhood_refined, 'annotation_source'] = 'neighborhood'
            
            print(f"Neighborhood refined: {neighborhood_refined.sum()} ({neighborhood_refined.sum()/len(adata)*100:.1f}%)")
    
    print("\n  Final annotation sources:")
    print(adata_integrated.obs['annotation_source'].value_counts())
    
    return adata_integrated


def plot_annotation_comparison(adata_integrated, output_dir):
    """Create comparison plots between annotation methods."""
    print("\n[Visualization] Creating annotation comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    
    # 1. CellTypist original
    sc.pl.umap(adata_integrated, color='majority_voting', 
               title='CellTypist Original', ax=axes[0, 0], show=False,
               legend_loc='right margin', frameon=False)
    
    # 2. Biomarker-based
    if 'cell_type' in adata_integrated.obs.columns and \
       adata_integrated.obs['cell_type'].nunique() > 1:
        sc.pl.umap(adata_integrated, color='cell_type', 
                   title='Biomarker-based', ax=axes[0, 1], show=False,
                   legend_loc='right margin', frameon=False)
    else:
        axes[0, 1].text(0.5, 0.5, 'Biomarker data not available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Biomarker-based')
    
    # 3. Final integrated
    sc.pl.umap(adata_integrated, color='cell_type_final', 
               title='Integrated (Final)', ax=axes[0, 2], show=False,
               legend_loc='right margin', frameon=False)
    
    # 4. CellTypist confidence
    sc.pl.umap(adata_integrated, color='celltypist_confidence', 
               title='CellTypist Confidence', ax=axes[1, 0], show=False, 
               cmap='RdYlGn', vmin=0, vmax=1, frameon=False)
    
    # 5. Biomarker confidence
    sc.pl.umap(adata_integrated, color='biomarker_confidence', 
               title='Biomarker Confidence', ax=axes[1, 1], show=False, 
               cmap='RdYlGn', vmin=0, vmax=1, frameon=False)
    
    # 6. Annotation source
    sc.pl.umap(adata_integrated, color='annotation_source', 
               title='Annotation Source', ax=axes[1, 2], show=False,
               legend_loc='right margin', frameon=False)
    
    plt.tight_layout()
    save_figure("annotation_integration_comparison.png", output_dir, dpi=300)
    plt.close()
    
    return fig


def visualize_confidence_distribution(adata, output_dir):
    """Visualize confidence score distributions."""
    print("\n [Visualization] Creating confidence distribution plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # CellTypist confidence by cell type
    if 'celltypist_confidence' in adata.obs.columns and 'majority_voting' in adata.obs.columns:
        plot_data = adata.obs[['majority_voting', 'celltypist_confidence']].copy()
        sns.violinplot(data=plot_data, x='majority_voting', y='celltypist_confidence', 
                       ax=axes[0], inner='box')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].set_title('CellTypist Confidence by Cell Type')
        axes[0].set_ylabel('Confidence Score')
        axes[0].set_xlabel('Cell Type')
        if 'celltypist_confidence' in adata.obs.columns:
            threshold = adata.obs['celltypist_confidence'].quantile(0.25)
            axes[0].axhline(y=threshold, color='r', linestyle='--', 
                          label=f'25th percentile ({threshold:.2f})')
            axes[0].legend()
    
    # Biomarker confidence by cell type
    if 'biomarker_confidence' in adata.obs.columns and 'cell_type_final' in adata.obs.columns:
        plot_data = adata.obs[['cell_type_final', 'biomarker_confidence']].copy()
        sns.violinplot(data=plot_data, x='cell_type_final', y='biomarker_confidence', 
                       ax=axes[1], inner='box')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].set_title('Biomarker Confidence by Cell Type')
        axes[1].set_ylabel('Confidence Score')
        axes[1].set_xlabel('Cell Type')
        if 'biomarker_confidence' in adata.obs.columns:
            threshold = adata.obs['biomarker_confidence'].quantile(0.25)
            axes[1].axhline(y=threshold, color='r', linestyle='--', 
                          label=f'25th percentile ({threshold:.2f})')
            axes[1].legend()
    
    plt.tight_layout()
    save_figure("confidence_distributions.png", output_dir, dpi=300)
    plt.close()
    
    return None
    
    
def plot_refinement_summary(adata_integrated, output_dir):
    """Create summary plots showing the refinement process."""
    print("\n [Visualization] Creating refinement summary...")
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.4)
    
    # 1. Sankey-style flow diagram (simplified as stacked bar)
    ax1 = fig.add_subplot(gs[0, :2])
    
    if 'annotation_source' in adata_integrated.obs.columns:
        source_counts = adata_integrated.obs['annotation_source'].value_counts()
        colors = {'celltypist': '#3498db', 'biomarker': '#e74c3c', 
                  'neighborhood': '#2ecc71', 'original': '#95a5a6'}
        
        bars = ax1.barh(range(len(source_counts)), source_counts.values,
                       color=[colors.get(x, '#95a5a6') for x in source_counts.index])
        ax1.set_yticks(range(len(source_counts)))
        ax1.set_yticklabels(source_counts.index)
        ax1.set_xlabel('Number of Cells')
        ax1.set_title('Annotation Source Distribution')
        
        for i, (idx, val) in enumerate(source_counts.items()):
            ax1.text(val, i, f' {val} ({val/len(adata_integrated)*100:.1f}%)', 
                    va='center')
    
    # 2. Confidence score comparison
    ax2 = fig.add_subplot(gs[0, 2:])
    
    if all(col in adata_integrated.obs.columns for col in 
           ['celltypist_confidence', 'biomarker_confidence', 'annotation_source']):
        
        conf_data = []
        for source in ['celltypist', 'biomarker', 'neighborhood']:
            mask = adata_integrated.obs['annotation_source'] == source
            if mask.sum() > 0:
                conf_data.append({
                    'Source': source.capitalize(),
                    'CellTypist': adata_integrated.obs.loc[mask, 'celltypist_confidence'].mean(),
                    'Biomarker': adata_integrated.obs.loc[mask, 'biomarker_confidence'].mean()
                })
        
        if conf_data:
            conf_df = pd.DataFrame(conf_data)
            conf_df.set_index('Source').plot(kind='bar', ax=ax2, rot=0)
            ax2.set_ylabel('Mean Confidence Score')
            ax2.set_title('Average Confidence by Annotation Source')
            ax2.legend(title='Confidence Type')
            ax2.set_ylim([0, 1])
    
    # ===================================================================
    # FIX: 3. Cell type changes (convert categorical to string for comparison)
    # ===================================================================
    ax3 = fig.add_subplot(gs[1, :2])
    
    if all(col in adata_integrated.obs.columns for col in ['majority_voting', 'cell_type_final']):
        # Convert both columns to string to avoid categorical comparison issues
        majority_voting_str = adata_integrated.obs['majority_voting'].astype(str)
        cell_type_final_str = adata_integrated.obs['cell_type_final'].astype(str)
        
        changed = majority_voting_str != cell_type_final_str
        unchanged = ~changed
        
        change_data = pd.DataFrame({
            'Status': ['Unchanged', 'Changed'],
            'Count': [unchanged.sum(), changed.sum()],
            'Percentage': [unchanged.mean()*100, changed.mean()*100]
        })
        
        wedges, texts, autotexts = ax3.pie(change_data['Count'], 
                                            labels=change_data['Status'],
                                            autopct='%1.1f%%',
                                            colors=['#3498db', '#e74c3c'],
                                            startangle=90)
        ax3.set_title('Cell Type Assignment Changes')
    
    # ===================================================================
    # FIX: 4. Top cell type transitions (convert categorical to string)
    # ===================================================================
    ax4 = fig.add_subplot(gs[1, 2:])
    
    if all(col in adata_integrated.obs.columns for col in ['majority_voting', 'cell_type_final']):
        # Convert both columns to string
        majority_voting_str = adata_integrated.obs['majority_voting'].astype(str)
        cell_type_final_str = adata_integrated.obs['cell_type_final'].astype(str)
        
        # Find changed cells
        changed_mask = majority_voting_str != cell_type_final_str
        
        if changed_mask.sum() > 0:
            # Create a dataframe with the changes
            changed_data = pd.DataFrame({
                'majority_voting': majority_voting_str[changed_mask],
                'cell_type_final': cell_type_final_str[changed_mask]
            })
            
            transitions = changed_data.groupby(['majority_voting', 'cell_type_final']).size()
            top_transitions = transitions.nlargest(min(10, len(transitions)))
            
            y_pos = range(len(top_transitions))
            ax4.barh(y_pos, top_transitions.values, color='#e74c3c')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([f"{k[0]} → {k[1]}" for k in top_transitions.index], 
                               fontsize=9)
            ax4.set_xlabel('Number of Cells')
            ax4.set_title('Top 10 Cell Type Transitions')
            ax4.invert_yaxis()
        else:
            ax4.text(0.5, 0.5, 'No cell type changes', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Top 10 Cell Type Transitions')
    
    plt.suptitle('Annotation Refinement Summary', fontsize=16, y=0.98)
    save_figure("refinement_summary.png", output_dir, dpi=300)
    plt.close()
    
    return None
    
    
def validate_annotations(adata, output_dir, marker_dict=None):
    """Comprehensive validation of cell type annotations."""
    print("\n [Validation] Performing annotation quality checks...")
    
    validation_results = {}
    
    # ===================================================================
    # Convert all relevant columns to string at the start
    # ===================================================================
    adata_work = adata.copy()
    
    # Convert categorical columns to string
    categorical_cols = ['majority_voting', 'cell_type_final', 'cell_type', 
                       'annotation_source', 'predicted_labels']
    
    for col in categorical_cols:
        if col in adata_work.obs.columns:
            if pd.api.types.is_categorical_dtype(adata_work.obs[col]):
                adata_work.obs[col] = adata_work.obs[col].astype(str)
    
    # 1. Neighborhood purity (Silhouette score)
    if 'X_umap' in adata_work.obsm:
        # Determine which cell type column to use
        cell_type_col = None
        if 'cell_type_final' in adata_work.obs.columns:
            cell_type_col = 'cell_type_final'
        elif 'cell_type' in adata_work.obs.columns:
            cell_type_col = 'cell_type'
        elif 'majority_voting' in adata_work.obs.columns:
            cell_type_col = 'majority_voting'
        
        if cell_type_col is not None:
            try:
                # Create categorical codes for silhouette score
                cell_types_cat = pd.Categorical(adata_work.obs[cell_type_col])
                labels = cell_types_cat.codes
                
                if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                    sil_score = silhouette_score(adata_work.obsm['X_umap'], labels)
                    validation_results['silhouette_score'] = sil_score
                    print(f"Silhouette score (UMAP space): {sil_score:.3f}")
                    print("Higher is better, range: -1 to 1)")
                else:
                    print("Cannot calculate silhouette score: only one cell type")
            except Exception as e:
                print(f"Could not calculate silhouette score: {e}")
    
    # 2. Compare annotations (confusion matrix)
    if 'majority_voting' in adata_work.obs and 'cell_type_final' in adata_work.obs:
        try:
            # Both columns are already converted to string above
            confusion_df = pd.crosstab(
                adata_work.obs['majority_voting'],
                adata_work.obs['cell_type_final'],
                normalize='index'
            )
            
            # Save confusion matrix
            confusion_df.to_csv(output_dir / "annotation_confusion_matrix.csv")
            print("Saved: annotation_confusion_matrix.csv")
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # Handle large confusion matrices
            if confusion_df.shape[0] > 30 or confusion_df.shape[1] > 30:
                # For very large matrices, don't annotate
                sns.heatmap(confusion_df, fmt='.2f', cmap='Blues', 
                           ax=ax, cbar_kws={'label': 'Proportion'},
                           xticklabels=True, yticklabels=True)
            else:
                sns.heatmap(confusion_df, annot=True, fmt='.2f', cmap='Blues', 
                           ax=ax, cbar_kws={'label': 'Proportion'})
            
            ax.set_title('CellTypist vs Final Annotations\n(row-normalized)', fontsize=14)
            ax.set_xlabel('Final Annotation', fontsize=12)
            ax.set_ylabel('CellTypist Annotation', fontsize=12)
            
            # Rotate labels if many categories
            if confusion_df.shape[0] > 10:
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
            if confusion_df.shape[1] > 10:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
            plt.tight_layout()
            save_figure("annotation_confusion.png", output_dir, dpi=300)
            plt.close()
            
            # Calculate agreement
            agreement = (adata_work.obs['majority_voting'] == adata_work.obs['cell_type_final']).mean()
            validation_results['annotation_agreement'] = agreement
            print(f"Annotation agreement: {agreement:.1%}")
            
        except Exception as e:
            print(f"Could not create confusion matrix: {e}")
    
    # 3. Marker expression validation
    if marker_dict is not None:
        # Determine which cell type column to use
        cell_type_col = None
        if 'cell_type_final' in adata_work.obs.columns:
            cell_type_col = 'cell_type_final'
        elif 'cell_type' in adata_work.obs.columns:
            cell_type_col = 'cell_type'
        
        if cell_type_col is not None:
            print("\n Marker expression validation:")
            marker_validation = {}
            
            for cell_type, markers in marker_dict.items():
                # Check if this cell type exists
                if cell_type not in adata_work.obs[cell_type_col].values:
                    continue
                
                mask = adata_work.obs[cell_type_col] == cell_type
                n_cells = mask.sum()
                
                if n_cells == 0:
                    continue
                
                available_markers = [m for m in markers if m in adata_work.var_names]
                
                if available_markers:
                    try:
                        # Calculate mean expression
                        if hasattr(adata_work.X, 'toarray'):
                            in_celltype = adata_work[mask, available_markers].X.toarray().mean()
                            out_celltype = adata_work[~mask, available_markers].X.toarray().mean()
                        else:
                            in_celltype = adata_work[mask, available_markers].X.mean()
                            out_celltype = adata_work[~mask, available_markers].X.mean()
                        
                        fold_change = in_celltype / (out_celltype + 1e-10)
                        
                        marker_validation[cell_type] = {
                            'n_cells': int(n_cells),
                            'n_markers': len(available_markers),
                            'mean_expression_in': float(in_celltype),
                            'mean_expression_out': float(out_celltype),
                            'fold_change': float(fold_change)
                        }
                        print(f"{cell_type}: FC={fold_change:.2f} ({len(available_markers)} markers, {n_cells} cells)")
                    
                    except Exception as e:
                        print(f"Warning: Could not validate {cell_type}: {e}")
            
            # Save marker validation
            if marker_validation:
                marker_val_df = pd.DataFrame(marker_validation).T
                marker_val_df.to_csv(Path(output_dir) / "marker_expression_validation.csv")
                print("\n Saved: marker_expression_validation.csv")
    
    # 4. Cell type size distribution
    cell_type_col = None
    if 'cell_type_final' in adata_work.obs.columns:
        cell_type_col = 'cell_type_final'
    elif 'cell_type' in adata_work.obs.columns:
        cell_type_col = 'cell_type'
    elif 'majority_voting' in adata_work.obs.columns:
        cell_type_col = 'majority_voting'
    
    if cell_type_col is not None:
        celltype_counts = adata_work.obs[cell_type_col].value_counts()
        print("\n Cell type distribution:")
        for ct, count in celltype_counts.items():
            print(f"{ct}: {count} ({count/len(adata_work)*100:.1f}%)")
        
        celltype_counts.to_csv(Path(output_dir) / "celltype_distribution.csv")
        print("Saved: celltype_distribution.csv")
    
    # Save validation results
    if validation_results:
        import json
        with open(Path(output_dir) / "validation_metrics.json", 'w') as f:
            json.dump(validation_results, f, indent=2)
        print("\n Saved: validation_metrics.json")
    
    return validation_results


def plot_marker_expression_overlay(
    adata,
    markers_dict,
    basis='X_umap',
    ncols=3,
    figsize_per_plot=(5, 4.5),
    cmap='YlOrRd',
    vmin=None,
    vmax=None,
    output_dir=None,
    filename_prefix="marker_expression",
    show_title=True,
    point_size=None
):
    """
    Create marker expression overlay plots on UMAP/t-SNE coordinates.
    Similar to the reference figure style.
    """
    print("\n[Marker Visualization] Creating expression overlay plots...")
    
    # Get coordinates
    if basis not in adata.obsm:
        print(f"Error: {basis} not found in adata.obsm")
        return {}
    
    coords = adata.obsm[basis]
    basis_name = basis.replace('X_', '').upper()
    
    # Determine point size
    if point_size is None:
        point_size = 120000 / len(adata)
        point_size = max(1, min(point_size, 20))
    
    figures = {}
    
    # Process each cell type
    for cell_type, markers in markers_dict.items():
        # Filter to available markers
        available_markers = [m for m in markers if m in adata.var_names]
        
        if not available_markers:
            print(f"⚠️  Skipping {cell_type}: no markers available")
            continue
        
        print(f"Creating plots for {cell_type} ({len(available_markers)} markers)")
        
        # Calculate grid dimensions
        nrows = int(np.ceil(len(available_markers) / ncols))
        fig_width = figsize_per_plot[0] * ncols
        fig_height = figsize_per_plot[1] * nrows
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
        
        # Flatten axes
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
        
        # Plot each marker
        for idx, marker in enumerate(available_markers):
            ax = axes[idx]
            
            # Get expression values
            if hasattr(adata.X, 'toarray'):
                expr = adata[:, marker].X.toarray().flatten()
            else:
                expr = adata[:, marker].X.flatten()
            
            # Calculate normalization
            if vmax is None:
                marker_vmax = np.percentile(expr[expr > 0], 95) if (expr > 0).any() else expr.max()
            else:
                marker_vmax = vmax
            
            marker_vmin = vmin if vmin is not None else 0
            
            # Create scatter plot
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=expr,
                cmap=cmap,
                s=point_size,
                alpha=0.8,
                edgecolors='none',
                vmin=marker_vmin,
                vmax=marker_vmax,
                rasterized=True
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.046)
            cbar.ax.tick_params(labelsize=8)
            
            # Formatting
            if show_title:
                pct_expressing = (expr > 0).sum() / len(expr) * 100
                mean_expr = expr[expr > 0].mean() if (expr > 0).any() else 0
                ax.set_title(
                    f'{marker}\n{pct_expressing:.1f}% cells, μ={mean_expr:.2f}',
                    fontsize=10,
                    pad=8
                )
            
            ax.set_xlabel(f'{basis_name}_1', fontsize=9)
            ax.set_ylabel(f'{basis_name}_2', fontsize=9)
            ax.tick_params(labelsize=8)
            
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        # Hide empty subplots
        for idx in range(len(available_markers), len(axes)):
            axes[idx].set_visible(False)
        
        # Add super title
        fig.suptitle(
            f'{cell_type} Markers',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout()
        
        # Save figure
        if output_dir is not None:
            clean_name = cell_type.replace(' ', '_').replace('/', '_')
            filename = f"{filename_prefix}_{clean_name}.png"
            save_figure(filename, output_dir, dpi=300)
        
        figures[cell_type] = fig
        plt.close()
    
    return figures


def create_combined_marker_summary(
    adata,
    markers_dict,
    basis='X_umap',
    top_n_markers=3,
    figsize=(20, 12),
    output_dir=None,
    filename="marker_summary_combined.png"
):
    """Create a combined summary figure showing top markers for each cell type."""
    print("\n [Combined Summary] Creating marker summary figure...")
    
    coords = adata.obsm[basis]
    basis_name = basis.replace('X_', '').upper()
    
    # Determine grid
    n_cell_types = len(markers_dict)
    ncols = min(4, n_cell_types)
    nrows = int(np.ceil(n_cell_types / ncols))
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols, hspace=0.4, wspace=0.3)
    
    point_size = 120000 / len(adata)
    point_size = max(1, min(point_size, 20))
    
    for idx, (cell_type, markers) in enumerate(markers_dict.items()):
        row = idx // ncols
        col = idx % ncols
        
        # Get available markers
        available_markers = []
        mean_expressions = []
        
        for marker in markers[:top_n_markers * 2]:
            if marker in adata.var_names:
                if hasattr(adata.X, 'toarray'):
                    expr = adata[:, marker].X.toarray().flatten()
                else:
                    expr = adata[:, marker].X.flatten()
                
                if (expr > 0).any():
                    available_markers.append(marker)
                    mean_expressions.append(expr[expr > 0].mean())
        
        if not available_markers:
            continue
        
        # Sort by expression and take top N
        sorted_idx = np.argsort(mean_expressions)[::-1][:top_n_markers]
        top_markers = [available_markers[i] for i in sorted_idx]
        
        # Calculate combined expression score
        combined_expr = np.zeros(len(adata))
        for marker in top_markers:
            if hasattr(adata.X, 'toarray'):
                expr = adata[:, marker].X.toarray().flatten()
            else:
                expr = adata[:, marker].X.flatten()
            combined_expr += expr
        
        # Normalize
        if combined_expr.max() > 0:
            combined_expr = combined_expr / combined_expr.max()
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=combined_expr,
            cmap='YlOrRd',
            s=point_size,
            alpha=0.8,
            edgecolors='none',
            vmin=0,
            vmax=1,
            rasterized=True
        )
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.046)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Normalized\nExpression', fontsize=8, rotation=270, labelpad=15)
        
        marker_str = ', '.join(top_markers)
        ax.set_title(f'{cell_type}\n{marker_str}', fontsize=10, fontweight='bold')
        
        ax.set_xlabel(f'{basis_name}_1', fontsize=9)
        ax.set_ylabel(f'{basis_name}_2', fontsize=9)
        ax.tick_params(labelsize=8)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    fig.suptitle('Combined Marker Expression by Cell Type', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if output_dir is not None:
        save_figure(filename, output_dir, dpi=300)
    
    plt.close()
    
    return fig


def create_marker_validation_report(
    adata,
    markers_dict,
    cell_type_col='cell_type_final',
    output_dir=None
):
    """Create a detailed report of marker expression in assigned cell types."""
    print("\n [Marker Validation] Creating validation report...")
    
    validation_data = []
    
    for cell_type, markers in markers_dict.items():
        if cell_type not in adata.obs[cell_type_col].values:
            continue
        
        mask = adata.obs[cell_type_col] == cell_type
        n_cells = mask.sum()
        
        for marker in markers:
            if marker not in adata.var_names:
                continue
            
            # Get expression
            if hasattr(adata.X, 'toarray'):
                expr_all = adata[:, marker].X.toarray().flatten()
            else:
                expr_all = adata[:, marker].X.flatten()
            
            expr_in = expr_all[mask]
            expr_out = expr_all[~mask]
            
            # Calculate statistics
            pct_in = (expr_in > 0).sum() / len(expr_in) * 100 if len(expr_in) > 0 else 0
            pct_out = (expr_out > 0).sum() / len(expr_out) * 100 if len(expr_out) > 0 else 0
            
            mean_in = expr_in[expr_in > 0].mean() if (expr_in > 0).any() else 0
            mean_out = expr_out[expr_out > 0].mean() if (expr_out > 0).any() else 0
            
            fold_change = mean_in / (mean_out + 1e-10)
            specificity = pct_in / (pct_out + 1e-10)
            
            validation_data.append({
                'cell_type': cell_type,
                'marker': marker,
                'n_cells': n_cells,
                'pct_expressing_in': pct_in,
                'pct_expressing_out': pct_out,
                'mean_expr_in': mean_in,
                'mean_expr_out': mean_out,
                'fold_change': fold_change,
                'specificity_ratio': specificity
            })
    
    validation_df = pd.DataFrame(validation_data)
    
    if output_dir is not None:
        validation_df.to_csv(Path(output_dir) / "marker_validation_detailed.csv", index=False)
        print("Saved: marker_validation_detailed.csv")
        
        # Create summary
        summary = validation_df.groupby('cell_type').agg({
            'marker': 'count',
            'fold_change': 'mean',
            'specificity_ratio': 'mean',
            'pct_expressing_in': 'mean'
        }).round(2)
        summary.columns = ['n_markers', 'mean_fold_change', 'mean_specificity', 'mean_pct_expressing']
        summary.to_csv(Path(output_dir) / "marker_validation_summary.csv")
        print("Saved: marker_validation_summary.csv")
    
    return validation_df


def create_interactive_marker_report(adata, validation_df, marker_dict, output_dir):
    """Create an interactive HTML report for marker exploration."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Marker Expression Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
            th {{ background-color: #3498db; color: white; padding: 10px; text-align: left; }}
            td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .good {{ color: #27ae60; font-weight: bold; }}
            .warning {{ color: #f39c12; font-weight: bold; }}
            .bad {{ color: #e74c3c; font-weight: bold; }}
            .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Marker Expression Validation Report</h1>
        <div class="summary">
            <h3>Summary Statistics</h3>
            <p><strong>Total cells:</strong> {len(adata)}</p>
            <p><strong>Cell types identified:</strong> {adata.obs['cell_type_final'].nunique() if 'cell_type_final' in adata.obs else 'N/A'}</p>
            <p><strong>Markers evaluated:</strong> {len(validation_df)}</p>
        </div>
        
        <h2>Marker Performance by Cell Type</h2>
    """
    
    for cell_type in sorted(validation_df['cell_type'].unique()):
        ct_data = validation_df[validation_df['cell_type'] == cell_type]
        ct_data = ct_data.sort_values('fold_change', ascending=False)
        
        html_content += f"""
        <h3>{cell_type}</h3>
        <table>
            <tr>
                <th>Marker</th>
                <th>% Expressing (In)</th>
                <th>% Expressing (Out)</th>
                <th>Fold Change</th>
                <th>Specificity</th>
                <th>Quality</th>
            </tr>
        """
        
        for _, row in ct_data.iterrows():
            # Determine quality
            if row['fold_change'] >= 2 and row['pct_expressing_in'] >= 30:
                quality = '<span class="good">Excellent</span>'
            elif row['fold_change'] >= 1.5 or row['pct_expressing_in'] >= 20:
                quality = '<span class="warning">Good</span>'
            else:
                quality = '<span class="bad">Poor</span>'
            
            html_content += f"""
            <tr>
                <td><strong>{row['marker']}</strong></td>
                <td>{row['pct_expressing_in']:.1f}%</td>
                <td>{row['pct_expressing_out']:.1f}%</td>
                <td>{row['fold_change']:.2f}x</td>
                <td>{row['specificity_ratio']:.2f}</td>
                <td>{quality}</td>
            </tr>
            """
        
        html_content += "</table>"
    
    html_content += f"""
    </body>
    </html>
    """
    
    with open(Path(output_dir) / "marker_report.html", 'w') as f:
        f.write(html_content)
    
    print("Saved: marker_report.html (open in browser)")
    
    return None
    
# =========================================================================
# PROCESS
# =========================================================================
#metadata_path = "metadata_final.csv"
#batch_dir = "batch_cores"
#process_dir = "process"
#output_folder = "celltype_cores"
#os.makedirs(output_folder, exist_ok=True)

#core_IDs = [str(core_ID) for core_ID in list(metadata.drop_duplicates(subset='ST_ID')['ST_ID']) if
 #           (core_ID + '.zarr') in os.listdir(batch_dir)]
#core_sdata = {v: sd.read_zarr(batch_dir + '/' + v + '.zarr') for v in core_IDs}

#for core_id in core_IDs:
    #os.makedirs(Path(output_folder) / core_id, exist_ok=True)
    #output_dir = Path(output_folder) / core_id

core_id = "BC_prime_ROI"

output_folder = "celltype_output"
output_dir = Path(output_folder) / core_id
os.makedirs(output_dir, exist_ok=True)

# =========================================================================
# 1. LOAD ZARR DATA
# =========================================================================
print("\n[1] Loading Zarr data")
sdata = sd.read_zarr(ZARR_DIR)
#adata = sdata.tables['table_refined']

adata = sdata.tables['table']

print(f"Cells (Whole Slide): {adata.n_obs}")
print(f"Genes: {adata.n_vars}")

# =========================================================================
# 2. CELL TYPING WITH CELLTYPIST
# =========================================================================

print("\n[2] Annotating cell types with CellTypist")

# Load model
try:
    model = celltypist.models.Model.load(f"{MODEL_NAME}.pkl")
except FileNotFoundError:
    celltypist.models.download_models(model=MODEL_NAME)
    model = celltypist.models.Model.load(f"{MODEL_NAME}.pkl")

# --- DIAGNOSTICS ---
print(f"DEBUG: Model expects features like: {model.features[:5]}")
print(f"DEBUG: Data currently has: {adata.var_names[:5].tolist()}")

# --- GENE MAPPING (SYMBOLS -> IDS) ---
print("Mapping Gene Symbols to Ensembl IDs...")

# Load and clean the CSV
xenium_table = pd.read_csv(TenX, sep=None, engine='python')
xenium_table['gene_id'] = xenium_table['gene_id'].astype(str).str.strip()
xenium_table['gene_name'] = xenium_table['gene_name'].astype(str).str.strip()

# --- CRITICAL CHANGE: REVERSE THE MAPPING ---
# Key = Gene Symbol (A2ML1), Value = Ensembl ID (ENSG...)
sym_to_ens = dict(zip(xenium_table['gene_name'], xenium_table['gene_id']))

# Prepare the AnnData names (Clean existing symbols)
current_names_series = adata.var_names.to_series().astype(str).str.strip()

# Map: Symbols -> IDs
mapped_series = current_names_series.map(sym_to_ens)
final_names = mapped_series.fillna(current_names_series)

# Update AnnData
adata.var_names = final_names.astype(str)
adata.var_names_make_unique()

# Check overlap
model_genes = model.features
data_genes = adata.var_names
overlap = len(set(model_genes).intersection(set(data_genes)))
print(f"Gene overlap with model: {overlap} genes")

if overlap == 0:
    print(f"DEBUG: Data genes (first 5): {adata.var_names[:5].tolist()}")
    raise ValueError("Overlap is 0. Mapping failed to produce Ensembl IDs.")

# 4. PREPARE DATA
print("Normalizing data (Log1p)...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 5. RUN ANNOTATION
print("Running CellTypist...")
result = annotate(adata, model=model, majority_voting=True)
adata.obs["cell_type"] = result.predicted_labels["majority_voting"]
adata.obs["conf_score"] = result.probability_matrix.max(axis=1)

# Run UMAP
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)

# Save
adata.write(Path(output_dir) / f"{core_id}_celltyped_trained_model.h5ad")

# Plot
plot_celltype_spatial(adata, 'cell_type', title=core_id)
save_figure(f'{core_id}_celltyping.png', output_dir, dpi=600)

# restore gene symbols for plotting
print("Restoring gene symbols for downstream analysis")

# create reverse mapping
ens_to_sym = {v: k for k, v in sym_to_ens.items()}

# map index back to symbols and convert to series for mapping
current_ids = adata.var_names.to_series()
restored_names = current_ids.map(ens_to_sym)

# fill any that did not match
final_symbols = restored_names.fillna(current_ids)

# update adata
adata.var_names = final_symbols.astype(str)
adata.var_names_make_unique()
print(f"Restored {len(adata.var_names)} gene symbols.")

print("Cell typing complete")

# =========================================================================
# 3. CANONICAL MARKER ANALYSIS
# =========================================================================
print("\n[3] Analyzing canonical markers")

# Load biomarker data
markers = pd.read_csv(biomarkers_csv)
markers[markers['Cell_Type'] != 'Adipocytes']
markers = markers[markers['Recommendation'] == 'USE THIS GENE']

biomarker_df = pd.read_csv(biomarkers_matrix_csv, index_col=0)
if 'Adipocyte' in biomarker_df.columns:
    biomarker_df = biomarker_df.drop('Adipocyte', axis=1)

# Ensure we only look for genes that exist in our markers list
valid_genes = [g for g in markers['Gene_Symbol_Resolved'] if g in biomarker_df.index]
biomarker_df = biomarker_df.loc[valid_genes]

# Clean biomarker dataframe (drop columns not in valid set if needed)
cols_to_drop = ['Total_Cell_Types', 'Dendritic cell']
biomarker_df = biomarker_df.drop(
    [col for col in cols_to_drop if col in biomarker_df.columns],
    axis=1
)

# Create marker dictionary
print("Creating marker dictionary:")
marker_dict, vis_cell_types, excl_cell_types = create_marker_dict(
    biomarker_df, get_custom_palette(extended=True)
)

# --- CRITICAL FIX START ---
# Use the annotated data we just created, NOT the raw table from disk
sdata_table = adata.copy()

# Create 'majority_voting' column because the plotting functions expect it
# (In Step 2 we saved it as 'cell_type')
sdata_table.obs['majority_voting'] = sdata_table.obs['cell_type']
# --------------------------

# Prepare genes for visualization
genes_present = [x for x in markers['Gene_Symbol_Resolved'] 
                 if x in sdata_table.var_names]
markers_for_core = markers[
    markers['Gene_Symbol_Resolved'].isin(genes_present)
]

print(f"Plotting {len(genes_present)} marker genes...")

# Matrix visualizations
# Now this works because 'majority_voting' exists in sdata_table
plot_matrix_visualizations(sdata_table, markers_for_core, "majority_voting", output_dir)

# Cell type distribution
plot_celltype_distribution(sdata_table, 'majority_voting', output_dir, palette=get_custom_palette(extended=True))

# =========================================================================
# 4. CONFIDENCE STATISTICS
# =========================================================================
print("\n[4] Computing confidence statistics")
save_confidence_statistics(
    sdata_table,
    Path(output_dir) / "mean_conf_by_celltype.csv",
)

# =========================================================================
# 5. BIOMARKER ANNOTATION
# =========================================================================
print("\n[5] Annotating cells with biomarker scores")
adata_biomarker = sdata_table.copy()
adata_biomarker = annotate_cells_with_marker_scores(adata_biomarker, marker_dict)
marker_dict, complete_cell_types, missing_markers = create_marker_dict_with_validation(
  biomarker_df, 
  sdata_table,
  get_custom_palette(extended=True))

# =========================================================================
# 6. COMPARE AND REASSIGN CELL TYPES
# =========================================================================
print("\n[6] compare and reassign cell types")
# Calculate optimal thresholds
if 'conf_score' in sdata_table.obs.columns:
    celltypist_threshold = calculate_optimal_threshold(
        sdata_table.obs['conf_score'], 
        percentile=50
    )
else:
    celltypist_threshold = 0.6

# Calculate biomarker threshold
biomarker_cols = [col for col in adata_biomarker.obs.columns if col.endswith('_score')]
if biomarker_cols:
    all_biomarker_scores = adata_biomarker.obs[biomarker_cols].max(axis=1)
    biomarker_threshold = calculate_optimal_threshold(
        all_biomarker_scores,
        percentile=50
    )
else:
    biomarker_threshold = 0.7

# Integrate annotations
adata_integrated = integrate_celltypist_and_biomarker_annotations(
    adata=sdata_table,
    biomarker_adata=adata_biomarker,
    celltypist_conf_threshold=celltypist_threshold,
    biomarker_conf_threshold=biomarker_threshold,
    use_neighborhood_refinement=True,
    n_neighbors=15,
    consensus_threshold=0.6
)

# Visualizations
plot_annotation_comparison(adata_integrated, output_dir)
visualize_confidence_distribution(adata_integrated, output_dir)
plot_refinement_summary(adata_integrated, output_dir)

# =========================================================================
# 7. OPTIONAL TARGETED REFINEMENT
# =========================================================================      
print("\n[7] target refinement")
# Make sure final cell type column is set
adata_refined = adata_integrated.copy()
adata_refined.obs['cell_type'] = adata_refined.obs['cell_type_final'].copy()

# =========================================================================
# 8. VALIDATION
# =========================================================================
def validate_annotations(adata, output_dir, marker_dict=None):
    """Comprehensive validation of cell type annotations."""
    print("\n [Validation] Performing annotation quality checks...")
    
    validation_results = {}
    
    # ===================================================================
    # Convert all relevant columns to string at the start
    # ===================================================================
    adata_work = adata.copy()
    
    # Convert categorical columns to string
    categorical_cols = ['majority_voting', 'cell_type_final', 'cell_type', 
                        'annotation_source', 'predicted_labels']
    
    for col in categorical_cols:
        if col in adata_work.obs.columns:
            if pd.api.types.is_categorical_dtype(adata_work.obs[col]):
                adata_work.obs[col] = adata_work.obs[col].astype(str)
    
    # 1. Neighborhood purity (Silhouette score)
    if 'X_umap' in adata_work.obsm:
        # Determine which cell type column to use
        cell_type_col = None
        if 'cell_type_final' in adata_work.obs.columns:
            cell_type_col = 'cell_type_final'
        elif 'cell_type' in adata_work.obs.columns:
            cell_type_col = 'cell_type'
        elif 'majority_voting' in adata_work.obs.columns:
            cell_type_col = 'majority_voting'
        
        if cell_type_col is not None:
            try:
                # Create categorical codes for silhouette score
                cell_types_cat = pd.Categorical(adata_work.obs[cell_type_col])
                labels = cell_types_cat.codes
                
                if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                    sil_score = silhouette_score(adata_work.obsm['X_umap'], labels)
                    # FIX: Convert numpy float to python float
                    validation_results['silhouette_score'] = float(sil_score)
                    print(f"Silhouette score (UMAP space): {sil_score:.3f}")
                    print("Higher is better, range: -1 to 1)")
                else:
                    print("Cannot calculate silhouette score: only one cell type")
            except Exception as e:
                print(f"Could not calculate silhouette score: {e}")
    
    # 2. Compare annotations (confusion matrix)
    if 'majority_voting' in adata_work.obs and 'cell_type_final' in adata_work.obs:
        try:
            # Both columns are already converted to string above
            confusion_df = pd.crosstab(
                adata_work.obs['majority_voting'],
                adata_work.obs['cell_type_final'],
                normalize='index'
            )
            
            # Save confusion matrix
            confusion_df.to_csv(output_dir / "annotation_confusion_matrix.csv")
            print("Saved: annotation_confusion_matrix.csv")
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # Handle large confusion matrices
            if confusion_df.shape[0] > 30 or confusion_df.shape[1] > 30:
                sns.heatmap(confusion_df, fmt='.2f', cmap='Blues', 
                            ax=ax, cbar_kws={'label': 'Proportion'},
                            xticklabels=True, yticklabels=True)
            else:
                sns.heatmap(confusion_df, annot=True, fmt='.2f', cmap='Blues', 
                            ax=ax, cbar_kws={'label': 'Proportion'})
            
            ax.set_title('CellTypist vs Final Annotations\n(row-normalized)', fontsize=14)
            ax.set_xlabel('Final Annotation', fontsize=12)
            ax.set_ylabel('CellTypist Annotation', fontsize=12)
            
            if confusion_df.shape[0] > 10:
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
            if confusion_df.shape[1] > 10:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
            plt.tight_layout()
            save_figure("annotation_confusion.png", output_dir, dpi=300)
            plt.close()
            
            # Calculate agreement
            agreement = (adata_work.obs['majority_voting'] == adata_work.obs['cell_type_final']).mean()
            # FIX: Convert numpy float to python float
            validation_results['annotation_agreement'] = float(agreement)
            print(f"Annotation agreement: {agreement:.1%}")
            
        except Exception as e:
            print(f"Could not create confusion matrix: {e}")
    
    # 3. Marker expression validation
    if marker_dict is not None:
        cell_type_col = None
        if 'cell_type_final' in adata_work.obs.columns:
            cell_type_col = 'cell_type_final'
        elif 'cell_type' in adata_work.obs.columns:
            cell_type_col = 'cell_type'
        
        if cell_type_col is not None:
            print("\n Marker expression validation:")
            marker_validation = {}
            
            for cell_type, markers in marker_dict.items():
                if cell_type not in adata_work.obs[cell_type_col].values:
                    continue
                
                mask = adata_work.obs[cell_type_col] == cell_type
                n_cells = mask.sum()
                
                if n_cells == 0:
                    continue
                
                available_markers = [m for m in markers if m in adata_work.var_names]
                
                if available_markers:
                    try:
                        # Calculate mean expression
                        if hasattr(adata_work.X, 'toarray'):
                            in_celltype = adata_work[mask, available_markers].X.toarray().mean()
                            out_celltype = adata_work[~mask, available_markers].X.toarray().mean()
                        else:
                            in_celltype = adata_work[mask, available_markers].X.mean()
                            out_celltype = adata_work[~mask, available_markers].X.mean()
                        
                        fold_change = in_celltype / (out_celltype + 1e-10)
                        
                        marker_validation[cell_type] = {
                            'n_cells': int(n_cells),
                            'n_markers': len(available_markers),
                            'mean_expression_in': float(in_celltype),
                            'mean_expression_out': float(out_celltype),
                            'fold_change': float(fold_change)
                        }
                        print(f"{cell_type}: FC={fold_change:.2f} ({len(available_markers)} markers, {n_cells} cells)")
                    
                    except Exception as e:
                        print(f"Warning: Could not validate {cell_type}: {e}")
            
            # Save marker validation
            if marker_validation:
                marker_val_df = pd.DataFrame(marker_validation).T
                marker_val_df.to_csv(Path(output_dir) / "marker_expression_validation.csv")
                print("\n Saved: marker_expression_validation.csv")
    
    # 4. Cell type size distribution
    cell_type_col = None
    if 'cell_type_final' in adata_work.obs.columns:
        cell_type_col = 'cell_type_final'
    elif 'cell_type' in adata_work.obs.columns:
        cell_type_col = 'cell_type'
    elif 'majority_voting' in adata_work.obs.columns:
        cell_type_col = 'majority_voting'
    
    if cell_type_col is not None:
        celltype_counts = adata_work.obs[cell_type_col].value_counts()
        print("\n Cell type distribution:")
        for ct, count in celltype_counts.items():
            print(f"{ct}: {count} ({count/len(adata_work)*100:.1f}%)")
        
        celltype_counts.to_csv(Path(output_dir) / "celltype_distribution.csv")
        print("Saved: celltype_distribution.csv")
    
    # Save validation results
    if validation_results:
        import json
        # FIX: Define a custom encoder to handle any remaining numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                    np.int16, np.int32, np.int64, np.uint8,
                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, 
                    np.float64)):
                    return float(obj)
                return json.JSONEncoder.default(self, obj)

        with open(Path(output_dir) / "validation_metrics.json", 'w') as f:
            json.dump(validation_results, f, indent=2, cls=NumpyEncoder)
        print("\n Saved: validation_metrics.json")
    
    return validation_results

# =========================================================================
# 9. MARKER EXPRESSION VISUALISATION
# =========================================================================
print("\n[9] MARKER EXPRESSION VISUALIZATION")

# Create output subdirectory
marker_output_dir = Path(output_dir) / "marker_expression_plots"
marker_output_dir.mkdir(exist_ok=True)

# Individual marker overlays
print("\n [9A] Creating individual marker expression overlays...")
plot_marker_expression_overlay(
    adata=adata_refined,
    markers_dict=marker_dict,
    basis='X_umap',
    ncols=3,
    cmap='YlOrRd',
    output_dir=marker_output_dir,
    filename_prefix="final"
)

# Combined summary
print("\n [9B] Creating combined marker summary...")
create_combined_marker_summary(
    adata=adata_refined,
    markers_dict=marker_dict,
    basis='X_umap',
    top_n_markers=3,
    figsize=(20, 12),
    output_dir=output_dir,
    filename="marker_summary_combined.png"
)

# Marker validation report
print("\n [9C] Creating marker validation report...")
validation_df = create_marker_validation_report(
    adata=adata_refined,
    markers_dict=marker_dict,
    cell_type_col='cell_type',
    output_dir=output_dir
)

# Interactive HTML report
print("\n [9D] Creating interactive marker report...")
create_interactive_marker_report(
    adata=adata_refined,
    validation_df=validation_df,
    marker_dict=marker_dict,
    output_dir=output_dir
)

# =========================================================================
# 10. SAVE REFINED DATA
# =========================================================================
print("\n[10] Saving refined annotations")
adata_integrated.write_h5ad(Path(output_dir) / "integrated_annotations.h5ad")
adata_refined.write_h5ad(Path(output_dir) / "refined_annotations.h5ad")

annotation_summary = pd.DataFrame({
    'cell_id': adata_refined.obs.index,
    'celltypist_original': sdata_table.obs['majority_voting'],
    'celltypist_confidence': sdata_table.obs.get('conf_score', 0.75),
    'biomarker_confidence': adata_refined.obs.get('biomarker_confidence', 0.0),
    'annotation_source': adata_refined.obs.get('annotation_source', 'celltypist'),
    'final_cell_type': adata_refined.obs['cell_type']
})
annotation_summary.to_csv(Path(output_dir) / "annotation_summary.csv", index=False)

print(f"Saved all outputs to: {output_dir}")

# =========================================================================
# 11. SPATIAL VISUALISATIONS
# =========================================================================
print("\n[11] Creating spatial visualizations")

# Original annotations
fig = plot_celltype_spatial(sdata_table, 'majority_voting')
save_figure("spatial_original.png", output_dir)

# Biomarker-based annotations
fig = plot_celltype_spatial(adata_biomarker, 'cell_type')
save_figure("spatial_biomarker.png", output_dir)

# Refined annotations
fig = plot_celltype_spatial(adata_refined, 'cell_type')
save_figure("spatial_refined.png", output_dir)

# =========================================================================
# 12. SAVE REFINED DATA
# =========================================================================
print("\n[12] Saving refined annotations")
adata_refined.write_h5ad(Path(output_dir) / "refined_annotations.h5ad")
print(f"Saved: {Path(output_dir) / 'refined_annotations.h5ad'}")

# =========================================================================
# 13. CORRELATION MATRICES
# =========================================================================
print("\n[13] Creating correlation matrices")

if len(np.unique(sdata_table.obs['majority_voting'])) > 1:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sc.pl.correlation_matrix(sdata_table, "majority_voting", show=False)
    save_figure("correlations.png", output_dir)

if len(np.unique(adata_refined.obs['cell_type'])) > 1:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sc.pl.correlation_matrix(adata_refined, "cell_type", show=False)
    save_figure("correlations_adata_refined.png", output_dir)

# =========================================================================
# 14. REFINED VISUALISATIONS
# =========================================================================
print("\n[14] Creating visualizations for refined annotations")

# Matrix visualizations
plot_matrix_visualizations(adata_refined, markers_for_core, "cell_type", output_dir)

# Cell type distribution - USE CONSISTENT PALETTE
plot_celltype_distribution(adata_refined, 'cell_type', output_dir, suffix="_refined", palette=get_custom_palette(extended=True))

print("PIPELINE COMPLETE")
print(f"All outputs saved to: {output_dir}")