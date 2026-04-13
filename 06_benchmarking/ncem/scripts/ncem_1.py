#### NCEM ####
# node-centric expression models (GNN)
# graph - radius-based (200um)

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0" # disable XLA

import logging
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import json
import traceback
from shapely.geometry import Point, Polygon as ShapelyPolygon
import ncem
from ncem.estimators import EstimatorGraph
from ncem.interpretation import InterpreterGraph
from ncem.estimators import EstimatorInteractions
from ncem.data import customLoader
import scipy.sparse as sp
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(f"DEBUG: Found {len(gpus)} GPUs") 
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("SUCCESS: GPU Memory Growth Enabled.")
    except RuntimeError as e:
        print(f"GPU Error: {e}")
else:
    print("No GPUs found. Slow training on CPU.")

mpl.rcParams['savefig.facecolor'] = 'white'

# set paths
path = "/scratch/users/k22026807/masters/project/benchmarking/ncem/"
os.chdir(path)
os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

 
#### Colour palette ####

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
 
#========================
#     1. LOAD DATA 
#========================

print("Loading data...") # backed mode to save RAM
adata_full = ad.read_h5ad("/scratch/users/k22026807/masters/project/celltyping/celltype_output/BC_prime/refined_annotations.h5ad", backed='r')

# ROI selection
with open('/scratch/users/k22026807/masters/project/alignment/region1_xenium.geojson') as f:
    roi = json.load(f)
roi_coords = roi['features'][0]['geometry']['coordinates'][0]
polygon = ShapelyPolygon(roi_coords)

print("Filtering ROI...")
spatial_coords = adata_full.obsm['spatial']
roi_mask = np.array([polygon.contains(Point(x, y)) for x, y in spatial_coords])

# create the actual in-memory object with ONLY the ROI cells
adata = adata_full[roi_mask].to_memory()
adata.obs['cell_type'] = adata.obs['cell_type'].cat.remove_unused_categories()
print(f"ROI cells loaded into memory: {adata.shape[0]}")

adata.var_names_make_unique()

#========================
#  2. PREPROCESSING
#========================

print("Preprocessing...")
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', span=1.0)

# keep only HVGs as NCEM is computationally intensive
adata = adata[:, adata.var['highly_variable']].copy()
print(f"HVGs retained: {adata.shape[1]}")


# SUBSAMPLE - REMOVE AFTER CHECK 
#sc.pp.subsample(adata, n_obs=5000, random_state=42)
#print(f"Subsampled to: {adata.shape[0]} cells.")

sc.pp.scale(adata, max_value=10)

# make cell type labels as categorical in obs for NCEM req
adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
cell_types = list(adata.obs['cell_type'].cat.categories)
print(f"Cell types: {cell_types}")

#=========================
# 3. BUILD SPATIAL GRAPH 
#=========================

# dummy column for metabric and tma id
adata.obs['sample_dummy'] = 'sample1'
# dummy entry to satisfy the NCEM internal cleanup
adata.uns['spatial'] = {}

print("Building spatial graph...")

if not sp.issparse(adata.X):
    adata.X = sp.csr_matrix(adata.X)

graph_adata = customLoader(
    adata=adata,
    patient='sample_dummy',
    library_id='sample_dummy',
    radius=200,
    cluster='cell_type')

# extract nodes and edges
n_nodes = graph_adata.adata.n_obs
n_edges = graph_adata.adata.obsp['distances'].nnz
mean_degree = n_edges / n_nodes

print(f"Graph nodes: {n_nodes}")
print(f"Graph edges: {n_edges}")
print(f"Mean degree: {mean_degree:.2f}")

# QC - degree distribution
degrees = np.array(graph_adata.adata.obsp['connectivities'].sum(axis=1)).flatten()

fig, ax = plt.subplots(figsize=(7,5), facecolor='white')
ax.hist(degrees, bins=40, color='steelblue', edgecolor='white')
ax.set_xlabel('Node degree (neighbors within 200um)')
ax.set_ylabel('Number of cells')
ax.set_title('Spatial Graph Degree Distribution', fontsize=12, fontweight='bold')
ax.axvline(degrees.mean(), color='firebrick', linestyle='--', label=f'Mean = {degrees.mean():.1f}')
ax.legend()
plt.tight_layout()
plt.savefig('figures/graph_degree_distribution.png', dpi=300)
plt.close()

# QC - spatial graph edges overlaid on tissue
pos = adata.obsm['spatial']
# edge list from the sparse matrix
rows, cols = graph_adata.adata.obsp['distances'].nonzero()
edge_indices = np.random.choice(len(rows), size=min(5000, len(rows)), replace=False)

fig, ax = plt.subplots(figsize=(10,9), facecolor='white')
for idx in edge_indices:
    i, j = rows[idx], cols[idx]
    ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
        c='lightgrey', lw=0.2, alpha=0.4, rasterized=True)

for ct, col in custom_palette.items():
    mask = adata.obs['cell_type'] == ct
    if mask.sum() > 0:
        ax.scatter(pos[mask, 0], pos[mask, 1], s=1, c=col,
                   label=ct, alpha=0.8, rasterized=True)
ax.set_title('Spatial Graph (200 um radius)', fontsize=12, fontweight='bold')
ax.set_xlabel('X (um)'); ax.set_ylabel('Y (um)')
ax.set_aspect('equal')
ax.legend(markerscale=4, fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/spatial_graph_overlay.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()

#========================================
#   HELPER - custom data initialisation
#========================================

# no native get_data() function

adata.obs['patient'] = 'sample1' 
adata.obs['batch'] = 'batch1' 

def bind_custom_ncem_data(adata,
    graph_loader, img_key='sample1'):

    """
    NCEM EstimatorGraph lacks a native get_data() function for custom datasets.
    This helper bridges AnnData shapes to TensorFlow dictionaries.
    """

    est = EstimatorGraph()
    est.data = graph_loader
    
    # image / patient keys
    est.img_keys = [img_key]
    est.complete_img_keys = [img_key]
    #est.img_keys_all = [img_key]
    est.img_keys_eval = []
    est.img_keys_test = []
    est.img_keys_train = [img_key]
    est.img_to_patient_dict = {img_key: 'sample1'}
    est.undefined_node_types = None
    est.domains = {img_key: 0}

    # expression data (h_0)
    dense_X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    est.h_0 = {img_key: np.array(dense_X, dtype=np.float32)}
    
    # spatial graph (a / adj)
    adj = sp.csr_matrix(adata.obsp['connectivities'].copy())
    adj.setdiag(1.0) # Add self-loops for GCN
    
    est.a = {img_key: adj}
    est.adj = {img_key: adj}

    # cell type labels (h_1 / node_labels)
    dummy_df = pd.get_dummies(adata.obs['cell_type'])
    est.h_1 = {img_key: dummy_df.values.astype(np.float32)}
    est.node_labels = {img_key: dummy_df.values.astype(np.float32)}

    # covariates (Filled with Zeros/Empty)
    est.node_covariates = {img_key: np.zeros((adata.shape[0], 0), dtype=np.float32)}
    est.node_covar = {img_key: np.zeros((adata.shape[0], 0), dtype=np.float32)}
    est.graph_covar = {img_key: np.zeros((1, 0), dtype=np.float32)}
    est.proportions = {img_key: np.zeros((adata.shape[0], 0), dtype=np.float32)}

    # utilities
    est.nodes_idx = {img_key: np.arange(adata.shape[0])}
    est.size_factors = {img_key: np.ones(adata.shape[0], dtype=np.float32)}
    est.sf = {img_key: np.ones(adata.shape[0], dtype=np.float32)}
    
    # names/metadata
    est.node_label_names = list(dummy_df.columns)
    est.node_feature_names = list(adata.var_names)
    est.node_type_names = list(dummy_df.columns)
    est.graph_covar_names = []
    
    # Essential Shape Registrations (The Integers)
    est.n_features_0 = adata.shape[1]       
    est.n_features_1 = dummy_df.shape[1]    
    est.n_node_labels = dummy_df.shape[1]   
    est.n_domains = 1                       
    est.n_node_features = adata.shape[1]    
    est.n_eval_nodes_per_graph = adata.shape[0]
    est.max_nodes = adata.shape[0]
    est.n_eval_nodes_per_graph = min(500, adata.shape[0])  # limit eval nodes
    
    est.vi_model = False

    # Missing Integer Fixes
    est.n_node_covariates = 0
    est.n_graph_covariates = 0
    est.n_features_standard = adata.shape[1]
    est.n_features_type = dummy_df.shape[1]
    
    return est

#===========================
# 4. NCEM MODEL DEFINITION
#===========================

print("Initialising NCEM GNN estimator...")

estimator = bind_custom_ncem_data(adata, graph_adata)

print("Splitting data...")
estimator.split_data_node(
    validation_split=0.1,
    test_split=0.1,
    seed=42)

print("Building model architecture...")


try:
    estimator.init_model(
        #n_intermediate=128,
        #depth_graph=1,
        #use_interactions=False,
        #dropout_rate=0.1,
        l2_coef=1e-4,
        learning_rate=1e-4,
        optimizer='adam'

    )
    print("SUCCESS: Model architecture built!")
except Exception as e:
    print("CRITICAL ERROR DURING init_model:")
    traceback.print_exc()
    raise RuntimeError("Stopping execution to view traceback.")

# safety check before it hits the train() function
#if getattr(estimator, 'model', None) is None:
#    raise RuntimeError("NCEM failed to build the model.")

try:
    print(estimator.model.training_model.summary())
except AttributeError:
    print("Model built successfully, proceeding to training...")

#===========================
#       5. TRAINING
#===========================

print("Training NCEM GNN...")

keras_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        patience=10, 
        min_lr=1e-5
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='models/ncem_gnn.weights.h5', 
        save_best_only=True, 
        save_weights_only=True
    )
]

estimator.train(
    epochs=200,
    batch_size=1024,
    callbacks=keras_callbacks
    )

# training curve
history = estimator.train_history
fig, axes = plt.subplots(1,2,figsize=(12,4), facecolor='white')
for ax, metric, title in zip(
    axes,
    ['loss', 'val_loss'],
    ['Training Loss', 'Validation Loss']
):
    if metric in history:
        ax.plot(history[metric], color='steelblue')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/training_curves.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()


#===========================
#      6. EVALUATION
#===========================

# reconstruction R^2 per cell type 
# tells us how well each cell types'
# expression can be predicted from its niche
# high R^2 = strong niche dependence

print("Evaluating model...")

eval_results = estimator.evaluate_any(
    #img_keys=None,
    #node_idx=None,
    batch_size=1024)

# R^2 per cell type
r2_df = pd.DataFrame({
    'cell_type': list(eval_results['r2_per_type'].keys()),
    'R2':        list(eval_results['r2_per_type'].values()),
}).sort_values('R2', ascending=False)
r2_df.to_csv('results/r2_per_celltype.csv', index=False)
print("R² per cell type:\n", r2_df.to_string(index=False))
 
fig, ax = plt.subplots(figsize=(8, max(4, len(r2_df) * 0.45)), facecolor='white')
colors = [custom_palette.get(ct, 'grey') for ct in r2_df['cell_type']]
ax.barh(r2_df['cell_type'], r2_df['R2'], color=colors)
ax.set_xlabel('Reconstruction R²')
ax.set_title('NCEM GNN: Niche Predictability per Cell Type',
             fontsize=12, fontweight='bold')
ax.axvline(0, color='black', lw=0.8)
ax.set_xlim(left=min(0, r2_df['R2'].min() - 0.02))
plt.tight_layout()
plt.savefig('figures/r2_per_celltype.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()

#===========================
#  7. CCC INTERPRETATION
#===========================

# InterpreterGraph decomposes how much each sender cell
# type drives expression changes in each target (receiver)
# cell type.
# This produces the sender -> receiver coupling matrix

print("Running CCC interpretation...")

interpreter = InterpreterGraph(estimator=estimator)

# Compute sender-type coupling coefficients
# coupling[target_type][sender_type] = mean absolute effect on target expression
# dynamically setting n_eval to avoid errors
min_cluster_size = adata.obs['cell_type'].value_counts().min()
safe_n_eval = min(500, min_cluster_size)
print(f"Using n_eval={safe_n_eval} based on smallest cluster.")

interpreter.compute_type_coupling(
    n_eval=safe_n_eval,        
    seed=42)
 
coupling_df = interpreter.type_coupling
coupling_df.to_csv('results/ncem_type_coupling.csv')
print("Type coupling matrix:\n", coupling_df.to_string())
 
# ── Plot 1: Coupling heatmap (sender → receiver) ──
fig, ax = plt.subplots(
    figsize=(max(8, len(coupling_df.columns) * 0.7),
             max(6, len(coupling_df.index)  * 0.6)),
    facecolor='white'
)
im = ax.imshow(coupling_df.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(coupling_df.columns)))
ax.set_xticklabels(coupling_df.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(coupling_df.index)))
ax.set_yticklabels(coupling_df.index, fontsize=8)
plt.colorbar(im, ax=ax, label='Coupling coefficient\n(sender → receiver influence)')
ax.set_xlabel('Sender cell type')
ax.set_ylabel('Receiver (target) cell type')
ax.set_title('NCEM GNN: Cell-Type Communication Coupling',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/ncem_coupling_heatmap.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
 
# ── Plot 2: Per-target-type sender ranking  ──
n_targets = len(coupling_df.index)
ncols = 3
nrows = int(np.ceil(n_targets / ncols))
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(ncols * 5, nrows * 3.5),
                          facecolor='white')
axes = axes.flatten()
for i, target in enumerate(coupling_df.index):
    ax = axes[i]
    row = coupling_df.loc[target].sort_values(ascending=False)
    bar_colors = [custom_palette.get(s, 'grey') for s in row.index]
    ax.bar(range(len(row)), row.values, color=bar_colors)
    ax.set_xticks(range(len(row)))
    ax.set_xticklabels(row.index, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Coupling coefficient')
    ax.set_title(f'→ {target}', fontsize=9, fontweight='bold')
    ax.set_facecolor('white')
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Sender Influence per Target Cell Type (NCEM GNN)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/ncem_sender_ranking_per_target.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
 
# ── Plot 3: Total outgoing influence per sender type ──
total_sent = coupling_df.sum(axis=0).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
bar_colors = [custom_palette.get(ct, 'grey') for ct in total_sent.index]
ax.bar(range(len(total_sent)), total_sent.values, color=bar_colors)
ax.set_xticks(range(len(total_sent)))
ax.set_xticklabels(total_sent.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Total coupling (sum over all receivers)')
ax.set_title('NCEM GNN: Total Outgoing Signal per Cell Type',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/ncem_total_outgoing_signal.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
 
# ── Plot 4: Total incoming influence per receiver type ──
total_recv = coupling_df.sum(axis=1).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
bar_colors = [custom_palette.get(ct, 'grey') for ct in total_recv.index]
ax.bar(range(len(total_recv)), total_recv.values, color=bar_colors)
ax.set_xticks(range(len(total_recv)))
ax.set_xticklabels(total_recv.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Total coupling (sum over all senders)')
ax.set_title('NCEM GNN: Total Incoming Signal per Cell Type',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/ncem_total_incoming_signal.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
 

#================================
#  8. GENE-LEVEL INTERPRETATION
#================================

# which genes in target type are most strongly driven
# by neighbour signals?
# produces a per-target-type list of niche-responsive
# genes for downstream enrichment

print("Computing gene-level niche effects...")

interpreter.compute_gene_effects(
    sender_keys=['cell_type'],
    n_eval=safe_n_eval,
    seed=42,)

# Save per-target-type gene effect tables
gene_effect_summary = {}
for target_type in cell_types:
    try:
        gene_effects = interpreter.get_gene_effects(target_type=target_type)
        # gene_effects: DataFrame with genes as index, senders as columns
        gene_effects.to_csv(
            f'results/gene_effects_{target_type.replace(" ", "_")}.csv'
        )
        # Top niche-responsive genes = highest mean effect across all senders
        gene_effects['mean_effect'] = gene_effects.abs().mean(axis=1)
        top_genes = gene_effects['mean_effect'].sort_values(ascending=False).head(20)
        gene_effect_summary[target_type] = top_genes
    except Exception as e:
        print(f"Gene effects failed for {target_type}: {e}")
 
# ── Plot: Top niche-responsive genes per target type ──
for target_type, top_genes in gene_effect_summary.items():
    if len(top_genes) == 0:
        continue
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    ax.barh(top_genes.index[::-1], top_genes.values[::-1], color='steelblue')
    ax.set_xlabel('Mean absolute niche effect')
    ax.set_title(f'Top Niche-Responsive Genes\nTarget: {target_type}',
                 fontsize=11, fontweight='bold')
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(
        f'figures/gene_effects_{target_type.replace(" ", "_")}.png',
        dpi=300, bbox_inches='tight', facecolor='white'
    )
    plt.close()
 
# Combined table: top 10 genes per target type
summary_rows = []
for target_type, top_genes in gene_effect_summary.items():
    for gene, effect in top_genes.head(10).items():
        summary_rows.append({
            'target_type': target_type,
            'gene':        gene,
            'mean_effect': effect,
        })
pd.DataFrame(summary_rows).to_csv('results/top_niche_genes_per_celltype.csv', index=False)


#================================
#  9. SPATIAL COUPLING MAPS
#================================

# project per-cell coupling scores onto
# tissue coordinates to show where CCC is
# strongest in physical space

print("Computing spatial coupling maps...")
 
try:
    # Get per-cell predicted vs actual reconstruction error
    # High residual = cell expression poorly explained by niche → low local CCC
    # Low residual  = cell expression strongly shaped by its niche → high local CCC
    cell_residuals = interpreter.compute_cell_residuals(batch_size=1024)
    adata.obs['niche_reconstruction_error'] = cell_residuals
    adata.obs['niche_coupling_score'] = 1.0 / (1.0 + cell_residuals)  # invert for interpretability
 
    for metric, cmap, label, fname in [
        ('niche_coupling_score',       'Reds',  'Niche Coupling Score',        'niche_coupling_score'),
        ('niche_reconstruction_error', 'Blues', 'Niche Reconstruction Error',  'niche_reconstruction_error'),
    ]:
        vals = adata.obs[metric].values
        vmax = np.percentile(vals, 95)
        fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
        sc_plot = ax.scatter(pos[:, 0], pos[:, 1], c=vals, cmap=cmap,
                             s=0.8, alpha=0.8, rasterized=True, vmin=0, vmax=vmax)
        plt.colorbar(sc_plot, ax=ax, shrink=0.6, label=label)
        ax.set_title(f'NCEM GNN: {label} (spatial)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (um)'); ax.set_ylabel('Y (um)')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'figures/{fname}_spatial.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close()
 
    # Per-cell-type spatial coupling
    fig, axes = plt.subplots(
        int(np.ceil(len(cell_types) / 3)), 3,
        figsize=(18, int(np.ceil(len(cell_types) / 3)) * 5),
        facecolor='white'
    )
    axes = axes.flatten()
    for i, ct in enumerate(cell_types):
        ax = axes[i]
        ax.set_facecolor('white')
        ct_mask = adata.obs['cell_type'] == ct
        ax.scatter(pos[:, 0], pos[:, 1], c='lightgrey', s=0.2,
                   alpha=0.2, rasterized=True)
        if ct_mask.sum() > 0:
            vals_ct = adata.obs.loc[ct_mask, 'niche_coupling_score'].values
            sc_plot = ax.scatter(
                pos[ct_mask, 0], pos[ct_mask, 1],
                c=vals_ct, cmap='Reds', s=1.5, alpha=0.9, rasterized=True,
                vmin=0, vmax=np.percentile(vals_ct, 95)
            )
            plt.colorbar(sc_plot, ax=ax, shrink=0.5, label='Coupling score')
        ax.set_title(ct, fontsize=9, fontweight='bold')
        ax.set_aspect('equal')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Niche Coupling Score by Cell Type (NCEM GNN)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/niche_coupling_per_celltype.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
 
    adata.obs[['niche_coupling_score', 'niche_reconstruction_error',
               'cell_type']].to_csv('results/cell_level_coupling_scores.csv')
 
except Exception as e:
    print(f"Spatial coupling maps failed: {e}")
 
#================================
#       10. SAVE RESULTS
#================================

print("Saving final AnnData...")
 
for key in list(adata.obsm.keys()):
    val = adata.obsm[key]
    if not isinstance(val, (np.ndarray, pd.DataFrame)):
        try:
            adata.obsm[key] = np.array(val)
        except Exception:
            del adata.obsm[key]
            print(f"Dropped non-serialisable obsm key: {key}")
 
adata.write_h5ad('BC_prime_ncem.h5ad')
 
print("\nNCEM outputs summary:")
print("  results/ncem_type_coupling.csv          — sender→receiver coupling matrix")
print("  results/r2_per_celltype.csv             — niche predictability per cell type")
print("  results/top_niche_genes_per_celltype.csv— top niche-responsive genes")
print("  results/cell_level_coupling_scores.csv  — per-cell spatial coupling scores")
print("  results/gene_effects_<celltype>.csv     — full gene-level effects per target")
print("  figures/ncem_coupling_heatmap.png       — CCC coupling matrix")
print("  figures/niche_coupling_score_spatial.png— spatial coupling map")
print("Finished.")
