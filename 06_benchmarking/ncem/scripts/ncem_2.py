#### NCEM ####
# node-centric expression models (GNN)
# graph - radius-based (50um)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use cpu
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
from shapely.geometry import Point, Polygon as ShapelyPolygon
import ncem
from ncem.estimators import EstimatorEDncem
from ncem.data import customLoader
from ncem.models import ModelEDncem
import scipy.sparse as sp
import tensorflow as tf
import types
import scipy.sparse as sparse_sci
import squidpy as sq

SKIP_TRAINING = False

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
#sc.pp.normalize_total(adata, inplace=True)
#sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=500, flavor='seurat')

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

n_genes = adata.shape[1] 
n_labels = len(adata.obs['cell_type'].cat.categories)

print(f"DEBUG: adata.shape = {adata.shape}")
print(f"DEBUG: n_genes={n_genes}, n_labels={n_labels}")

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

sq.gr.spatial_neighbors(
    adata,
    coord_type='generic',
    radius=50.0,          # ~7 neighbours — comparable to paper density
    key_added='spatial_knn'
)

degrees_fixed = np.array(
    (adata.obsp['spatial_knn_connectivities'] > 0).sum(axis=1)
).flatten()
print(f"Fixed graph: mean degree={degrees_fixed.mean():.1f}, "
      f"median={np.median(degrees_fixed):.1f}, "
      f"max={degrees_fixed.max()}")

# extract nodes and edges
n_nodes = graph_adata.adata.n_obs
n_edges = graph_adata.adata.obsp['distances'].nnz
mean_degree = n_edges / n_nodes

print(f"Graph nodes: {n_nodes}")
print(f"Graph edges: {n_edges}")
print(f"Mean degree: {mean_degree:.2f}")

# QC - degree distribution
degrees = np.array(
    (adata.obsp['spatial_knn_connectivities'] > 0).sum(axis=1)
        ).flatten()

fig, ax = plt.subplots(figsize=(7,5), facecolor='white')
ax.hist(degrees, bins=40, color='steelblue', edgecolor='white')
ax.set_xlabel('Node degree (neighbors within 50um)')
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
rows, cols = adata.obsp['spatial_knn_connectivities'].nonzero()
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
ax.set_title('Spatial Graph (50 um radius)', fontsize=12, fontweight='bold')
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

    """Bridge AnnData to EstimatorEDncem for custom datasets."""

    est = EstimatorEDncem(
        cond_type='gcn',
        use_type_cond=True,
        log_transform=False,
    )

    est.data = graph_loader
    
    # image / patient keys
    est.img_keys          = [img_key]
    est.complete_img_keys = [img_key]
    est.img_keys_eval     = []
    est.img_keys_test     = []
    est.img_keys_train    = [img_key]
    est.img_to_patient_dict = {img_key: 'sample1'}
    est.undefined_node_types = None
    est.domains           = {img_key: 0}

    # expression data
    dense_X = np.array(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                   dtype=np.float32)
    est.h_0 = {img_key: dense_X}

    # adjacency — _get_dataset uses self.a and self.cond_depth
    adj = sp.csr_matrix(adata.obsp['spatial_knn_connectivities'].copy())
    adj.setdiag(1.0)
    est.a = {img_key: adj}

    # cond_depth MUST be set — _get_dataset reads self.cond_depth directly
    est.cond_depth = 1

    # cell type labels (h_1)
    dummy_df = pd.get_dummies(adata.obs['cell_type'])
    est.h_1          = {img_key: dense_X}
    est.node_labels  = {img_key: dummy_df.values.astype(np.float32)}
    est.node_features = {img_key: dense_X}

    # covariates — zero-width arrays (no covariates)
    est.node_covar  = {img_key: dummy_df.values.astype(np.float32)}   # was zeros
    est.node_covariates = {img_key: dummy_df.values.astype(np.float32)}

    est.graph_covar     = {img_key: np.zeros((1, 0), dtype=np.float32)}

    # size factors — ones (no normalisation adjustment needed)
    est.size_factors = {img_key: np.ones(adata.shape[0], dtype=np.float32)}

    # node indices
    est.nodes_idx = {img_key: np.arange(adata.shape[0])}

    # shape registrations
    est.n_features_0          = adata.shape[1]
    est.n_features_1          = adata.shape[1]
    est.n_node_labels         = dummy_df.shape[1]
    est.n_domains             = 1
    est.n_node_features       = adata.shape[1]
    est.max_nodes             = adata.shape[0]
    est.n_eval_nodes_per_graph = min(50, adata.shape[0])
    est.n_node_covariates     = dummy_df.shape[1]
    est.n_graph_covariates    = 0
    est.n_features_standard   = adata.shape[1]
    est.n_features_type       = dummy_df.shape[1]

    # names
    est.node_label_names  = list(dummy_df.columns)
    est.node_feature_names = list(adata.var_names)
    est.node_type_names   = {i: ct for i, ct in enumerate(dummy_df.columns)}
    est.graph_covar_names = []

    # interpreter support
    est.model_class = 'ed_ncem'
    est._model_kwargs = {'input_shapes': None}
    est.model_id    = 'ncem_custom'
    est.vi_model    = False

    est.test_img_keys = []
    est.val_img_keys  = []

    return est

#===========================
# 4. NCEM MODEL DEFINITION
#===========================

print("Initialising NCEM estimator...")

estimator = bind_custom_ncem_data(adata, graph_adata)

#estimator.init_model(
#    model_type='ed',         # Encoder-Decoder (GNN)
#    interact_type='all', 
#    n_features_interact=estimator.n_node_features,
#    latent_dim=128,
#    dropout_rate=0.1,
#    l2_reg=1e-5
#)

print("Splitting data...")
estimator.split_data_node(
    validation_split=0.1,
    test_split=0.1,
    seed=42)

print("Building model architecture...")


if getattr(estimator, 'model', None) is None:
    print("Manual binding of model required...")
    # manually define shapes since internal helpers are missing
    # n_features_0 = genes, n_features_1 = cell types
    input_shapes = (
        n_genes,                            # [0] in_node_feature_dim
        n_genes,                            # [1] out_node_feature_dim (reconstruct genes)
        estimator.max_nodes,                # [2] graph_dim = full graph size
        estimator.n_eval_nodes_per_graph,   # [3] in_node_dim = 500
        n_labels,                           # [4] categ_condition_dim
        1,                                  # [5] domain_dim
    )

    print(f"debug: input_shapes = {input_shapes}")
    
    estimator._model_kwargs['input_shapes'] = input_shapes

    estimator.model = ModelEDncem(
        input_shapes=input_shapes,
        latent_dim=128,
        dropout_rate=0.1,
        l2_coef=1e-5,
        use_type_cond=True,
        output_layer='nb_shared_disp', #one dimension parameter shared across all genes
        cond_type='gcn',      # explicit — GCN graph conditioning
        cond_depth=1,
        cond_dim=64,
        )

def nb_loss(y_true, y_pred):
    n_genes = tf.shape(y_true)[-1]
    mu      = tf.nn.softplus(y_pred[:, :, :n_genes]) + 1e-8
    theta   = tf.nn.softplus(y_pred[:, :, n_genes:]) + 1e-8
    log_theta_mu_eps = tf.math.log(theta + mu + 1e-8)
    nll = -(
        tf.math.lgamma(y_true + theta)
        - tf.math.lgamma(theta)
        - tf.math.lgamma(y_true + 1.0)
        + theta * (tf.math.log(theta + 1e-8) - log_theta_mu_eps)
        + y_true * (tf.math.log(mu + 1e-8) - log_theta_mu_eps)
    )
    return tf.reduce_mean(nll)

estimator.model.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=nb_loss)
print("Model compiled successfully")

try:
    print(estimator.model.training_model.summary())
except AttributeError:
    print("Model built successfully, proceeding to training...")

#===========================
#       5. TRAINING
#===========================

# safety check
required_attrs = ['model', 'img_keys_train', 'h_0', 'a']
for attr in required_attrs:
    if getattr(estimator, attr, None) is None:
        raise RuntimeError(f"Missing critical attribute: {attr}. NCEM will fail.")

if estimator.model.training_model is None:
    raise RuntimeError("The Keras training model was not built correctly.")

if not SKIP_TRAINING:
    print("Training NCEM GNN...")
    estimator.train(
        epochs=200,
        batch_size=1,
        max_steps_per_epoch=500,     # limits steps per epoch for large datasets
        patience=30,                # early stopping patience
        lr_schedule_min_lr=1e-5,
        lr_schedule_factor=0.2,
        lr_schedule_patience=15,
        early_stopping=True,
        reduce_lr_plateau=True,
        log_dir='models/ncem_logs',
        )

    os.makedirs('models', exist_ok=True)
    estimator.model.training_model.save_weights('models/ncem_weights.h5')
    print("Saved model weights.")
else:
    print("Skipping training — loading saved weights...")
    estimator.model.training_model.load_weights('models/ncem_weights.h5')

#===========================
#      6. EVALUATION
#===========================

print("Evaluating model...")

# evaluate_any requires explicit img_keys and node_idx
try:
    eval_dict = estimator.evaluate_any(
        img_keys=estimator.img_keys_train,
        node_idx=estimator.nodes_idx_train,
        batch_size=1,
    )
    print("Evaluation metrics:", eval_dict)
    pd.DataFrame([eval_dict]).to_csv('results/eval_metrics.csv', index=False)
except TypeError:
    results = estimator.model.training_model.evaluate(
        estimator._get_dataset(
            image_keys=estimator.img_keys_train,
            nodes_idx=estimator.nodes_idx_train,
            batch_size=1,
            shuffle_buffer_size=None,
            train=False,
        ),
        verbose=0
    )
    eval_dict = {'loss': float(results) if np.isscalar(results) else float(results[0])}
    print("Evaluation metrics:", eval_dict)
    pd.DataFrame([eval_dict]).to_csv('results/eval_metrics.csv', index=False)

# Training curve
history = estimator.history
if history:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='white')
    for ax, metric, title in zip(
        axes,
        ['loss', 'val_loss'],
        ['Training Loss', 'Validation Loss']
    ):
        if metric in history:
            ax.plot(history[metric], color='steelblue')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

#===========================
#  7. CCC INTERPRETATION
#===========================


print("Running CCC interpretation...")

# Use InterpreterEDncem — correct class for ModelEDncem
from ncem.interpretation import InterpreterEDncem

interpreter = InterpreterEDncem()

# Copy all estimator attributes to interpreter
# InterpreterEDncem inherits from the same base — it needs all the same data
for attr in vars(estimator):
    try:
        setattr(interpreter, attr, getattr(estimator, attr))
    except Exception:
        pass

interpreter.cell_names = list(estimator.node_type_names.values())

# Reinitialize model for gradient computation in saliencies
# This creates self.reinit_model which target_cell_saliencies requires
print("Reinitialising model for interpretation...")
interpreter.n_eval_nodes_per_graph = 1
interpreter._model_kwargs['input_shapes'] = (
    n_genes,                    # in_node_feature_dim
    n_genes,                    # out_node_feature_dim
    estimator.max_nodes,        # graph_dim
    1,                          # in_node_dim = 1 (not 50)
    n_labels,                   # categ_condition_dim
    1,                          # domain_dim
)

interpreter.reinitialize_model(changed_model_kwargs={
    'cond_dim': 64,
    'cond_depth': 1,
    'latent_dim': 128,
    'dropout_rate': 0.1,
    'l2_coef': 1e-5,
    'use_type_cond': True,
    'output_layer': 'nb_shared_disp',
    'cond_type': 'gcn',
    })

interpreter.model.training_model.load_weights('models/ncem_weights.h5')
print("Loaded model weights into interpreter.")

dummy_df = pd.get_dummies(adata.obs['cell_type'])
interpreter.node_labels_matrix = dummy_df.values.astype(np.float32)

if not hasattr(np, 'float'):
    np.float = float


# use full graph adjacency from estimator not batch adjacency
def patched_neighbourhood_frequencies(self, a, h_0_full, discretize_adjacency=True):
    # Use the full stored adjacency matrix instead of batch adj
    # self.a contains the full graph adjacency
    img_key = self.img_keys_train[0]
    full_adj = self.a[img_key]  # sparse (53372, 53372)
    
    # Convert to binary
    full_adj_binary = full_adj.copy()
    full_adj_binary.data = np.ones_like(full_adj_binary.data)
    
    # node_labels_matrix: (53372, 10)
    # For each node, count how many neighbours belong to each cell type
    # Result: (53372, 10)
    node_label_mat = self.node_labels_matrix.astype(np.float32)
    neighbourhood_counts = full_adj_binary @ node_label_mat  # (53372, 10)
    
    # Average over all nodes (or sum — either gives relative frequencies)
    result = np.asarray(neighbourhood_counts.mean(axis=0)).flatten()
    
    return pd.DataFrame(result.reshape(1, -1), columns=self.cell_names)

interpreter._neighbourhood_frequencies = types.MethodType(
    patched_neighbourhood_frequencies, interpreter)

def patched_target_cell_saliencies(self, target_cell_type, drop_columns=None,
                                    dop_images=None, partition='test',
                                    multicolumns=None):
    from tqdm import tqdm
    import scipy.sparse as sparse
    target_cell_idx = list(self.node_type_names.values()).index(target_cell_type)

    if partition == "train":
        img_keys  = self.img_keys_train
        node_idxs = self.nodes_idx_train
    elif partition == "val":
        img_keys  = self.img_keys_eval
        node_idxs = self.nodes_idx_eval
    else:
        img_keys  = self.img_keys_train
        node_idxs = self.nodes_idx_train

    img_saliency = []
    keys = []
    with tqdm(total=len(img_keys)) as pbar:
        for key in img_keys:
            idx = {key: node_idxs[key]}
            ds = self._get_dataset(
                image_keys=[key],
                nodes_idx=idx,
                batch_size=1,
                shuffle_buffer_size=1,
                train=False,
                seed=None,
                reinit_n_eval=1,
            )
            saliencies = []
            h_1        = []
            h_0        = []
            h_0_full   = []
            a          = []

            MAX_STEPS_PER_CT = 100
            step_count = 0

            for _step, (x_batch, _y_batch) in enumerate(ds):
                if step_count >= MAX_STEPS_PER_CT:
                    break
                h_1_batch      = x_batch[0].numpy().squeeze()      # gene expression
                h_0_batch      = x_batch[6].numpy()[0, 0, :]
                h_0_full_batch = x_batch[3]                        # full graph (1, 53372, 500)
                a_batch        = x_batch[5]
                h_0_full_labels = self.node_labels_matrix                    # full adjacency SparseTensor

                if len(h_0_batch.shape) == 0:
                    continue
                if h_0_batch[target_cell_idx] == 1.0:
                    step_count += 1
                    h_1.append(h_1_batch)
                    h_0.append(h_0_batch)
                    h_0_full.append(h_0_full_labels)
                    a.append(
                        sparse.csr_matrix(
                            (
                                a_batch.values.numpy(),
                                (a_batch.indices.numpy()[:, 1],
                                 a_batch.indices.numpy()[:, 2]),
                            ),
                            shape=a_batch.dense_shape.numpy()[1:],
                        ).toarray()
                    )
                    with tf.GradientTape(persistent=True) as tape:
                        tape.watch([h_0_full_batch])
                        model_out = self.reinit_model.training_model(x_batch)[0]
                    grads = tape.gradient(model_out, h_0_full_batch)[0].numpy()
                    grads_squeezed = np.squeeze(grads)  # (53372, 500)
                    if grads_squeezed.ndim == 1:
                        grads_squeezed = np.expand_dims(grads_squeezed, axis=-1)

                    node_importance = np.abs(grads_squeezed).sum(axis=-1)  # (53372,)
                    n_labels = len(self.node_type_names)

                    n_ct = len(self.node_type_names)
                    grads_for_pp = (h_0_full_labels * node_importance[:, np.newaxis]).T  # (n_celltypes, n_nodes)

                    grads_pp = self._pp_saliencies(
                        gradients=grads_for_pp,
                        h_0=h_0_batch,
                        h_0_full=np.eye(n_ct),
                        remove_own_gradient=False,
                        absolute_saliencies=False,
                    )
                    saliencies.append(grads_pp)

            if len(saliencies) == 0:
                continue
            saliencies   = np.concatenate(saliencies, axis=0)
            saliencies   = np.sum(saliencies, axis=0)

            neighbourhood = self._neighbourhood_frequencies(
                a=a, h_0_full=h_0_full, discretize_adjacency=True)
            neighbourhood = np.array(neighbourhood).flatten()

            neighbourhood_safe = np.where(neighbourhood == 0, np.nan, neighbourhood)
            img_saliency.append(np.where(
                neighbourhood == 0, 0.0, saliencies / neighbourhood_safe))
            keys.append(key)
            pbar.update(1)

    if not img_saliency:
        raise ValueError(f"No cells of type '{target_cell_type}' found in partition")

    df = pd.DataFrame(
        np.concatenate(np.expand_dims(img_saliency, axis=0), axis=1).T,
        columns=keys,
        index=list(self.node_type_names.values()),
    )
    df = df.reindex(sorted(df.columns), axis=1)
    if drop_columns:
        df = df.drop(drop_columns)
    return df



# Bind patched method to interpreter instance
interpreter.target_cell_saliencies = types.MethodType(
    patched_target_cell_saliencies, interpreter)

# target_cell_saliencies uses 'train' partition since test is empty
# Returns DataFrame: rows=sender cell types, cols=image keys
print("Attempting saliency computation...")
coupling_rows = {}

priority_cell_types = cell_types  # all cell types
print(f"Priority cell types: {priority_cell_types}")

for target_ct in priority_cell_types:
    try:
        sal_df = interpreter.target_cell_saliencies(
            target_cell_type=target_ct,
            partition='train',
        )
        coupling_rows[target_ct] = sal_df.mean(axis=1)
        print(f"  Saliencies computed for: {target_ct}")
    except Exception as e:
        print(f"  Saliencies failed for {target_ct}: {e}")

if coupling_rows:
    import seaborn as sns
    coupling_df = pd.DataFrame(coupling_rows).T
    print("Type coupling matrix:\n", coupling_df.to_string())

    # Abundance normalisation
    cell_type_counts = adata.obs['cell_type'].value_counts()
    coupling_df_norm = coupling_df.copy()
    for ct in coupling_df_norm.columns:
        if ct in cell_type_counts.index:
            coupling_df_norm[ct] = coupling_df_norm[ct] / cell_type_counts[ct]

    coupling_df.to_csv('results/ncem_type_coupling_raw.csv')
    coupling_df_norm.to_csv('results/ncem_type_coupling_normalised.csv')

    # Both heatmaps — raw and normalised
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), facecolor='white')
    for ax, df, title in zip(axes,
        [coupling_df, coupling_df_norm],
        ['Raw saliency', 'Abundance-normalised saliency']):
        sns.heatmap(df, cmap='seismic', center=0, ax=ax,
                    xticklabels=True, yticklabels=True)
        ax.set_title(f'NCEM: {title}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Sender'); ax.set_ylabel('Receiver')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig('figures/ncem_coupling_heatmap_both.png', dpi=300, bbox_inches='tight')
    plt.close()

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
        ax.set_ylabel('Saliency (coupling)')
        ax.set_title(f'→ {target}', fontsize=9, fontweight='bold')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Sender Influence per Target Cell Type', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ncem_sender_ranking_per_target.png', dpi=300, bbox_inches='tight')
    plt.close()

    total_sent = coupling_df.sum(axis=0).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.bar(range(len(total_sent)), total_sent.values,
           color=[custom_palette.get(ct, 'grey') for ct in total_sent.index])
    ax.set_xticks(range(len(total_sent)))
    ax.set_xticklabels(total_sent.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Total saliency (sum over all receivers)')
    ax.set_title('NCEM GNN: Total Outgoing Signal per Cell Type',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ncem_total_outgoing_signal.png', dpi=300, bbox_inches='tight')
    plt.close()

    total_recv = coupling_df.sum(axis=1).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.bar(range(len(total_recv)), total_recv.values,
           color=[custom_palette.get(ct, 'grey') for ct in total_recv.index])
    ax.set_xticks(range(len(total_recv)))
    ax.set_xticklabels(total_recv.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Total saliency (sum over all senders)')
    ax.set_title('NCEM GNN: Total Incoming Signal per Cell Type',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ncem_total_incoming_signal.png', dpi=300, bbox_inches='tight')
    plt.close()

else:
    print("All saliencies failed — skipping coupling matrix plots")

#================================
#  8. GENE-LEVEL INTERPRETATION
#================================

# Replace get_decoding_weights() with direct layer extraction
decoder_weights = None
for layer in interpreter.model.training_model.layers:
    w = layer.get_weights()
    if w and w[0].shape == (128, 500):  # latent_dim x n_genes
        decoder_weights = w[0]
        print(f"Found decoder layer: {layer.name}, shape: {w[0].shape}")
        break
    elif w and w[0].shape == (500, 128):
        decoder_weights = w[0].T
        print(f"Found decoder layer (transposed): {layer.name}")
        break

if decoder_weights is not None:
    gene_importance = np.abs(decoder_weights).mean(axis=0)
else:
    print("Decoder weights not found — using first large weight matrix")
    for layer in interpreter.model.training_model.layers:
        w = layer.get_weights()
        if w and max(w[0].shape) == 500:
            gene_importance = np.abs(w[0]).mean(axis=0)
            if gene_importance.shape[0] != 500:
                gene_importance = np.abs(w[0]).mean(axis=1)
            break

if gene_importance is not None:
    gene_imp_df = pd.DataFrame({
        'gene': adata.var_names,
        'mean_abs_weight': gene_importance
    }).sort_values('mean_abs_weight', ascending=False)
    gene_imp_df.to_csv('results/gene_decoding_weights.csv', index=False)

    top_genes = gene_imp_df.head(30)
    fig, ax = plt.subplots(figsize=(8, 10), facecolor='white')
    ax.barh(top_genes['gene'][::-1], top_genes['mean_abs_weight'][::-1],
            color='steelblue')
    ax.set_xlabel('Mean absolute decoding weight')
    ax.set_title('Top 30 Genes by Decoding Weight\n(NCEM GNN)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ncem_top_genes_decoding.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top gene: {gene_imp_df.iloc[0]['gene']} ({gene_imp_df.iloc[0]['mean_abs_weight']:.4f})")

#================================
#  9. LATENT EMBEDDING + SPATIAL
#================================

print("Computing latent embeddings...")

try:
    # --- FIX: Build a direct embedding model ---
    # Extract the encoder layer from the compiled training model
    encoder_layer = estimator.model.training_model.get_layer('encoder')
    
    # Create a submodel that takes standard inputs and outputs ONLY the mean (index 0)
    embedding_model = tf.keras.Model(
        inputs=estimator.model.training_model.inputs,
        outputs=encoder_layer.output[0]  
    )
    # -------------------------------------------

    ds = estimator._get_dataset(
        image_keys=estimator.img_keys_train,
        nodes_idx=estimator.nodes_idx_train,
        batch_size=1,
        shuffle_buffer_size=None,
        train=False,
    )
    
    all_embeddings = []
    for x_batch, _ in ds:
        # Pass the raw batch directly into our custom embedding model
        z_mean = embedding_model(x_batch, training=False)
        
        # Safely squeeze and append
        all_embeddings.append(np.squeeze(z_mean.numpy())) 

    embeddings = np.vstack(all_embeddings)  # Should correctly be (n_train_cells, 128)
    print(f"Embedding shape: {embeddings.shape}")

    # Note: embeddings only cover training nodes, not all 53372
    # Get the training node indices to match spatial coordinates
    train_idx = estimator.nodes_idx_train[estimator.img_keys_train[0]]
    n = min(len(embeddings), len(train_idx))
    embeddings = embeddings[:n]
    train_idx  = train_idx[:n]
    pos_train  = pos[train_idx]

    import anndata as ad
    emb_adata = ad.AnnData(X=embeddings)
    emb_adata.obs['cell_type'] = adata.obs['cell_type'].values[train_idx]
    sc.pp.neighbors(emb_adata, use_rep='X', n_neighbors=15)
    sc.tl.umap(emb_adata)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    for ct, col in custom_palette.items():
        mask = emb_adata.obs['cell_type'] == ct
        if mask.sum() > 0:
            ax.scatter(emb_adata.obsm['X_umap'][mask, 0],
                       emb_adata.obsm['X_umap'][mask, 1],
                       s=0.5, c=col, label=ct, alpha=0.7, rasterized=True)
    ax.set_title('NCEM Latent Embedding (UMAP)', fontsize=12, fontweight='bold')
    ax.legend(markerscale=6, fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/ncem_latent_umap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # After the main UMAP figure, add:
    # Per cell type UMAP — shows neighbourhood structure within each type
    major_types = ['Malignant cell', 'T cell', 'Epithelial cell', 'Myeloid cell']
    fig, axes = plt.subplots(1, len(major_types),
                              figsize=(len(major_types)*5, 5),
                              facecolor='white')
    for ax, ct in zip(axes, major_types):
        mask = emb_adata.obs['cell_type'] == ct
        if mask.sum() < 10:
            ax.set_visible(False)
            continue
        ct_adata = emb_adata[mask].copy()
        sc.pp.neighbors(ct_adata, use_rep='X', n_neighbors=min(15, mask.sum()-1))
        sc.tl.umap(ct_adata)
        ax.scatter(ct_adata.obsm['X_umap'][:, 0],
                   ct_adata.obsm['X_umap'][:, 1],
                   s=1, c=custom_palette.get(ct, 'grey'),
                   alpha=0.5, rasterized=True)
        ax.set_title(f'{ct}\n(n={mask.sum():,})', fontsize=9)
        ax.axis('off')
    plt.suptitle('Per-cell-type latent UMAP\n(neighbourhood structure within type)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ncem_latent_umap_per_celltype.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
    for i, ax in enumerate(axes):
        vals = embeddings[:, i]
        vmax = np.percentile(np.abs(vals), 95)
        sc_plot = ax.scatter(pos_train[:, 0], pos_train[:, 1], c=vals,
                             cmap='seismic', s=0.5, alpha=0.8,
                             vmin=-vmax, vmax=vmax, rasterized=True)
        plt.colorbar(sc_plot, ax=ax, shrink=0.6)
        ax.set_title(f'Latent dim {i+1} (spatial)', fontsize=10)
        ax.set_aspect('equal')
    plt.suptitle('NCEM Latent Space — Spatial Projection',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ncem_latent_spatial.png', dpi=300, bbox_inches='tight')
    plt.close()

    np.save('results/ncem_embeddings.npy', embeddings)
    np.save('results/ncem_train_indices.npy', train_idx)
    print("Saved embeddings and train indices")

except Exception as e:
    print(f"Embedding computation failed: {e}")
 
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