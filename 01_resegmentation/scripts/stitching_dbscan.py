import spatialdata as sd
import geopandas as gpd
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import scanpy as sc
import re
from sklearn.cluster import DBSCAN

plt.rcParams['savefig.facecolor'] = 'white'
sc.set_figure_params(transparent=False, facecolor='white', dpi=2500)

path_core = "/scratch/users/k22026807/masters/project/resegmentation/data/patches_core"
path_seam = "/scratch/users/k22026807/masters/project/resegmentation/data/patches_seams"

# target integrated files to avoid duplicates
core_files = sorted(glob.glob(os.path.join(path_core, "integrated_*.zarr")))
seam_files = sorted(glob.glob(os.path.join(path_seam, "integrated_*.zarr")))
all_files = core_files + seam_files

print(f"Found {len(core_files)} core patches and {len(seam_files)} seam patches.")

gdf_list = []

PATCH_W = 51265 / 4  # ~12816
PATCH_H = 74945 / 8  # ~9368
N_COLS = 4
GLOBAL_HEIGHT = 74945  

print("Loading and transforming Zarr files...")
for idx, file in enumerate(all_files):
    fname = os.path.basename(file)
    try:
        sdata = sd.read_zarr(file)
        
        # 1. SET THE CORRECT KEYS
        if 'patch' in fname:
            shape_key = "cell_boundaries_refined"
            table_key = "table_refined"
        else:
            shape_key = "cell_boundaries_seam_final"
            table_key = "table_refined"

        # 2. LOAD DATA
        shapes_obj = sdata.shapes[shape_key]
        gdf = shapes_obj.compute().copy() if hasattr(shapes_obj, 'compute') else shapes_obj.copy()
        obs = sdata.tables[table_key].obs.copy()

        # 3. TRANSLATE (Move to Global Space)
        if 'patch' in fname:
            match = re.search(r'patch_(\d+)', fname)
            p_num = int(match.group(1))
            col, row = p_num % N_COLS, p_num // N_COLS
            gdf['geometry'] = gdf['geometry'].translate(xoff=col * PATCH_W, yoff=row * PATCH_H)
        else:
            import json
            sidecar = file.replace(".zarr", "_offsets.json")
            if os.path.exists(sidecar):
                with open(sidecar) as f:
                    offsets = json.load(f)
                gdf['geometry'] = gdf['geometry'].translate(xoff=offsets["x_start"], yoff=offsets["y_start"])

        # 4. SET BOUNDARIES (For Centrality)
        if 'patch' in fname:
            gdf['bound_xmin'], gdf['bound_xmax'] = col * PATCH_W, (col + 1) * PATCH_W
            gdf['bound_ymin'], gdf['bound_ymax'] = row * PATCH_H, (row + 1) * PATCH_H
        else:
            b = gdf.total_bounds
            gdf['bound_xmin'], gdf['bound_ymin'], gdf['bound_xmax'], gdf['bound_ymax'] = b[0], b[1], b[2], b[3]

        # 5. FILTERING (One single time)
        gdf['temp_id'] = fname + '_' + gdf.index.astype(str)
        valid_ids_prefixed = set(fname + '_' + str(i) for i in obs.index)
        gdf_filtered = gdf[gdf['temp_id'].isin(valid_ids_prefixed)].copy()

        if len(gdf_filtered) > 0:
            gdf_list.append(gdf_filtered)
            print(f"Successfully added {len(gdf_filtered)} cells from {fname}")

    except Exception as e:
        print(f"Error in {fname}: {e}")

# merge and scale
print("Merging and Scaling...")

combined_gdf = pd.concat(gdf_list, ignore_index=True)
combined_gdf = combined_gdf[combined_gdf.geometry.area < 10000].copy()

# Update centroids after the flip
combined_gdf['cx'] = combined_gdf.geometry.centroid.x
combined_gdf['cy'] = combined_gdf.geometry.centroid.y

# Logic: Pick the cell furthest from its own patch edge
def calculate_centrality(row):
    """Calculates min distance from centroid to the 4 edges of its source patch."""
    dist_x = min(abs(row['cx'] - row['bound_xmin']), abs(row['cx'] - row['bound_xmax']))
    dist_y = min(abs(row['cy'] - row['bound_ymin']), abs(row['cy'] - row['bound_ymax']))
    return min(dist_x, dist_y)

combined_gdf['centrality'] = combined_gdf.apply(calculate_centrality, axis=1)
combined_gdf['source_priority'] = combined_gdf['temp_id'].apply(lambda x: 1 if 'seam' in x else 0)

# deduplication using DBSCAN
print("Running spatial DBSCAN deduplication")
coords = combined_gdf[['cx', 'cy']].values
# eps=20 groups any cells whose centroids are within 3 microns of each other
clustering = DBSCAN(eps=20, min_samples=1).fit(coords)
combined_gdf['cluster_id'] = clustering.labels_

# deduplication using centrality
final_gdf = combined_gdf.sort_values(
    by=['cluster_id', 'centrality', 'source_priority'], 
    ascending=[True, False, False]
).drop_duplicates(subset=['cluster_id'], keep='first').copy()

print(f"Pre-deduplication total cells: {len(combined_gdf)}")
print(f"Post-deduplication total cells: {len(final_gdf)}")

# save
final_gdf.to_file("data/final_stitched_dbscan_updated.gpkg", driver="GPKG", mode='w')

# plain plot without labels
fig, ax = plt.subplots(figsize=(10, 15))
ax.set_aspect('equal')
final_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.03, aspect=None)
ax.invert_yaxis()
plt.savefig("/scratch/users/k22026807/masters/project/resegmentation/results/final_stitched.png", dpi=2500, bbox_inches='tight', facecolor='white')
plt.close()

# plot with explicit limits to verify tiling and patch labels
fig, ax = plt.subplots(figsize=(10, 15))
ax.set_aspect('equal')
final_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.03, aspect=None)
ax.invert_yaxis()

for file in all_files:
    fname = os.path.basename(file)
    try:
        if 'patch' in fname:
            match = re.search(r'patch_(\d+)', fname)
            patch_num = int(match.group(1))
            col, row = patch_num % N_COLS, patch_num // N_COLS
            
            # X center remains the same
            cx = (col + 0.5) * PATCH_W
            cy = (row + 0.5) * PATCH_H
            
            ax.text(cx, cy, f'P{patch_num}', fontsize=6, ha='center', 
                    va='center', color='red', fontweight='bold')
        else:
            # For seams, extract seam id and plot at centroid of its current bounds
            match = re.search(r'seam_([HV])_(\d+)', fname)
            if match:
                seam_type = match.group(1)
                seam_num = int(match.group(2))
                
                # Get the actual bounds from the final flipped dataframe
                seam_mask = final_gdf['temp_id'].str.startswith(fname)
                if seam_mask.sum() > 0:
                    bounds = final_gdf[seam_mask].total_bounds
                    cx = (bounds[0] + bounds[2]) / 2
                    cy = (bounds[1] + bounds[3]) / 2
                    ax.text(cx, cy, f'S{seam_type}{seam_num}', fontsize=4, 
                            ha='center', va='center', color='green')
    except Exception as e:
        pass

plt.savefig("/scratch/users/k22026807/masters/project/resegmentation/results/final_stitched_labelled.png", dpi=2500, bbox_inches='tight')
plt.close()


# visualise overlap region between SH9, SH10, SV10, SV7 for diagnostics
# first find the coordinate bounds of this intersection
overlap_seams = ['integrated_proseg_seam_H_09', 'integrated_proseg_seam_H_10',
                 'integrated_proseg_seam_V_07', 'integrated_proseg_seam_V_10']
colors = {'H_09': 'red', 'H_10': 'blue', 'V_07': 'green', 'V_10': 'orange'}

# get zoom bounds from intersection of all four seams
all_bounds = []
seam_gdfs = {}
for seam_name in overlap_seams:
    key = seam_name.split('seam_')[1]
    # use combined_gdf (before dedup) to see ALL cells from each seam
    mask = combined_gdf['temp_id'].str.contains(seam_name)
    if mask.sum() > 0:
        seam_gdfs[key] = combined_gdf[mask]
        all_bounds.append(combined_gdf[mask].total_bounds)

if all_bounds:
    bounds = np.array(all_bounds)
    xmin = bounds[:, 0].max()
    xmax = bounds[:, 2].min()
    ymin = bounds[:, 1].max()
    ymax = bounds[:, 3].min()

    if xmin >= xmax or ymin >= ymax:
        print("4-way intersection empty - defaulting to center of junction")
        union_xmin, union_ymin = bounds[:, 0].min(), bounds[:, 1].min()
        union_xmax, union_ymax = bounds[:, 2].max(), bounds[:, 3].max()
        cx, cy = (union_xmin + union_xmax) / 2, (union_ymin + union_ymax) / 2
        
        # Create a fixed 1000-unit viewing window around the center
        xmin, xmax = cx - 500, cx + 500
        ymin, ymax = cy - 500, cy + 500
    else:
        buffer_size = 300 
        xmin -= buffer_size
        xmax += buffer_size
        ymin -= buffer_size
        ymax += buffer_size

    # window masks using fast cx/cy
    combined_mask = (
        (combined_gdf['cx'] >= xmin) & (combined_gdf['cx'] <= xmax) &
        (combined_gdf['cy'] >= ymin) & (combined_gdf['cy'] <= ymax)
    )
    final_mask = (
        (final_gdf['cx'] >= xmin) & (final_gdf['cx'] <= xmax) &
        (final_gdf['cy'] >= ymin) & (final_gdf['cy'] <= ymax)
    )

    combined_window = combined_gdf[combined_mask]
    final_window = final_gdf[final_mask]

    # diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
    axes = axes.flatten()

    # pre-dedup
    ax = axes[0]
    ax.set_facecolor('white')
    for key, gdf in seam_gdfs.items():
        gdf_window = gdf[ 
            (gdf['cx'] >= xmin) & (gdf['cx'] <= xmax) & 
            (gdf['cy'] >= ymin) & (gdf['cy'] <= ymax) 
        ]
        if not gdf_window.empty:
            gdf_window.plot(ax=ax, facecolor='none', edgecolor=colors[key], linewidth=0.8, alpha=0.7, label=key, aspect=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f'PRE-DEDUP: All Seams Overlaid\nTotal Cells in Window: {len(combined_window)}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # post-dedup
    ax = axes[1]
    ax.set_facecolor('white')
    final_window.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8, alpha=0.9, aspect=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f'POST-DEDUP: Final Landscape\nTotal Cells in Window: {len(final_window)}', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')

    # individual seams
    for i, (key, gdf) in enumerate(seam_gdfs.items()):
        ax = axes[i + 2]
        ax.set_facecolor('white')
        
        gdf_window = gdf[ 
            (gdf['cx'] >= xmin) & (gdf['cx'] <= xmax) & 
            (gdf['cy'] >= ymin) & (gdf['cy'] <= ymax) 
        ]
        
        combined_window.plot(ax=ax, facecolor='lightgrey', edgecolor='grey', linewidth=0.3, alpha=0.5, aspect=1)
        
        if not gdf_window.empty:
            gdf_window.plot(ax=ax, facecolor='none', edgecolor=colors[key], linewidth=1.2, aspect=1)
            
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f'Seam {key} Only\nCells in Window: {len(gdf_window)}', fontsize=10)
        ax.set_aspect('equal')

    plt.suptitle('Deduplication Diagnostics: Overlap Region', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/scratch/users/k22026807/masters/project/resegmentation/results/overlap_check_counts.png', dpi=2500, bbox_inches='tight', facecolor='white')
    plt.close()

