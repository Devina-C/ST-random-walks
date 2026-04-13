import spatialdata as sd
import geopandas as gpd
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import scanpy as sc
import re

# Standardize visual environment
plt.rcParams['savefig.facecolor'] = 'white'
sc.set_figure_params(transparent=False, facecolor='white', dpi=1000)

path_core = "/scratch/users/k22026807/masters/project/resegmentation/patches_core"
path_seam = "/scratch/users/k22026807/masters/project/resegmentation/patches_seams"

# target integrated files to avoid duplicates
core_files = sorted(glob.glob(os.path.join(path_core, "integrated_*.zarr")))
seam_files = sorted(glob.glob(os.path.join(path_seam, "integrated_*.zarr")))
all_files = core_files + seam_files

print(f"Found {len(core_files)} core patches and {len(seam_files)} seam patches.")

gdf_list = []

# based on 4x8 grid setup
#WHOLE_ROI = {'xmin': 0, 'ymin': 0, 'width': 51265, 'height': 74945}
#n_cols = 4
#n_rows = 8
PATCH_W = 51265 / 4  # ~12816
PATCH_H = 74945 / 8  # ~9368
N_COLS = 4

for idx, file in enumerate(all_files):
    fname = os.path.basename(file)
    
    try:
        if 'patch' in fname:
            match = re.search(r'patch_(\d+)', fname)
            patch_num = int(match.group(1))
            col = patch_num % N_COLS
            row = patch_num // N_COLS
            x_offset = col * PATCH_W
            y_offset = row * PATCH_H
        else:
            x_offset, y_offset = 0, 0

        sdata = sd.read_zarr(file)
        shape_key = next((k for k in sdata.shapes.keys() if 'refined' in k or 'reseg' in k), list(sdata.shapes.keys())[0])
        gdf = sdata.shapes[shape_key].copy()

        # flip y axis for all patches (proseg uses bottom-up y convention)
        if 'patch' in fname:
            gdf['geometry'] = gdf['geometry'].affine_transform([1, 0, 0, -1, 0, 0])
            gdf['geometry'] = gdf['geometry'].translate(xoff=x_offset, yoff=y_offset + PATCH_H)
        else:
            # for seams, flip around the seam's own y centre
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            y_centre = (bounds[1] + bounds[3]) / 2
            gdf['geometry'] = gdf['geometry'].affine_transform([1, 0, 0, -1, 0, 2 * y_centre])
            gdf['geometry'] = gdf['geometry'].translate(xoff=x_offset, yoff=y_offset)

        # filtering
        gdf['temp_id'] = fname + '_' + gdf.index.astype(str)
        obs = sdata.tables['table_refined'].obs.copy()
        valid_ids = obs.index.astype(str)
        valid_ids_prefixed = set(fname + '_' + i for i in valid_ids)
        gdf_filtered = gdf[gdf['temp_id'].isin(valid_ids_prefixed)].copy()

        if len(gdf_filtered) > 0:
            gdf_list.append(gdf_filtered)

    except Exception as e:
        print(f"Error in {fname}: {e}")


# merge and scale
print("Merging and Scaling...")
combined_gdf = pd.concat(gdf_list)
combined_gdf = combined_gdf[combined_gdf.geometry.area < 10000].copy()

# compute centroids from transformed geometries
combined_gdf = combined_gdf.set_crs(None, allow_override=True)
combined_gdf['cx'] = combined_gdf.geometry.centroid.x.round(0)
combined_gdf['cy'] = combined_gdf.geometry.centroid.y.round(0)
combined_gdf['source'] = combined_gdf['temp_id'].apply(lambda x: 'seam' if 'seam' in x else 'core')
combined_gdf = combined_gdf.sort_values('source', ascending=False)

# centroid deduplication
final_gdf = combined_gdf.drop_duplicates(subset=['cx', 'cy'], keep='first').copy()

# Save
final_gdf.to_file("final_stitched_resegmentation.gpkg", driver="GPKG", mode='w')

# plain plot without labels
fig, ax = plt.subplots(figsize=(10, 15))
ax.set_aspect('equal')
final_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.03, aspect=None)
plt.savefig("final_stitched.png", dpi=1000, bbox_inches='tight', facecolor='white')
plt.close()

# Plot with explicit limits to verify tiling and patch labels
fig, ax = plt.subplots(figsize=(10, 15))
ax.set_aspect('equal')
final_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.03, aspect=None)

# add label for each patch at its centroid
for file in all_files:
    fname = os.path.basename(file)
    try:
        if 'patch' in fname:
            match = re.search(r'patch_(\d+)', fname)
            patch_num = int(match.group(1))
            col = patch_num % N_COLS
            row = patch_num // N_COLS
            cx = (col + 0.5) * PATCH_W
            cy = (row + 0.5) * PATCH_H
            ax.text(cx, cy, f'P{patch_num}', fontsize=6, ha='center', 
                   va='center', color='red', fontweight='bold')
        else:
            # for seams extract seam id and plot at centroid of seam bounds
            match = re.search(r'seam_([HV])_(\d+)', fname)
            if match:
                seam_type = match.group(1)
                seam_num = int(match.group(2))
                # get seam gdf from final_gdf using temp_id
                seam_mask = final_gdf['temp_id'].str.startswith(fname)
                if seam_mask.sum() > 0:
                    bounds = final_gdf[seam_mask].total_bounds
                    cx = (bounds[0] + bounds[2]) / 2
                    cy = (bounds[1] + bounds[3]) / 2
                    ax.text(cx, cy, f'S{seam_type}{seam_num}', fontsize=4, 
                           ha='center', va='center', color='green')
    except Exception as e:
        pass

plt.savefig("final_stitched_labelled.png", dpi=1000, bbox_inches='tight')
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
        print(f"Seams don't overlap in global space: x={xmin:.0f}-{xmax:.0f}, y={ymin:.0f}-{ymax:.0f}")
        print("Seam bounds:")
        for seam_name, b in zip(overlap_seams, bounds):
            print(f"  {seam_name}: x={b[0]:.0f}-{b[2]:.0f}, y={b[1]:.0f}-{b[3]:.0f}")

    else:
            
            buffer_size = 300 
            xmin -= buffer_size
            xmax += buffer_size
            ymin -= buffer_size
            ymax += buffer_size

            # get window masks using our fast cx/cy columns
            combined_mask = (
                (combined_gdf['cx'] >= xmin) & (combined_gdf['cx'] <= xmax) &
                (combined_gdf['cy'] >= ymin) & (combined_gdf['cy'] <= ymax)
            )
            final_mask = (
                (final_gdf['cx'] >= xmin) & (final_gdf['cx'] <= xmax) &
                (final_gdf['cy'] >= ymin) & (final_gdf['cy'] <= ymax)
            )

            # slice the dataframes to just this viewing window to get accurate local counts
            combined_window = combined_gdf[combined_mask]
            final_window = final_gdf[final_mask]

            # Create a 2x3 grid for our diagnostic plots
            fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
            axes = axes.flatten()

            # --- PLOT 1: Pre-Deduplication (All Overlaid) ---
            ax = axes[0]
            ax.set_facecolor('white')
            for key, gdf in seam_gdfs.items():
                # Slice individual seam to the window for plotting
                gdf_window = gdf[ 
                    (gdf['cx'] >= xmin) & (gdf['cx'] <= xmax) & 
                    (gdf['cy'] >= ymin) & (gdf['cy'] <= ymax) 
                ]
                gdf_window.plot(ax=ax, facecolor='none', edgecolor=colors[key], 
                                linewidth=0.8, alpha=0.7, label=key)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f'PRE-DEDUP: All Seams Overlaid\nTotal Cells in Window: {len(combined_window)}', 
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.set_aspect('equal')

            # --- PLOT 2: Post-Deduplication (Final Landscape) ---
            ax = axes[1]
            ax.set_facecolor('white')
            final_window.plot(ax=ax, facecolor='none', edgecolor='black', 
                              linewidth=0.8, alpha=0.9)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f'POST-DEDUP: Final Landscape\nTotal Cells in Window: {len(final_window)}', 
                         fontsize=11, fontweight='bold')
            ax.set_aspect('equal')

            # --- PLOTS 3 to 6: Individual Seams ---
            for i, (key, gdf) in enumerate(seam_gdfs.items()):
                ax = axes[i + 2]
                ax.set_facecolor('white')
                
                gdf_window = gdf[ 
                    (gdf['cx'] >= xmin) & (gdf['cx'] <= xmax) & 
                    (gdf['cy'] >= ymin) & (gdf['cy'] <= ymax) 
                ]
                
                # Plot the pre-dedup background in faint grey
                combined_window.plot(ax=ax, facecolor='lightgrey', edgecolor='grey', 
                                     linewidth=0.3, alpha=0.5)
                
                # Overlay this specific seam's cells in color
                gdf_window.plot(ax=ax, facecolor='none', edgecolor=colors[key], 
                                linewidth=1.2)
                
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_title(f'Seam {key} Only\nCells in Window: {len(gdf_window)}', fontsize=10)
                ax.set_aspect('equal')

            plt.suptitle('Deduplication Diagnostics: Overlap Region', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('overlap_check_counts.png', dpi=1000, bbox_inches='tight', facecolor='white')
            plt.close()

# print original xenium image for comparison
sdata = sd.read_zarr("/scratch/users/k22026807/masters/project/xenium_output/BC_prime.zarr")
img = sdata.images['morphology_focus']['scale4'].ds['image']
dapi = img.sel(c='DAPI').values
fig, ax = plt.subplots(figsize=(10, 15))
ax.imshow(dapi, cmap='gray', origin='upper')
plt.savefig('original_xenium_dapi.png', dpi=1000, bbox_inches='tight')
plt.close()