#### SEAMS ####

"""
Seam resegmentation — modelled exactly on reseg.py (the working 32-patch script).
The ONLY difference from reseg.py is how the crop window is calculated:
  - Patches  : each crop = one tile (col*patch_w : (col+1)*patch_w)
  - Seams    : each crop = a margin-wide strip centred on a tile boundary

The stitcher handles global placement.

Grid layout (4 cols × 8 rows = 32 patches):
  Horizontal seams : strips centred on each row boundary  → (n_rows-1)*n_cols = 28
  Vertical   seams : strips centred on each col boundary  → n_rows*(n_cols-1) = 24
  Total seam jobs  : 52  (indices 0-51)

Seam index mapping:
  0-27  → horizontal seams (H), indexed row-major
  28-51 → vertical   seams (V), indexed row-major
"""

import os
import sys
import gc
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import anndata as ad
import logging
import json
from PIL import Image
from collections import OrderedDict
from shapely.geometry import Polygon
from shapely.affinity import scale, translate
from skimage.measure import label, regionprops, find_contours, regionprops_table
from skimage.transform import resize
import tensorflow as tf
from dask.distributed import Client
from cellpose import io, models
import spatialdata as sd
from spatialdata import bounding_box_query
from spatialdata.models import Labels2DModel, ShapesModel, PointsModel
import dask.dataframe as dd
from tqdm import tqdm
from spatialdata.transformations import Affine
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from pathlib import Path
from proseg_wrapper import (
    run_proseg_refinement,
    fix_zarr_metadata,
    validate_and_parse_components,
    create_integrated_spatialdata,
    fix_anndata_table,
    load_proseg_components,
    create_visualizations,
    generate_comparison_statistics,
    export_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# Resegmentation_xenium
# ─────────────────────────────────────────────────────────────────────────────

class Resegmentation_xenium:
    """Pipeline for spatial omics segmentation and data integration.

    This class encapsulates preprocessing, segmentation, mask filtering,
    shape/label creation, transcript assignment, and writing results
    back to a SpatialData Zarr file.
    
    Attributes:
        zarr_dir (str): Directory containing the input Zarr file.
        zarr_name (str): Name of the input Zarr file.
        output (str): Path to save preprocessed image output.
        factor_rescale (int): Factor by which to rescale coordinates between 
            segmentation and original resolution.
        image_name (str): Name of the image layer in SpatialData.
        label_name (str): Name for the output label layer.
        shape_name (str): Name for the output shape layer.
        point_name (str): Name of the point layer in SpatialData.
        sdata (sd.SpatialData): Loaded SpatialData object.
        image (PIL.Image): Preprocessed image for segmentation.
        masks (np.ndarray): Segmentation masks from Cellpose.
        flows (tuple): Flow outputs from Cellpose.
        styles (np.ndarray): Style outputs from Cellpose.
        gdf_polygons (gpd.GeoDataFrame): Polygon geometries for segmented cells.
        shapes_model (gpd.GeoDataFrame): SpatialData shapes model.
        labels_model (sd.models.Labels2DModel): SpatialData labels model.
        gdf_points (gpd.GeoDataFrame): Point geometries for transcripts.
        vdata (ad.AnnData): AnnData object containing gene expression matrix.
    """

    def __init__(self, zarr_dir: str,
                 zarr_name: str,
                 output: str,
                 factor_rescale: int,
                 image_name: str,
                 label_name: str,
                 shape_name: str,
                 point_name: str):
        """Initialize Resegmentation pipeline.
        
        Args:
            zarr_dir (str): Directory containing the input Zarr file.
            zarr_name (str): Name of the input Zarr file.
            output (str): Path to save preprocessed image output.
            factor_rescale (int): Rescaling factor between segmentation and 
                original resolution (e.g., 4 means segmentation is 4x smaller).
            image_name (str): Name of the image layer in SpatialData.
            label_name (str): Name for the output label layer.
            shape_name (str): Name for the output shape layer.
            point_name (str): Name of the point layer in SpatialData.
        """
        
        self.zarr_dir = zarr_dir
        self.zarr_name = zarr_name
        self.output = output
        self.factor_rescale = factor_rescale
        self.image_name = image_name
        self.label_name = label_name
        self.shape_name = shape_name
        self.point_name = point_name
        self.sdata = None
        self.image = None
        self.masks = None
        self.flows = None
        self.styles = None
        self.gdf_polygons = None
        self.shapes_model = None
        self.labels_model = None
        self.gdf_points = None
        self.vdata = None
        self.output_folder = "data/seams_masks"
        os.makedirs(self.output_folder, exist_ok=True)

    # ---------------- GPU UTILITIES ---------------- #

    @staticmethod
    def check_gpu():
        """Print Python environment, TensorFlow version, and GPU availability.
        
        Displays diagnostic information about the current Python environment
        and TensorFlow GPU configuration for debugging purposes.
        
        Prints:
            - Python executable path
            - TensorFlow version
            - List of available GPU devices
        """
        print("Python path:", sys.executable)
        print("TensorFlow version:", tf.__version__)
        print("GPU available:", tf.config.list_physical_devices("GPU"))


    @staticmethod
    def clear_gpu_memory():
        """Clear TensorFlow GPU memory and trigger garbage collection.
        
        Clears the TensorFlow Keras session to free GPU memory and runs
        Python's garbage collector to free system memory. Useful between
        processing steps to prevent memory accumulation.
        
        Note:
            Errors during clearing are caught and printed but do not raise
            exceptions to allow continued execution.
        """
        
        try:
            if tf.config.list_physical_devices("GPU"):
                tf.keras.backend.clear_session()
            gc.collect()
        except Exception as e:
            print(f"Error clearing GPU memory: {e}")

    # ---------------- IMAGE PREPROCESS ---------------- #

    def preprocess_image(
        self,
        channel_names: list[str],
        channels_to_use: list[str] = ["Y", "U"],
    ):
        """Load and preprocess Zarr image for segmentation.
        
        Loads image data from SpatialData Zarr file, selects specified channels,
        applies contrast stretching, converts to 8-bit RGB format, and downscales
        for segmentation. The preprocessed image is saved to disk.
        
        Args:
            channel_names (list[str]): List of all channel names in the image 
                (e.g., ['R', 'G', 'B', 'Y', 'U']).
            channels_to_use (list[str], optional): List of channel names to use
                for segmentation. Defaults to ["Y", "U"].
        
        Returns:
            PIL.Image: Preprocessed RGB image ready for segmentation, downscaled
                by `factor_rescale`.
        
        Note:
            - Applies 2nd-98th percentile contrast stretching
            - Converts grayscale or 2-channel images to 3-channel RGB
            - Clears memory after preprocessing
            - Sets `self.image` attribute
        """
        
        zarr_path = os.path.join(self.zarr_dir, self.zarr_name)
        self.sdata = sd.read_zarr(zarr_path)

        img_xr = self.sdata.images[self.image_name]["scale0"].ds["image"]
        data_dask = img_xr.data 
        
        # detect order
        is_channel_first = False
        if hasattr(img_xr, 'dims') and img_xr.dims[0] == 'c':
            is_channel_first = True
            print("Detected Order: Channel First (c, y, x)")

        # setup slice
        step = int(self.factor_rescale)
        channel_indices = [channel_names.index(ch) for ch in channels_to_use]

        # apply slice
        if is_channel_first:
            # slice (C, Y, X) -> channels first, then step Y and X
            data_dask = data_dask[channel_indices, ::step, ::step]
        else:
            # slice (Y, X, C) -> step Y and X, then channels
            data_dask = data_dask[::step, ::step, channel_indices]

        print(f"Loading small chunk with step={step}...")

        # compute
        img = data_dask.compute()

        # transpose in RAM
        if is_channel_first:
            # (C, Y, X) -> (Y, X, C)
            img = img.transpose(1, 2, 0)

        # Contrast stretching
        p2, p98 = np.percentile(img, (2, 98))
        img_stretched = np.clip((img - p2) / (p98 - p2),
                                0, 1).astype(np.float32)

        # Convert to 8-bit
        img_8bit = (img_stretched * 255).astype(np.uint8)

        # Ensure 3-channel RGB
        if img_8bit.ndim == 3 and img_8bit.shape[-1] == 2:
            zero_channel = np.zeros_like(img_8bit[..., :1])
            img_8bit = np.concatenate([img_8bit, zero_channel], axis=-1)

        elif img_8bit.ndim == 3 and img_8bit.shape[-1] == 1:
            zero_channel = np.zeros_like(img_8bit[..., :1])
            img_8bit = np.concatenate(
                [img_8bit, zero_channel, zero_channel], axis=-1)

        elif img_8bit.ndim == 2:
            img_8bit = np.stack([img_8bit] * 3, axis=-1)

        seg_pil = Image.fromarray(img_8bit, mode="RGB")

        y_downscale = int(img_xr.shape[1]/self.factor_rescale)
        x_downscale = int(img_xr.shape[2]/self.factor_rescale)
        downscale = (y_downscale, x_downscale)

        # Resize for segmentation model
        seg_pil.thumbnail(downscale[::-1], Image.Resampling.LANCZOS)

        # Save if requested
        if self.output is not None:
            seg_pil.save(self.output)

        # Clean up memory
        for var_name in ['img_xr', 'img', 'img_8bit', 'zero_channel']:
            if var_name in globals():
                del globals()[var_name]

        self.clear_gpu_memory()

        self.image = seg_pil
        print("Preprocessing finished")

        return seg_pil

    # ---------------- SEGMENTATION ---------------- #

    def run_cellpose(
        self,
        model_type: str = "cyto3",
        gpu: bool = True,
        tile_overlap: float = 0.1,
        diameter: float or None = None,
        flow_threshold: float = 1,
        cellprob_threshold: float = -3,
    ):
        """Run Cellpose segmentation on the preprocessed image.
        
        Loads the preprocessed image and runs Cellpose segmentation model
        to generate cell masks. Saves masks to disk and clears GPU memory.
        
        Args:
            model_type (str, optional): Cellpose model type. Options include
                'cyto', 'cyto2', 'cyto3', 'nuclei'. Defaults to "cyto3".
            gpu (bool, optional): Whether to use GPU for inference. 
                Defaults to True.
            tile_overlap (float, optional): Overlap fraction between tiles
                for large images (0-1). Defaults to 0.1.
            diameter (float or None, optional): Expected cell diameter in pixels.
                If None, uses automatic diameter detection. Defaults to None.
            flow_threshold (float, optional): Flow error threshold for mask
                reconstruction. Higher values = more lenient. Defaults to 1.
            cellprob_threshold (float, optional): Cell probability threshold.
                Lower values = more permissive segmentation. Defaults to -3.
        
        Returns:
            tuple: Contains three elements:
                - masks (np.ndarray): Labeled segmentation masks (H, W).
                - flows (tuple): Flow field outputs from Cellpose.
                - styles (np.ndarray): Style vector outputs from Cellpose.
        
        Note:
            - Sets `self.masks`, `self.flows`, and `self.styles` attributes
            - Saves mask image as "mask.png"
            - Clears GPU memory after segmentation
        """
        
        img = io.imread(self.output)
        model = models.CellposeModel(model_type=model_type, gpu=gpu)

        self.masks, self.flows, self.styles = model.eval(
            img,
            diameter=diameter,
            tile_overlap=tile_overlap,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            batch_size=4
        )
        print("Cellpose segmentation finished")
        img = Image.fromarray(self.masks)
        img.save("mask.png")

        self.clear_gpu_memory()

        return self.masks, self.flows, self.styles

    # ---------------- MASKS & SHAPES ---------------- #

    @staticmethod
    def filter_cell_by_regionprops(
            seg_masks,
            max_eccentricity=None,
            min_area=0,
            min_absolute_area=100,
            max_area=None,
            min_solidity=None,
            max_solidity=None,
            min_extent=None,
            max_extent=None,
            min_compactness=None,
            max_convexity_deficit=None,
            max_perimeter_area_ratio=None,
            area_std_filter=None,
            save_debug=False,
            verbose=True):
        """Filter segmented cell masks by multiple morphological criteria.
        
        Analyzes region properties of segmented masks and filters cells based on
        shape metrics including area, eccentricity, solidity, extent, compactness,
        and convexity deficit. Provides detailed statistics and rejection reasons 
        when verbose mode is enabled.
        
        Args:
            seg_masks (np.ndarray): Input segmentation masks with integer labels.
            max_eccentricity (float, optional): Maximum eccentricity (elongation)
                threshold. Range: 0-1, where 0=circle, 1=line. Typical: 0.95.
                Defaults to None (no filtering).
            min_area (str or float, optional): Minimum area filter. Options:
                - 'median': Use median(areas) as threshold
                - 'median_div2': Use median(areas)/2 as threshold
                - 0-100: Use percentile of area distribution
                - Absolute value: Use as pixel count threshold
                Defaults to 'median_div2'.
            min_absolute_area (int, optional): Hard minimum area to remove 
                tiny artifacts regardless of distribution. Defaults to 50.
            max_area (float, optional): Maximum area threshold in pixels.
                Defaults to None (no filtering).
            min_solidity (float, optional): Minimum solidity (area/convex_area).
                Range: 0-1. Filters highly concave cells. Typical: 0.7-0.9.
                Defaults to None (no filtering).
            max_solidity (float, optional): Maximum solidity. Filters perfectly
                convex cells (unrealistic). Typical: 0.98-0.995. 
                Defaults to None (no filtering).
            min_extent (float, optional): Minimum extent (area/bounding_box_area).
                Range: 0-1. Filters sparse cells. Typical: 0.3-0.5.
                Defaults to None (no filtering).
            max_extent (float, optional): Maximum extent. Filters perfectly
                rectangular cells. Typical: 0.9-0.95.
                Defaults to None (no filtering).
            min_compactness (float, optional): Minimum compactness (Polsby-Popper).
                Compactness = 4π * area / perimeter². Range: 0-1, where 1=perfect circle.
                Filters irregular shapes. Typical: 0.5-0.7 for cells, half-moons ~0.3-0.5.
                Defaults to None (no filtering).
            max_convexity_deficit (float, optional): Maximum convexity deficit.
                Deficit = 1 - solidity = (convex_area - area) / convex_area.
                Range: 0-1, where 0=perfectly convex. Filters concave cells like half-moons.
                Typical: 0.10-0.20, half-moons ~0.25-0.50.
                Defaults to None (no filtering).
            max_perimeter_area_ratio (float, optional): Maximum perimeter/sqrt(area) ratio.
                Normalized perimeter measure. Filters cells with excessive perimeter.
                Typical cells: 3.5-5.0, irregular cells: >6.0.
                Defaults to None (no filtering).
            area_std_filter (float, optional): Remove area outliers beyond N
                standard deviations from mean. E.g., 3.0 removes outliers
                beyond 3σ. Defaults to None (no filtering).
            save_debug (bool, optional): Whether to save debug image 
                "cleaned_mask.png". Defaults to False.
            verbose (bool, optional): Whether to print filtering statistics
                and rejection breakdown. Defaults to True.
        
        Returns:
            np.ndarray: Cleaned and relabeled mask array with consecutive 
                integer labels starting from 1. Zero indicates background.
        
        Shape Metrics Explained:
            Solidity = area / convex_area
                - ~1.0: Very smooth, convex (circle, ellipse) - may be unrealistic
                - 0.7-0.9: Slightly irregular - typical realistic cells
                - <0.7: Highly concave/irregular - likely artifacts
            
            Extent = area / bounding_box_area
                - ~1.0: Fills bounding box (square, rectangle)
                - 0.5-0.8: Typical cell shapes
                - <0.3: Very sparse/thin - likely artifacts
            
            Eccentricity = elongation measure
                - 0: Perfect circle
                - 0.95: Very elongated
                - >0.95: Extremely elongated - may be artifacts
            
            Compactness = 4π * area / perimeter² (Polsby-Popper)
                - 1.0: Perfect circle
                - 0.6-0.8: Typical cells
                - <0.5: Irregular/elongated shapes (including half-moons)
            
            Convexity Deficit = 1 - solidity
                - 0: Perfectly convex
                - 0.05-0.15: Slight irregularity (normal cells)
                - >0.20: Significant concavity (half-moon cells, artifacts)
            
            Perimeter/Area Ratio = perimeter / sqrt(area)
                - 3.5-5.0: Typical cells
                - >6.0: Irregular perimeter (fragmented, concave cells)
        
        Note:
            - Uses vectorized operations for efficient relabeling
            - Prints detailed statistics when verbose=True
            - Returns empty mask if no regions pass filtering
            - New metrics (compactness, convexity deficit) are especially effective
              for removing half-moon/crescent-shaped cells
        """
        
        if verbose:
            print("Labeling connected components")

        labeled = label(seg_masks)

        if verbose:
            print(f"Extracting properties for {labeled.max()} regions")

        # Extract properties
        properties = ['label', 'area', 'eccentricity', 'perimeter']
        if min_solidity is not None or max_solidity is not None or max_convexity_deficit is not None:
            properties.append('solidity')
        if min_extent is not None or max_extent is not None:
            properties.append('extent')

        props = regionprops_table(labeled, properties=properties)

        n_regions = len(props['label'])
        if n_regions == 0:
            if verbose:
                print("No regions found")
            return np.zeros_like(seg_masks, dtype=np.int32)

        areas = props['area']
        perimeters = props['perimeter']

        # ========== CALCULATE NEW METRICS ==========

        # 1. Compactness (Polsby-Popper): 4π * area / perimeter²
        # Circle = 1.0, irregular shapes < 0.5
        # Add epsilon to avoid division by zero
        compactness = (4 * np.pi * areas) / (perimeters ** 2 + 1e-10)

        # 2. Convexity deficit: 1 - solidity
        # How much area is "missing" compared to convex hull
        convexity_deficit = None
        if 'solidity' in props:
            solidity = props['solidity']
            convexity_deficit = 1 - solidity

        # 3. Perimeter-to-area ratio (normalized by sqrt)
        # Normalized measure of perimeter complexity
        perimeter_area_ratio = perimeters / (np.sqrt(areas) + 1e-10)

        # Determine minimum area threshold
        if isinstance(min_area, str) and min_area == 'median':
            min_area_value = np.median(areas)
            threshold_type = "median"
        elif isinstance(min_area, str) and min_area == 'median_div2':
            min_area_value = np.median(areas)/2
            threshold_type = "median_div2"
        elif isinstance(min_area, (int, float)) and 0 <= min_area <= 100:
            min_area_value = np.percentile(areas, min_area)
            threshold_type = f"{min_area}th percentile"
        else:
            min_area_value = min_area
            threshold_type = "absolute"

        if verbose:

            print("SHAPE STATISTICS")
            print("Area:")
            print(f"Range: {areas.min():.1f} - {areas.max():.1f}")
            print(
                f"Mean: {areas.mean():.1f}, Median: {np.median(areas):.1f}, Std: {areas.std():.1f}")
            print("Eccentricity:")
            ecc = props['eccentricity']
            print(f"Range: {ecc.min():.3f} - {ecc.max():.3f}")
            print(f"Mean: {ecc.mean():.3f}, Median: {np.median(ecc):.3f}")
            print("Compactness (1.0=circle, <0.5=irregular):")
            print(
                f"Range: {compactness.min():.3f} - {compactness.max():.3f}")
            print(
                f"Mean: {compactness.mean():.3f}, Median: {np.median(compactness):.3f}")
            print(f"25th percentile: {np.percentile(compactness, 25):.3f}")
            print(f"10th percentile: {np.percentile(compactness, 10):.3f}")

            if 'solidity' in props:
                print("Solidity (convexity):")
                print(f"Range: {solidity.min():.3f} - {solidity.max():.3f}")
                print(
                    f"Mean: {solidity.mean():.3f}, Median: {np.median(solidity):.3f}")
                print("Convexity Deficit (0=convex, high=concave):")
                print(
                    f"Range: {convexity_deficit.min():.3f} - {convexity_deficit.max():.3f}")
                print(
                    f"Mean: {convexity_deficit.mean():.3f}, Median: {np.median(convexity_deficit):.3f}")
                print(
                    f"75th percentile: {np.percentile(convexity_deficit, 75):.3f}")
                print(
                    f"90th percentile: {np.percentile(convexity_deficit, 90):.3f}")

            print("Perimeter/√Area Ratio (typical: 3.5-5.0):")
            print(
                f"Range: {perimeter_area_ratio.min():.3f} - {perimeter_area_ratio.max():.3f}")
            print(
                f"Mean: {perimeter_area_ratio.mean():.3f}, Median: {np.median(perimeter_area_ratio):.3f}")
            print(
                f"75th percentile: {np.percentile(perimeter_area_ratio, 75):.3f}")
            print(
                f"90th percentile: {np.percentile(perimeter_area_ratio, 90):.3f}")

            if 'extent' in props:
                extent = props['extent']
                print("Extent (bbox filling):")
                print(f"Range: {extent.min():.3f} - {extent.max():.3f}")
                print(
                    f"Mean: {extent.mean():.3f}, Median: {np.median(extent):.3f}")

            print(f"Using {threshold_type} area threshold: {min_area_value:.1f}")

        # Build filter mask
        valid_mask = np.ones(n_regions, dtype=bool)
        rejection_reasons = {
            'tiny_artifacts': 0,
            'too_small': 0,
            'too_large': 0,
            'area_outliers': 0,
            'too_eccentric': 0,
            'too_convex': 0,
            'too_concave': 0,
            'too_sparse': 0,
            'too_rectangular': 0,
            'low_compactness': 0,
            'high_convexity_deficit': 0,
            'irregular_perimeter': 0
        }

        # 1. Remove tiny artifacts
        if min_absolute_area > 0:
            artifact_mask = areas >= min_absolute_area
            rejection_reasons['tiny_artifacts'] = (~artifact_mask).sum()
            valid_mask &= artifact_mask

        # 2. Apply relative area threshold
        small_mask = areas >= min_area_value
        rejection_reasons['too_small'] = (~small_mask & valid_mask).sum()
        valid_mask &= small_mask

        # 3. Remove very large regions
        if max_area is not None:
            large_mask = areas <= max_area
            rejection_reasons['too_large'] = (~large_mask & valid_mask).sum()
            valid_mask &= large_mask

        # 4. Remove area outliers
        if area_std_filter is not None:
            mean_area = areas.mean()
            std_area = areas.std()
            lower_bound = mean_area - area_std_filter * std_area
            upper_bound = mean_area + area_std_filter * std_area
            outlier_mask = (areas >= lower_bound) & (areas <= upper_bound)
            rejection_reasons['area_outliers'] = (
                ~outlier_mask & valid_mask).sum()
            valid_mask &= outlier_mask
            if verbose:
                print(
                    f"Area outlier bounds ({area_std_filter}σ): {lower_bound:.1f} - {upper_bound:.1f}")

        # 5. Apply eccentricity filter
        if max_eccentricity is not None:
            eccentric_mask = props['eccentricity'] <= max_eccentricity
            rejection_reasons['too_eccentric'] = (
                ~eccentric_mask & valid_mask).sum()
            valid_mask &= eccentric_mask
            if verbose:
                print(f"Max eccentricity threshold: {max_eccentricity:.3f}")

        # 6. Filter by solidity (convexity)
        if 'solidity' in props:
            # Remove too convex cells (unrealistically smooth)
            if max_solidity is not None:
                convex_mask = solidity <= max_solidity
                rejection_reasons['too_convex'] = (
                    ~convex_mask & valid_mask).sum()
                valid_mask &= convex_mask
                if verbose:
                    print(f"Max solidity (remove convex): {max_solidity:.3f}")

            # Remove too concave cells (artifacts)
            if min_solidity is not None:
                concave_mask = solidity >= min_solidity
                rejection_reasons['too_concave'] = (
                    ~concave_mask & valid_mask).sum()
                valid_mask &= concave_mask
                if verbose:
                    print(f"Min solidity (remove concave): {min_solidity:.3f}")

        # 7. Filter by extent (bounding box filling)
        if 'extent' in props:
            extent = props['extent']

            # Remove sparse/thin cells
            if min_extent is not None:
                sparse_mask = extent >= min_extent
                rejection_reasons['too_sparse'] = (
                    ~sparse_mask & valid_mask).sum()
                valid_mask &= sparse_mask
                if verbose:
                    print(f"Min extent (remove sparse): {min_extent:.3f}")

            # Remove perfectly rectangular cells
            if max_extent is not None:
                rect_mask = extent <= max_extent
                rejection_reasons['too_rectangular'] = (
                    ~rect_mask & valid_mask).sum()
                valid_mask &= rect_mask
                if verbose:
                    print(f"Max extent (remove rectangular): {max_extent:.3f}")

        # 8. Filter by compactness (CRITICAL for half-moon cells)
        if min_compactness is not None:
            compact_mask = compactness >= min_compactness
            rejection_reasons['low_compactness'] = (
                ~compact_mask & valid_mask).sum()
            valid_mask &= compact_mask
            if verbose:
                print(f"Min compactness threshold: {min_compactness:.3f}")

        # 9. Filter by convexity deficit (CRITICAL for half-moon cells)
        if max_convexity_deficit is not None and convexity_deficit is not None:
            deficit_mask = convexity_deficit <= max_convexity_deficit
            rejection_reasons['high_convexity_deficit'] = (
                ~deficit_mask & valid_mask).sum()
            valid_mask &= deficit_mask
            if verbose:
                print(
                    f"Max convexity deficit threshold: {max_convexity_deficit:.3f}")

        # 10. Filter by perimeter-area ratio
        if max_perimeter_area_ratio is not None:
            perimeter_mask = perimeter_area_ratio <= max_perimeter_area_ratio
            rejection_reasons['irregular_perimeter'] = (
                ~perimeter_mask & valid_mask).sum()
            valid_mask &= perimeter_mask
            if verbose:
                print(
                    f"Max perimeter/√area ratio: {max_perimeter_area_ratio:.3f}")

        n_valid = valid_mask.sum()

        if verbose:

            print(
                f"FILTERING RESULTS: {n_valid}/{n_regions} regions passed ({100*n_valid/n_regions:.1f}%)")

            if n_valid > 0:
                valid_areas = areas[valid_mask]
                valid_compactness = compactness[valid_mask]
                valid_perimeter_ratio = perimeter_area_ratio[valid_mask]

                print("Valid regions statistics:")
                print(
                    f"Area range: {valid_areas.min():.1f} - {valid_areas.max():.1f}")
                print(f"Mean area: {valid_areas.mean():.1f}")
                print(
                    f"Compactness range: {valid_compactness.min():.3f} - {valid_compactness.max():.3f}")
                print(
                    f"Perimeter ratio range: {valid_perimeter_ratio.min():.3f} - {valid_perimeter_ratio.max():.3f}")

                if 'solidity' in props:
                    valid_solidity = solidity[valid_mask]
                    valid_deficit = convexity_deficit[valid_mask]
                    print(
                        f"Solidity range: {valid_solidity.min():.3f} - {valid_solidity.max():.3f}")
                    print(
                        f"Convexity deficit range: {valid_deficit.min():.3f} - {valid_deficit.max():.3f}")

                if 'extent' in props:
                    valid_extent = extent[valid_mask]
                    print(
                        f"Extent range: {valid_extent.min():.3f} - {valid_extent.max():.3f}")

            print("Rejection breakdown:")
            total_rejected = n_regions - n_valid
            print(f"Total rejected: {total_rejected}")
            for reason, count in rejection_reasons.items():
                if count > 0:
                    pct = 100 * count / n_regions
                    print(
                        f"{reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")

        if n_valid == 0:
            if verbose:
                print("WARNING: No regions passed filtering criteria!")
            return np.zeros_like(seg_masks, dtype=np.int32)

        # Vectorized relabeling via lookup table
        valid_labels = props['label'][valid_mask]
        lut = np.zeros(labeled.max() + 1, dtype=np.int32)
        lut[valid_labels] = np.arange(1, n_valid + 1, dtype=np.int32)
        cleaned_mask = lut[labeled]

        if save_debug:
            from PIL import Image
            Image.fromarray(cleaned_mask).save("cleaned_mask.png")
            if verbose:
                print("Saved debug image: cleaned_mask.png")

        return cleaned_mask

    @staticmethod
    def masks_to_polygons(seg_masks, factor_rescale):
        """Convert segmentation masks into scaled polygons.
        
        Extracts contours from each labeled region in the segmentation mask,
        converts them to Shapely polygons, and rescales coordinates to match
        the original image resolution.
        
        Args:
            seg_masks (np.ndarray): Labeled segmentation masks where each 
                unique integer represents a different cell.
            factor_rescale (int): Factor by which to upscale polygon coordinates.
                E.g., if masks are 4x smaller than original, use factor_rescale=4.
        
        Returns:
            list[shapely.geometry.Polygon]: List of upscaled polygon geometries,
                one per valid segmented region. Invalid or empty polygons are
                excluded.
        
        Note:
            - Finds the longest contour for each region
            - Ensures contours are closed (first point = last point)
            - Buffers polygons by 0 to fix any self-intersections
            - If factor_rescale=0, no scaling is applied
        """
        
        upscale_polygons = []

        for region in regionprops(seg_masks):
            mask = (seg_masks == region.label).astype(int)
            contours = find_contours(np.array(mask))
            if contours:
                contour = max(contours, key=lambda x: len(x))

                # ensure the contour is closed (first point = last point)
                if not ((contour[0] == contour[-1]).all()):
                    contour = np.vstack([contour, contour[0]])
                poly = Polygon(contour[:, [1, 0]]).buffer(0)

                # only keep polygons that are valid and not empty
                if poly.is_valid and not poly.is_empty:
                    upscale_polygons.append(poly)

        h, w = seg_masks.shape

        if factor_rescale != 0:
            upscale_polygons = [scale(poly, xfact=factor_rescale, yfact=factor_rescale, origin=(
                0, 0)) for poly in upscale_polygons]

        return upscale_polygons


    @staticmethod
    def mirror_y0(geom):
        """Mirror a polygon geometry across the x-axis (y=0).
        
        Flips polygon coordinates vertically by negating y-values. Used to
        correct coordinate system orientation differences between image and
        spatial coordinate systems.
        
        Args:
            geom (shapely.geometry.Polygon or other): Input geometry to mirror.
        
        Returns:
            shapely.geometry: Mirrored geometry of the same type as input.
                If input is not a Polygon, returns input unchanged.
        
        Example:
            Point (x, y) becomes (x, -y)
        """
        
        return type(geom)([
            (x, -y) for x, y in geom.exterior.coords
        ]) if geom.geom_type == "Polygon" else geom


    def process_masks_to_shapes(self):
        """Filter masks, convert to polygons, and create shapes model.
        
        Applies morphological filtering to segmentation masks, converts filtered
        masks to polygon geometries, mirrors coordinates, and creates a 
        SpatialData-compatible shapes model with cell IDs.
        
        Returns:
            tuple: Contains three elements:
                - masks (np.ndarray): Filtered and relabeled segmentation masks.
                - gdf_polygons (gpd.GeoDataFrame): GeoDataFrame with cell_id and
                  geometry columns.
                - shapes_model (gpd.GeoDataFrame): SpatialData shapes model with
                  region names and geometries.
        
        Raises:
            RuntimeError: If no valid polygons are extracted from masks after
                filtering.
        
        Note:
            - Filters masks using `filter_cell_by_regionprops()`
            - Converts to polygons and upscales by `factor_rescale`
            - Mirrors y-coordinates and shifts to ensure all y >= 0
            - Sets `self.masks`, `self.gdf_polygons`, and `self.shapes_model`
        
        TODO:
            Optimize this function for better performance.
        """
        
        # Filter masks
        self.masks = self.filter_cell_by_regionprops(self.masks)

        # Convert masks → polygons
        polygons = self.masks_to_polygons(self.masks, self.factor_rescale)

        if not polygons:
            raise RuntimeError(
                "No polygons extracted from mask — check your segmentation.")

        # Shapes
        shapes_df = gpd.GeoDataFrame({
            "geometry": polygons,
            "cell_id": [f"cell_{i+1}" for i in range(len(polygons))]
        })
        shapes_df.set_geometry("geometry", inplace=True)
        self.shapes_model = ShapesModel.parse(shapes_df)

        # Apply to GeoDataFrame
        self.shapes_model["geometry"] = self.shapes_model["geometry"].apply(
            self.mirror_y0)

        miny = self.shapes_model.total_bounds[1]
        # If miny < 0, shift upward
        if miny < 0:
            self.shapes_model["geometry"] = self.shapes_model["geometry"].apply(
                lambda g: translate(g, xoff=0, yoff=-miny))

        self.gdf_polygons = self.shapes_model[["cell_id", "geometry"]]

        return self.masks, self.gdf_polygons, self.shapes_model

    # ---------------- LABELS ---------------- #

    def process_labels(self):
        """Upscale masks and create Labels2DModel.
        
        Upscales the filtered segmentation masks to the original image resolution
        and creates a SpatialData-compatible Labels2DModel for visualization and
        further analysis.
        
        Returns:
            tuple: Contains two elements:
                - masks (np.ndarray): Original filtered masks (not upscaled).
                - labels_model (sd.models.Labels2DModel): Upscaled labels model
                  with name set to `self.label_name`.
        
        Note:
            - Uses nearest-neighbor interpolation (order=0) to preserve integer labels
            - Upscales by `factor_rescale` to match original image dimensions
            - Sets `self.labels_model` attribute with specified label name
        """
        rescale_size = tuple(
            map(lambda x: x * self.factor_rescale, self.masks.shape))
        masks_upscale = resize(
            self.masks,
            rescale_size,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.int32)

        self.labels_model = Labels2DModel.parse(
            data=np.squeeze(masks_upscale),
            dims=("y", "x"),
        )
        self.labels_model.name = self.label_name

        return self.masks, self.labels_model

    # ---------------- POINTS & TABLES ---------------- #

    def process_points_and_tables(self):
        """Assign transcript points to segmented cells and build AnnData table.
        
        Loads transcript point data from SpatialData, applies coordinate 
        transformations, performs spatial join to assign transcripts to cells,
        builds a gene expression matrix, and creates an AnnData object.
        
        Returns:
            tuple: Contains two elements:
                - points_model (dd.DataFrame): Dask DataFrame of points with
                  transformed coordinates and cell assignments.
                - vdata (ad.AnnData): AnnData object containing:
                    - X: Gene expression count matrix (cells × genes)
                    - obs: Cell metadata with cell IDs
                    - var: Gene metadata with gene names
        
        Note:
            - Concatenates all point layers from SpatialData
            - Applies global transformation and FOV shift corrections
            - Uses spatial join with polygon geometries to assign transcripts
            - Creates cross-tabulated expression matrix via pd.crosstab
            - Clears memory after processing
            - Sets `self.gdf_points` and `self.vdata` attributes
        
        Warning:
            Assumes FOV size of 4256 pixels for shift correction. May need
            adjustment for different imaging systems.
        """

        # Xenium pixel size
        pixel_size = 0.2125
        # Load points
        self.points_model = dd.concat(
            [self.sdata.points[x] for x in self.sdata.points.keys()], axis=0, ignore_index=False)
        points_df = self.points_model[[
            'x', 'y', 'feature_name', 'z', 'qv']].compute()
        points_df = points_df[points_df['qv']>=20]

        self.gdf_points = gpd.GeoDataFrame(
            points_df[['x', 'y', 'feature_name', 'z']],
            geometry=gpd.points_from_xy(x=points_df["x"], y=points_df["y"]),
            crs=self.gdf_polygons.crs)

        self.gdf_points['geometry'] = self.gdf_points['geometry'].apply(
            lambda geom: scale(geom, xfact=(1/pixel_size), yfact=-(1/pixel_size), origin=(0, 0)))

        minx, miny, maxx, maxy = self.gdf_points.total_bounds
        minx_new, miny_new, maxx_new, maxy_new = self.gdf_points.total_bounds
        self.gdf_points['geometry'] = self.gdf_points['geometry'].apply(
            lambda geom: translate(geom, xoff=-minx_new, yoff=-miny_new))
        self.gdf_points['x_updated'] = self.gdf_points['geometry'].apply(
            lambda geom: geom.x)
        self.gdf_points['y_updated'] = self.gdf_points['geometry'].apply(
            lambda geom: geom.y)

        # --- FIX: Re-number the rows so the index is 100% unique ---
        self.gdf_points = self.gdf_points.reset_index(drop=True)

        # Spatial join (assign transcripts to polygons)
        joined = gpd.sjoin(
            self.gdf_points,
            self.gdf_polygons[["cell_id", "geometry"]],
            how="left",
            predicate="within",
        )
        
        joined = joined[~joined.index.duplicated(keep='first')]

        self.gdf_points["cell_id"] = joined["cell_id"].values
        # Build expression matrix
        all_genes = self.gdf_points["feature_name"].unique().tolist()

        # Create a cross-tabulation of cell_id vs target
        cell_gene_matrix = pd.crosstab(
            self.gdf_points["cell_id"], self.gdf_points["feature_name"])

        # Ensure all genes appear as columns (even if missing in some cells)
        cell_gene_matrix = cell_gene_matrix.reindex(
            columns=all_genes, fill_value=0)

        # Convert to dict of OrderedDicts
        cell_gene_counts = {str(cell): OrderedDict(cell_gene_matrix.loc[cell].to_dict(
        )) for cell in tqdm(cell_gene_matrix.index, total=len(cell_gene_matrix.index))}
        df = pd.DataFrame.from_dict(
            cell_gene_counts, orient="index", columns=all_genes)
        df = df.reindex(columns=all_genes, fill_value=0)

        # Convert to AnnData
        self.vdata = ad.AnnData(X=df.values)
        self.vdata.obs_names = df.index
        self.vdata.var_names = df.columns
        self.vdata.obs["cells"] = self.vdata.obs_names
        self.vdata.var["genes"] = self.vdata.var_names

        # Clean up memory
        for var in (points_df,):
            try:
                del var
            except Exception:
                pass
        self.clear_gpu_memory()
        points_model_updated = dd.from_pandas(pd.merge(self.gdf_points[['x', 'y', 'feature_name', 'z', 'cell_id', 'x_updated', 'y_updated']], self.points_model[[
            'x', 'y', 'z', 'codeword_category', 'is_gene', 'transcript_id', 'codeword_index', 'fov_name', 'qv']].compute(),
            how="left",
            on=['x', 'y', 'z']),
            npartitions=self.points_model.npartitions)

        self.points_model = PointsModel.parse(points_model_updated)

        return self.points_model, self.vdata
    

    def process_xenium_adata(self, gene_list_path="gene_list_10X.csv"):
        """
        Process Xenium AnnData: compute QC metrics, remove controls, add missing genes
        
        Parameters:
        -----------
        adata : AnnData
            Input AnnData object
        gene_list_path : str
            Path to reference gene list CSV
        
        Returns:
        --------
        adata : AnnData
            Processed AnnData object
        """

        # Define control keywords
        keywords = {
            'control_probe': 'NegControlProbe',
            'genomic_control': 'Intergenic_Region',
            'control_codeword': 'NegControlCodeword',
            'unassigned_codeword': 'UnassignedCodeword',
            'deprecated_codeword': 'DeprecatedCodeword'
        }

        vdata = self.new_sdata.tables['table']
        print('vdata')
        print(vdata)
        cols_to_copy = ['TMA_ID', 'core_ID', 'metabric_ID',
                        'MB_Number', 'metallomics_ID'] #'ST_ID'
        available_cols = [c for c in cols_to_copy if c in self.sdata.tables['table'].obs.columns]
        
        if available_cols:
            print(f"Copying metadata columns: {available_cols}")
            info_core = self.sdata.tables['table'].obs[available_cols].iloc[0]
            for k in info_core.index:
                vdata.uns[k] = info_core.loc[k]
        else:
            print("Warning: Requested metadata not found. Skipping.")

        # obs['transcript_counts'] = np.array(new_sdata.tables['table'].X.sum(axis=1)).flatten()
        obs = vdata.obs

        # Filter shapes
        valid_cells = vdata.obs_names
        gdf_cells = self.new_sdata.shapes[self.shape_name].copy()
        gdf_cells_filtered = gdf_cells[gdf_cells['cell_id'].isin(valid_cells)]
        self.new_sdata.shapes[self.shape_name] = gdf_cells_filtered

        # Compute and attach cell area
        cells = self.new_sdata.shapes[self.shape_name]
        obs = vdata.obs.copy()
        obs['cell_area'] = cells.geometry.area.values

        if 'TMA_ID' in vdata.uns:
            obs['TMA_ID'] = vdata.uns['TMA_ID']
            
        if 'metabric_ID' in vdata.uns:
            obs['metabric_ID'] = vdata.uns['metabric_ID']
            
        centroids = cells.geometry.centroid
        vdata.obsm['spatial'] = np.column_stack([centroids.x, centroids.y])

        vdata.obs = obs

        print("=" * 60)
        print("PROCESSING XENIUM ANNDATA")
        print("=" * 60)
        print(f"Original shape: {vdata.shape}")

        # 1. Compute QC metrics
        print("1. Computing QC metrics")

        for metric_name, keyword in keywords.items():
            mask = vdata.var_names.str.contains(keyword, na=False, regex=False)
            counts = np.array(vdata[:, mask].X.sum(axis=1)).flatten()
            vdata.obs[f'{metric_name}_counts'] = counts
            print(f" {metric_name}_counts: {mask.sum()} features")

        # Identify all control features
        all_control_mask = pd.Series(False, index=vdata.var_names)
        for keyword in keywords.values():
            all_control_mask |= vdata.var_names.str.contains(
                keyword, na=False, regex=False)

        # Transcript counts (non-control genes)
        transcript_mask = ~all_control_mask
        vdata.obs['transcript_counts'] = np.array(
            vdata[:, transcript_mask].X.sum(axis=1)
        ).flatten()
        print(f" transcript_counts: {transcript_mask.sum()} features")

        # Total counts
        vdata.obs['total_counts'] = np.array(vdata.X.sum(axis=1)).flatten()

        # Summary stats
        print("   QC Metrics Summary:")
        qc_cols = ['control_probe_counts', 'genomic_control_counts',
                   'control_codeword_counts', 'unassigned_codeword_counts',
                   'deprecated_codeword_counts', 'transcript_counts', 'total_counts']
        print(vdata.obs[qc_cols].describe().loc[['mean', 'std', 'min', 'max']])

        # 2. Remove control features
        print("2. Removing control features")
        print(f" Control features: {all_control_mask.sum()}")
        print(f" Regular genes: {transcript_mask.sum()}")

        vdata = vdata[:, transcript_mask].copy()
        print(f" After removal: {vdata.shape}")

        # 3. Add missing genes
        print("3. Adding missing genes from reference")

        gene_list = pd.read_csv(gene_list_path)['gene_name'].tolist()
        print(f" Reference list: {len(gene_list)} genes")

        current_genes = set(vdata.var_names)
        missing_genes = [
            gene for gene in gene_list if gene not in current_genes]
        print(f" Missing genes: {len(missing_genes)}")

        if len(missing_genes) > 0:
            # Store original uns and other metadata
            original_uns = vdata.uns.copy()
            original_obsm = vdata.obsm.copy() if vdata.obsm.keys() else {}
            original_obsp = vdata.obsp.copy() if vdata.obsp.keys() else {}
            original_varm = vdata.varm.copy() if vdata.varm.keys() else {}
            original_varp = vdata.varp.copy() if vdata.varp.keys() else {}
            original_layers = {
                k: v for k, v in vdata.layers.items()} if vdata.layers.keys() else {}

            # Create zero matrix for missing genes
            missing_matrix = np.zeros((vdata.n_obs, len(missing_genes)))

            # Match the format of existing X matrix (sparse or dense)
            if hasattr(vdata.X, 'toarray'):  # If sparse
                missing_matrix = csr_matrix(missing_matrix)

             # Create var DataFrame matching existing structure
            missing_var = pd.DataFrame(index=missing_genes)
            # Copy column structure from existing var, fill with appropriate defaults
            for col in vdata.var.columns:
                if vdata.var[col].dtype == 'object' or vdata.var[col].dtype == 'string':
                    missing_var[col] = ''  # Empty string instead of NaN
                elif vdata.var[col].dtype == 'bool':
                    missing_var[col] = False
                else:
                    missing_var[col] = 0  # Numeric columns get 0

            missing_adata = sc.AnnData(
                X=missing_matrix,
                obs=vdata.obs.copy(),
                var=pd.DataFrame(index=missing_genes)
            )

            # Concatenate along gene axis (axis=1)
            vdata = sc.concat([vdata, missing_adata], axis=1,
                              join='outer', merge='same')

            # Clean up any remaining NaN values in var
            for col in vdata.var.columns:
                if vdata.var[col].dtype == 'object' or vdata.var[col].dtype == 'string':
                    vdata.var[col] = vdata.var[col].fillna('')
                elif vdata.var[col].dtype == 'bool':
                    vdata.var[col] = vdata.var[col].fillna(False)

            # Restore metadata
            vdata.uns = original_uns
            if original_obsm:
                vdata.obsm = original_obsm
            if original_obsp:
                vdata.obsp = original_obsp
            if original_varm:
                vdata.varm = original_varm
            if original_varp:
                vdata.varp = original_varp
            if original_layers:
                for key, value in original_layers.items():
                    # For layers, we need to add zeros for missing genes
                    if hasattr(value, 'toarray'):  # sparse
                        missing_layer = csr_matrix(
                            np.zeros((vdata.n_obs, len(missing_genes))))
                        vdata.layers[key] = sp.hstack([value, missing_layer])
                    else:  # dense
                        missing_layer = np.zeros(
                            (vdata.n_obs, len(missing_genes)))
                        vdata.layers[key] = np.hstack([value, missing_layer])

            print(f"After adding: {vdata.shape}")

        # 4. Reorder to match reference
        print("4. Reordering genes to match reference")
        genes_in_order = [
            gene for gene in gene_list if gene in vdata.var_names]

        if len(genes_in_order) != len(gene_list):
            print(
                f" WARNING: {len(gene_list) - len(genes_in_order)} genes from reference not found!")

        vdata = vdata[:, genes_in_order].copy()
        print("PROCESSING COMPLETE")
        print(f"Final shape: {vdata.shape}")
        print(f"Cells: {vdata.n_obs}")
        print(f"Genes: {vdata.n_vars}")

        self.new_sdata.tables['table'] = vdata

        return vdata
    

    def run_proseg(self, 
                   samples=1000, 
                   voxel_size=0.5, 
                   voxel_layers=2, 
                   nuclear_reassignment_prob=0.2, 
                   diffusion_probability=0.2, 
                   num_threads=12,):
        """
        Run complete Proseg refinement pipeline for Xenium spatial transcriptomics data.
        
        This method:
        1. Loads SpatialData and extracts components
        2. Runs Proseg refinement
        3. Fixes metadata and validates components
        4. Creates integrated SpatialData with original and refined data
        5. Generates visualizations and comparison statistics
        6. Exports results in multiple formats
        7. Processes final AnnData with Xenium-specific QC
        
        Returns
        -------
        tuple
            (integrated_sdata, adata_refined, comparison, output_files, exported_files)
        """
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.log = logging.getLogger(__name__)
        self.log.info("Starting Proseg Refinement Pipeline for Xenium Data")

        # Extract components from existing SpatialData (already loaded in self.new_sdata)
        self.log.info("Extracting components from SpatialData")
        sdata = self.new_sdata
        
        # Get transcripts from points
        transcripts_df = sdata.points[self.point_name].compute()

        # Define output paths
        output_path = Path(f"proseg_output_{self.zarr_name}")
        output_integrated_path = Path(f"integrated_proseg_{self.zarr_name}")
        temp_int_path = Path(f"temp_integration_{self.zarr_name}")
        
        # Run Proseg refinement
        self.log.info("Running Proseg refinement")
        output_path = run_proseg_refinement(
            transcripts_df=transcripts_df,
            output_path=str(output_path),
            proseg_binary="/users/k22026807/.cargo/bin/proseg", #"/users/k2481276/.cargo/bin/proseg",
            #"/Users/k2481276/Downloads/proseg-main/target/release/proseg",
            x_col="x_updated",
            y_col="y_updated",
            z_col="z",
            gene_col="feature_name",
            cell_id_col="cell_id",
            samples=samples,
            voxel_size=voxel_size,
            voxel_layers=voxel_layers,
            nuclear_reassignment_prob=nuclear_reassignment_prob,
            diffusion_probability=diffusion_probability,
            num_threads=num_threads,
            overwrite=True,
            logger=self.log
        )

        # Fix metadata
        self.log.info("Fixing zarr metadata")
        fix_zarr_metadata(output_path, self.log)

        self.log.info("Fixing AnnData table")
        fix_anndata_table(output_path, self.log)

        # Load Proseg components
        self.log.info("Loading Proseg components")
        refined_shapes_gdf, adata_proseg, refined_transcripts_df = load_proseg_components(
            output_path, self.log
        )

        # Validate and parse components
        self.log.info("Validating and parsing components")
        refined_sdata, adata_sanitized, refined_shapes_parsed, instance_key = validate_and_parse_components(
            output_path, refined_shapes_gdf, adata_proseg, self.log
        )

        # Create integrated SpatialData
        self.log.info("Creating integrated SpatialData")
        integrated_sdata = create_integrated_spatialdata(
            sdata,
            output_integrated_path,
            refined_shapes_parsed,
            adata_sanitized,
            instance_key,
            transcripts_df,
            refined_transcripts_df,
            self.log
        )

        # Create visualizations
        self.log.info("Creating visualizations")
        create_visualizations(integrated_sdata, refined_shapes_gdf, self.log)

        # Generate comparison statistics
        self.log.info("Generating comparison statistics")
        generate_comparison_statistics(
            sdata, refined_sdata, integrated_sdata, self.log)

        # Export data
        self.log.info("Exporting data to various formats")
        export_data(
            refined_shapes_gdf,
            adata_proseg,
            refined_transcripts_df,
            self.log
        )
        
        print("EXPORTED!!!")
        # Reload and process final data
        self.log.info("Reloading integrated SpatialData for final processing")
        self.integrated_sdata = sd.read_zarr(output_integrated_path)

        # Process with Xenium-specific processing
        self.log.info("Processing refined AnnData with Xenium-specific QC")
        self.process_xenium_adata_proseg()

        # Save updated integrated SpatialData
        self.log.info("Saving updated integrated SpatialData")

        final_temp_path = Path(f"final_temp_{self.zarr_name}")
        if final_temp_path.exists():
            shutil.rmtree(final_temp_path)

        self.integrated_sdata.write(str(final_temp_path))

        shutil.rmtree(output_integrated_path)
        shutil.move(str(final_temp_path), str(output_integrated_path))

        self.log.info("Proseg Refinement Pipeline Complete!")

        return self.integrated_sdata


    def process_xenium_adata_proseg(self, gene_list_path="gene_list_10X.csv"):
        """
        Process Proseg-refined Xenium AnnData: compute QC metrics, remove controls, add missing genes.
        
        This is an adapted version of process_xenium_adata for Proseg-refined data.
        
        Parameters
        ----------
        adata : AnnData
            Input refined AnnData object from Proseg
        gene_list_path : str, optional
            Path to reference gene list CSV. Defaults to "gene_list_10X.csv".
        
        Returns
        -------
        adata : AnnData
            Processed AnnData object with QC metrics and standardized gene set
        """
        
        # Define control keywords
        keywords = {
            'control_probe': 'NegControlProbe',
            'genomic_control': 'Intergenic_Region',
            'control_codeword': 'NegControlCodeword',
            'unassigned_codeword': 'UnassignedCodeword',
            'deprecated_codeword': 'DeprecatedCodeword'
        }
        
        adata = self.integrated_sdata.tables['table_refined']
        print("PROCESSING PROSEG-REFINED XENIUM ANNDATA")
        print(f"Original shape: {adata.shape}")

        self.log.info("Copying metadata from original SpatialData")

        if 'table' in self.integrated_sdata.tables:
            info_core = self.integrated_sdata.tables['table_original'].uns.keys()
            for k in info_core:
                adata.obs[k] = self.integrated_sdata.tables['table_original'].uns[k]

        # Get cell geometries and compute spatial metrics
        self.log.info("Computing cell areas and centroids")
        
        # Note: For Proseg data, shapes are in 'cell_boundaries_refined'
        if hasattr(self, 'new_sdata') and 'cell_boundaries_refined' in self.integrated_sdata.shapes:
            cells = self.integrated_sdata.shapes['cell_boundaries_refined']

        elif 'shapes' in self.integrated_sdata.shapes:
            cells = self.integrated_sdata.shapes['shapes']

        else:
            self.log.warning("No cell shapes found, skipping spatial metrics")
            cells = None
    
        if cells is not None:
            # Filter to valid cells in adata
            valid_cells = adata.obs_names.astype(int)
            cells_filtered = cells[cells.index.isin(valid_cells)]
            # Compute metrics
            adata.obs['cell_area'] = cells_filtered.geometry.area.values
            centroids = cells_filtered.geometry.centroid
            adata.obsm['spatial'] = np.column_stack([centroids.x, centroids.y])
            
        # 1. Compute QC metrics
        print("1. Computing QC metrics")

        for metric_name, keyword in keywords.items():
            mask = adata.var_names.str.contains(keyword, na=False, regex=False)
            counts = np.array(adata[:, mask].X.sum(axis=1)).flatten()
            adata.obs[f'{metric_name}_counts'] = counts
            print(f"{metric_name}_counts: {mask.sum()} features")

        # Identify all control features
        all_control_mask = pd.Series(False, index=adata.var_names)
        for keyword in keywords.values():
            all_control_mask |= adata.var_names.str.contains(
                keyword, na=False, regex=False)

        # Transcript counts (non-control genes)
        transcript_mask = ~all_control_mask
        adata.obs['transcript_counts'] = np.array(
            adata[:, transcript_mask].X.sum(axis=1)
        ).flatten()
        print(f"transcript_counts: {transcript_mask.sum()} features")

        # Total counts
        adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()

        # Summary stats
        print("QC Metrics Summary:")
        qc_cols = ['control_probe_counts', 'genomic_control_counts',
                   'control_codeword_counts', 'unassigned_codeword_counts',
                   'deprecated_codeword_counts', 'transcript_counts', 'total_counts']
        print(adata.obs[qc_cols].describe().loc[['mean', 'std', 'min', 'max']])

        # 2. Remove control features
        print("2. Removing control features")
        print(f"Control features: {all_control_mask.sum()}")
        print(f"Regular genes: {transcript_mask.sum()}")
        adata = adata[:, transcript_mask].copy()
        print(f"After removal: {adata.shape}")

        # 3. Add missing genes
        print("3. Adding missing genes from reference")
        gene_list = pd.read_csv(gene_list_path)['gene_name'].tolist()
        print(f"Reference list: {len(gene_list)} genes")
        current_genes = set(adata.var_names)
        missing_genes = [
            gene for gene in gene_list if gene not in current_genes]
        print(f"Missing genes: {len(missing_genes)}")

        if len(missing_genes) > 0:
            # Store original metadata
            original_uns = adata.uns.copy()
            original_obsm = adata.obsm.copy() if adata.obsm.keys() else {}
            original_obsp = adata.obsp.copy() if adata.obsp.keys() else {}
            original_varm = adata.varm.copy() if adata.varm.keys() else {}
            original_varp = adata.varp.copy() if adata.varp.keys() else {}
            original_layers = {
                k: v for k, v in adata.layers.items()} if adata.layers.keys() else {}

            # Create zero matrix for missing genes
            missing_matrix = np.zeros((adata.n_obs, len(missing_genes)))

            # Match the format of existing X matrix (sparse or dense)
            if hasattr(adata.X, 'toarray'):  # If sparse
                missing_matrix = csr_matrix(missing_matrix)

            # Create var DataFrame
            missing_var = pd.DataFrame(index=missing_genes)
            for col in adata.var.columns:
                if adata.var[col].dtype == 'object' or adata.var[col].dtype == 'string':
                    missing_var[col] = ''

                elif adata.var[col].dtype == 'bool':
                    missing_var[col] = False

                else:
                    missing_var[col] = 0

            missing_adata = sc.AnnData(
                X=missing_matrix,
                obs=adata.obs.copy(),
                var=missing_var
            )

            # Concatenate
            adata = ad.concat([adata, missing_adata], axis=1,
                              join='outer', merge='same')

            # Clean up NaN values
            for col in adata.var.columns:
                if adata.var[col].dtype == 'object' or adata.var[col].dtype == 'string':
                    adata.var[col] = adata.var[col].fillna('')

                elif adata.var[col].dtype == 'bool':
                    adata.var[col] = adata.var[col].fillna(False)

            # Restore metadata
            adata.uns = original_uns
            if original_obsm:
                adata.obsm = original_obsm

            if original_obsp:
                adata.obsp = original_obsp

            if original_varm:
                adata.varm = original_varm

            if original_varp:
                adata.varp = original_varp

            if original_layers:
                for key, value in original_layers.items():
                    if hasattr(value, 'toarray'):  # sparse
                        missing_layer = csr_matrix(
                            np.zeros((adata.n_obs, len(missing_genes))))
                        adata.layers[key] = sp.hstack([value, missing_layer])

                    else:  # dense
                        missing_layer = np.zeros(
                            (adata.n_obs, len(missing_genes)))
                        adata.layers[key] = np.hstack([value, missing_layer])

            print(f"After adding: {adata.shape}")

        # 4. Reorder to match reference
        print("4. Reordering genes to match reference")
        genes_in_order = [
            gene for gene in gene_list if gene in adata.var_names]

        if len(genes_in_order) != len(gene_list):
            print(
                f"WARNING: {len(gene_list) - len(genes_in_order)} genes from reference not found!")

        adata = adata[:, genes_in_order].copy()

        print("PROCESSING COMPLETE")
        print(f"Final shape: {adata.shape}")
        print(f"Cells: {adata.n_obs}")
        print(f"Genes: {adata.n_vars}")
        
        self.integrated_sdata.tables['table_refined'] = adata

        return adata

    # ---------------- SAVE ---------------- #

    def update_spatialdata(self, proseg_refinement=True, run_pipeline=True):

        """Save processed SpatialData into a new Zarr file.
        
        Executes the complete processing pipeline and writes all results to a
        new SpatialData Zarr file. Creates a new SpatialData object containing
        the original image, processed labels, shapes, points, and gene expression
        table.
        
        Returns:
            sd.SpatialData: New SpatialData object containing all processed layers:
                - images: Original image from input
                - labels: Upscaled segmentation masks
                - shapes: Cell polygon geometries
                - points: Transcript coordinates with cell assignments
                - tables: AnnData gene expression matrix
        
        Note:
            - Runs full pipeline: masks→shapes, labels, points→tables
            - Creates output path: "updated_{zarr_name}" in zarr_dir
            - Overwrites existing output if present
            - Closes any active Dask client
            - Clears GPU memory and deletes intermediate variables
            - Prints progress messages for each step
        
        Pipeline Steps:
            1. process_masks_to_shapes(): Filter and convert masks to polygons
            2. process_labels(): Upscale masks to full resolution
            3. process_points_and_tables(): Assign transcripts and create expression matrix
            4. Write all data to new Zarr file
        """
        
        if run_pipeline:
            self.process_masks_to_shapes()
            print("Masks to shapes finished")
            self.process_labels()
            print("Labels process finished")
            self.process_points_and_tables()
            print("Points and Tables process finished")

        self.new_sdata = sd.SpatialData()
        dst_path = os.path.join(
            self.output_folder, f"updated_{self.zarr_name}")

        # Images
        img = self.sdata.images[self.image_name]
        try:
            # if dask-backed
            self.new_sdata.images[self.image_name] = img.compute()
            
        except Exception:
            # already in memory
            self.new_sdata.images[self.image_name] = img.copy()

        # Labels
        lbl = self.labels_model
        try:
            self.new_sdata.labels[self.label_name] = lbl.compute()
            
        except Exception:
            self.new_sdata.labels[self.label_name] = lbl.copy()

        # Shapes
        shp = self.shapes_model
        try:
            self.new_sdata.shapes[self.shape_name] = shp.compute()
            
        except Exception:
            self.new_sdata.shapes[self.shape_name] = shp.copy()

        # Points
        try:
            transcripts_df = self.points_model.compute()
            transcripts_df = transcripts_df.dropna(subset=['cell_id']).copy()
            
            transcripts_df = transcripts_df.reset_index(drop=True)

            for col in transcripts_df.select_dtypes(include=['object', 'string']).columns:
                transcripts_df[col] = transcripts_df[col].astype('category')
                
            self.new_sdata.points[self.point_name] = PointsModel.parse(transcripts_df)
            
        except Exception as e:
            print(f"Warning: Could not compute points cleanly. Error: {e}")
            self.new_sdata.points[self.point_name] = self.points_model.copy()
        
        tbl = self.vdata.copy()
        print('tbl')
        print(tbl)

        # Remove NaN columns from obs and var
        tbl.obs = tbl.obs.loc[:, tbl.obs.columns.notna()]
        tbl.var = tbl.var.loc[:, tbl.var.columns.notna()]
        # Replace NaN index values with strings (AnnData requires non-NaN unique index)
        tbl.obs = tbl.obs.map(lambda x: "NA" if pd.isna(x) else x)
        tbl.var = tbl.var.map(lambda x: "NA" if pd.isna(x) else x)

        # Ensure indices are unique (AnnData requires unique obs/var names)
        tbl.obs.index = tbl.obs.index.map(
            lambda x: str(x) if pd.notna(x) else "unknown_obs")
        tbl.var.index = tbl.var.index.map(
            lambda x: str(x) if pd.notna(x) else "unknown_var")

        if not tbl.obs.index.is_unique:
            tbl.obs.index = pd.Index(
                [f"cell_{i}" for i in range(len(tbl.obs))])
            
        if not tbl.var.index.is_unique:
            tbl.var.index = pd.Index(
                [f"gene_{i}" for i in range(len(tbl.var))])

        self.new_sdata.tables['table'] = tbl

        self.process_xenium_adata()

        if proseg_refinement == True:
            self.run_proseg()

        print("updated_data")

        try:
            client = Client.current()
            client.close()
            
        except ValueError:
            pass

        # Todo: update the following line when compatibility of spatialdata 0.6 ok
        # with other package. It will allow update_sdata_path argument to bypass
        # not self-contained issue when saving
        # new_sdata.write(dst_path, overwrite=True, update_sdata_path=False)
        self.new_sdata.write(dst_path, overwrite=True)
        print("Data saved")

        for var in (self.masks, self.flows, self.styles,
                    self.gdf_points, self.gdf_polygons):
            try:
                del var
            except Exception:
                pass
        self.clear_gpu_memory()

        return self.new_sdata


# ─────────────────────────────────────────────────────────────────────────────
# SEAM DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# Layout (same as reseg.py):
#   4 cols × 8 rows  →  patch_w ≈ 12816 px,  patch_h ≈ 9368 px
#
# Horizontal seams (28 total, indices 0-27):
#   Centred on each ROW boundary.
#   Strip height = 2 * margin (narrow, spans full col width + margin each side).
#   Indexed row-major: seam_index = row_boundary_idx * n_cols + col_idx
#     row_boundary_idx ∈ [0, n_rows-2]  (7 boundaries)
#     col_idx          ∈ [0, n_cols-1]  (4 columns)
#
# Vertical seams (24 total, indices 28-51):
#   Centred on each COL boundary.
#   Strip width = 2 * margin (narrow, spans full row height + margin each side).
#   Indexed row-major: v_index = row_idx * (n_cols-1) + col_boundary_idx
#     row_idx          ∈ [0, n_rows-1]  (8 rows)
#     col_boundary_idx ∈ [0, n_cols-2]  (3 boundaries)
#
# margin = 1500 px  (covers the largest expected cell diameter and then some)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    original_zarr_name = "BC_prime.zarr"
    original_zarr_dir  = "/scratch/users/k22026807/masters/project/xenium_output"
    temp_crop_dir      = "/scratch/users/k22026807/masters/project/resegmentation/temp_crops_seams"

    os.makedirs(temp_crop_dir, exist_ok=True)

    WHOLE_ROI = {
        'xmin': 0,
        'ymin': 0,
        'width': 51265,
        'height': 74945
    }
    
    n_cols   = 4
    n_rows   = 8
    patch_w  = WHOLE_ROI['width']  / n_cols   # ~12816 px
    patch_h  = WHOLE_ROI['height'] / n_rows   # ~9368  px
    margin   = 1500                            # px around each boundary

    total_H  = (n_rows - 1) * n_cols          # 28 horizontal seams
    total_V  = n_rows * (n_cols - 1)          # 24 vertical seams
    total    = total_H + total_V              # 52

    try:
        seam_index = int(sys.argv[1])
    except (IndexError, ValueError):
        print("Warning: no index provided, defaulting to 0.")
        seam_index = 0

    if seam_index >= total:
        raise ValueError(f"seam_index {seam_index} out of range (0-{total-1}).")

    # ── Calculate crop window ─────────────────────────────────────────────────

    if seam_index < total_H:
        # ── Horizontal seam ──────────────────────────────────────────────────
        # Strip centred on a row boundary, full column width (+ margin each side)
        row_boundary = seam_index // n_cols          # which row gap (0-6)
        col_idx      = seam_index  % n_cols          # which column  (0-3)

        y_center = (row_boundary + 1) * patch_h      # boundary pixel
        y_start  = max(0,                 y_center - margin)
        y_end    = min(WHOLE_ROI['height'], y_center + margin)

        x_start  = max(0,                col_idx       * patch_w - margin)
        x_end    = min(WHOLE_ROI['width'], (col_idx + 1) * patch_w + margin)

        patch_name_str = f"seam_H_{seam_index:02d}"
        print(f"Horizontal seam {seam_index}: row_boundary={row_boundary+1}, col={col_idx}")

    else:
        # ── Vertical seam ────────────────────────────────────────────────────
        # Strip centred on a col boundary, full row height (+ margin each side)
        v_index      = seam_index - total_H
        row_idx      = v_index // (n_cols - 1)       # which row band (0-7)
        col_boundary = v_index  % (n_cols - 1)       # which col gap  (0-2)

        x_center = (col_boundary + 1) * patch_w      # boundary pixel
        x_start  = max(0,                 x_center - margin)
        x_end    = min(WHOLE_ROI['width'],  x_center + margin)

        y_start  = max(0,                  row_idx      * patch_h - margin)
        y_end    = min(WHOLE_ROI['height'], (row_idx + 1) * patch_h + margin)

        patch_name_str = f"seam_V_{v_index:02d}"
        print(f"Vertical seam {v_index}: row={row_idx}, col_boundary={col_boundary+1}")

    print(f"Crop: X[{x_start:.0f} : {x_end:.0f}]  Y[{y_start:.0f} : {y_end:.0f}]")

    # ── Crop & save ───────────────────────────────────────────────────────────

    crop_name = f"{patch_name_str}_{original_zarr_name}"
    crop_path = os.path.join(temp_crop_dir, crop_name)

    if os.path.exists(crop_path):
        shutil.rmtree(crop_path)

    zarr_full_path = os.path.join(original_zarr_dir, original_zarr_name)
    print(f"Attempting to load original Zarr from: {zarr_full_path}")
    full_sdata = sd.read_zarr(zarr_full_path)

    sdata_roi = bounding_box_query(
        full_sdata,
        axes=["x", "y"],
        min_coordinate=[x_start, y_start],
        max_coordinate=[x_end,   y_end],
        target_coordinate_system='global',
    )
    print(f"Saving crop to {crop_path}...")
    sdata_roi.write(crop_path)

    # ── Run pipeline — identical call pattern to reseg.py ────────────────────

    reseg = Resegmentation_xenium(
        zarr_dir      = temp_crop_dir,
        zarr_name     = crop_name,
        output        = f"images/dapi_{patch_name_str}.png",
        factor_rescale= 2,
        image_name    = "morphology_focus",
        label_name    = "cell_labels_seam",
        shape_name    = "cell_boundaries_seam",
        point_name    = "transcripts",
    )

    reseg.check_gpu()
    reseg.preprocess_image(channel_names=["DAPI"], channels_to_use=["DAPI"])

    reseg.run_cellpose(
        model_type         = "cyto3",
        diameter           = None,
        gpu                = True,
        tile_overlap       = 0.1,
    )

    # ── Extract and save cellpose flows ──────────────────────────────────────────────────────────────
    print("Extracting cellpose flows for lattice stitching...")

    dP = reseg.flows[1]
    cellprob = reseg.flows[2]

    # scale global offsets to factor rescale
    factor = reseg.factor_rescale
    y_offset_scaled = int(((row_boundary + 1) * patch_h - margin) // factor)
    x_offset_scaled = int(x_start // factor) 

    offsets = np.array([y_offset_scaled, x_offset_scaled])

    # save to a dedicated directory for the stitching
    seam_flow_dir = os.path.join("data", "seams_flows")
    os.makedirs(seam_flow_dir, exist_ok=True)

    np.save(os.path.join(seam_flow_dir, f"{patch_name_str}_dP.npy"), dP)
    np.save(os.path.join(seam_flow_dir, f"{patch_name_str}_cellprob.npy"), cellprob)
    np.save(os.path.join(seam_flow_dir, f"{patch_name_str}_offsets.npy"), offsets)

    print(f"Saved flows and downscaled offsets to {seam_flow_dir}")

    # ── CLEANUP ──────────────────────────────────────────────────────────────
    print("\n--- Initiating post-processing cleanup ---")
    
    # Clean up the original crop
    if os.path.exists(crop_path):
        shutil.rmtree(crop_path)

    # Clean up the "temp_crops_seams" folder if it's now empty
    if os.path.exists(reseg.output_folder) and not os.listdir(reseg.output_folder):
        os.rmdir(reseg.output_folder)

    print(f"Finished job for {patch_name_str}")