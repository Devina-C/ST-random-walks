import os
import logging
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Union
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from tqdm import tqdm
import zarr
from anndata import read_zarr as read_anndata_zarr
import shutil
import matplotlib.pyplot as plt
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity
from spatialdata import SpatialData
import dask.dataframe as dd
#from spatialdata import sanitize_table
from spatialdata.models import PointsModel
from packaging.version import Version
# Configure logging
log = logging.getLogger(__name__)

def _check_proseg_available(proseg_binary: str) -> str:
    """
    Check if proseg is available and return version string.
    """
    
    try:
        result = subprocess.run(
            [proseg_binary, "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip().split(
        )[1] if result.stdout else "unknown"
        
        return version
    
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        raise RuntimeError(
            f"Proseg binary '{proseg_binary}' not found or not executable. "
            "Please install Proseg from: https://github.com/dcjones/proseg"
        )


def _check_proseg_version_gte(version: str, target: str) -> bool:
    """
    Check if proseg version is >= target version.
    """
    
    try:
        return Version(version) >= Version(target)
    
    except Exception:
        log.warning(
            f"Could not parse proseg version '{version}'. Assuming >= {target}")
        
        return True


def run_proseg_refinement(
    transcripts_df: pd.DataFrame,
    output_path: Union[str, Path],
    proseg_binary: Union[str, Path],
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    gene_col: str = "feature_name",
    cell_id_col: str = "cell_id",
    samples: int = 1000,
    voxel_size: float = 0.5,
    voxel_layers: int = 1,
    nuclear_reassignment_prob: float = 0.2,
    diffusion_probability: float = 0.2,
    num_threads=12,
    overwrite: bool = True,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Run Proseg refinement on transcript data.
    
    Parameters
    ----------
    transcripts_df : pd.DataFrame
        DataFrame containing transcript data with coordinates, gene names, and cell IDs
    output_path : str or Path
        Path where the Proseg output zarr will be saved
    proseg_binary : str or Path
        Path to the Proseg executable
    x_col : str, default "x"
        Name of the X coordinate column
    y_col : str, default "y"
        Name of the Y coordinate column
    z_col : str, default "z"
        Name of the Z coordinate column
    gene_col : str, default "feature_name"
        Name of the gene/feature column
    cell_id_col : str, default "cell_id"
        Name of the cell ID column
    samples : int, default 1000
        Number of MCMC sampling iterations
    voxel_size : float, default 0.5
        Size of voxels for segmentation (smaller = higher resolution)
    voxel_layers : int, default 1
        Number of voxel layers in Z-direction
    nuclear_reassignment_prob : float, default 0.2
        Probability of reassigning transcripts from nuclear regions
    diffusion_probability : float, default 0.2
        Probability of transcript diffusion to neighboring cells
    overwrite : bool, default True
        Whether to overwrite existing output
    logger : logging.Logger, optional
        Logger instance. If None, uses root logger
        
    Returns
    -------
    Path
        Path to the output zarr file
        
    Raises
    ------
    RuntimeError
        If Proseg execution fails
    ValueError
        If required columns are missing from transcripts_df
        
    Examples
    --------
    >>> output = run_proseg_refinement(
    ...     transcripts_df=my_transcripts,
    ...     output_path="proseg_output.zarr",
    ...     proseg_binary="/path/to/proseg",
    ...     voxel_size=1.0,
    ...     samples=500
    ... )
    """

    log.info("Running Proseg refinement")
    # Setup logger
    if logger is None:
        logger = logging.getLogger(__name__)
    
    os.environ['RAYON_NUM_THREADS'] = str(num_threads)
    logger.info(f"Using {num_threads} CPU threads for Proseg")

    # Convert paths
    output_path = Path(output_path)
    proseg_binary = Path(proseg_binary)

    # Validate inputs
    required_columns = [x_col, y_col, z_col, gene_col, cell_id_col]
    missing_columns = [
        col for col in required_columns if col not in transcripts_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in transcripts_df: {missing_columns}")

    # Check proseg availability
    proseg_version = _check_proseg_available(proseg_binary)
    use_zarr_output = _check_proseg_version_gte(proseg_version, "3.0.0")
    logger.info(f"Using proseg version: {proseg_version}")

    # Prepare transcripts
    logger.info("Preparing transcript data")
    proseg_columns = [x_col, y_col, z_col, gene_col, cell_id_col]
    transcript_table = transcripts_df[proseg_columns].copy()
    logger.info(
        f"Selected {len(transcript_table)} transcripts with columns: {proseg_columns}")

    # Create working directory and run proseg
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_path = tmp_path / "transcripts.csv"

        # Save transcripts to CSV
        transcript_table.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(transcript_table)} transcripts to {csv_path}")

        # Build Proseg command
        cmd = [
            str(proseg_binary),
            str(csv_path),
            "-x", x_col,
            "-y", y_col,
            "-z", z_col,
            "--gene-column", gene_col,
            "--cell-id-column", cell_id_col,
            "--cell-id-unassigned", "0",
            "--samples", str(samples),
            "--voxel-size", str(voxel_size),
            "--voxel-layers", str(voxel_layers),
            "--nuclear-reassignment-prob", str(nuclear_reassignment_prob),
            "--diffusion-probability", str(diffusion_probability),
        ]

        if use_zarr_output:
            cmd.extend(["--output-spatialdata", str(output_path)])

        if overwrite:
            cmd.append("--overwrite")

        # Run Proseg
        logger.info("Running Proseg refinement")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("Parameters:")
        logger.info(f"- Samples: {samples}")
        logger.info(f"- Voxel size: {voxel_size}")
        logger.info(f"- Voxel layers: {voxel_layers}")
        logger.info(
            f"- Nuclear reassignment prob: {nuclear_reassignment_prob}")
        logger.info(f"- Diffusion probability: {diffusion_probability}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error(
                f"Proseg failed with return code: {result.returncode}")
            logger.error(f"Proseg stdout:{result.stdout}")
            logger.error(f"Proseg stderr:{result.stderr}")
            raise RuntimeError(
                f"Proseg execution failed with return code {result.returncode}")

        logger.info("Proseg completed successfully!")

        # Log output info
        if result.stdout:
            logger.debug(f"Proseg stdout:{result.stdout}")

    # Verify output exists
    if not output_path.exists():
        raise RuntimeError(f"Proseg output not found at {output_path}")

    logger.info(f"Output saved to: {output_path}")

    return output_path


def fix_anndata_table(output_path, log):
    """
    Fix AnnData table by ensuring proper categorical types and spatialdata attributes.
    
    Parameters
    ----------
    output_path : Path
        Base path containing the tables directory
    log : logging.Logger
        Logger instance for output messages
        
    Returns
    -------
    bool
        True if table was found and processed, False otherwise
    """
    
    table_path = output_path / "tables" / "table"

    if not table_path.exists():
        log.warning(f"Table path does not exist: {table_path}")
        
        return False

    log.info("Reading table from zarr")
    adata = read_anndata_zarr(table_path)
    needs_save = False

    # Fix region column if it exists
    if 'region' in adata.obs.columns:
        log.info(f"Current 'region' dtype: {adata.obs['region'].dtype}")

        if not pd.api.types.is_categorical_dtype(adata.obs['region']):
            log.info("  Converting 'region' to categorical")
            adata.obs['region'] = pd.Categorical(adata.obs['region'])
            needs_save = True
            
        else:
            log.info("Region already categorical")
            
    else:
        log.warning("'region' column not found in adata.obs")

    # Ensure spatialdata_attrs exists in uns
    if 'spatialdata_attrs' not in adata.uns:
        log.info("Adding spatialdata_attrs to uns")
        adata.uns['spatialdata_attrs'] = {
            'region': 'cell_boundaries',
            'region_key': 'region',
            'instance_key': 'cell'  # Note: Proseg uses 'cell' not 'cell_id'
        }
        needs_save = True

    # Save if needed
    if needs_save:
        log.info("Saving updated table")
        temp_path = table_path.parent / "table_temp"

        if temp_path.exists():
            shutil.rmtree(temp_path)

        adata.write_zarr(temp_path)
        shutil.rmtree(table_path)
        shutil.move(str(temp_path), str(table_path))
        log.info("Table saved with fixes")
        
    else:
        log.info("No table changes needed")

    log.info("All fixes applied!")
    
    return True


def _assign_transcripts_to_cells(
    transcripts_df: pd.DataFrame,
    cell_shapes: gpd.GeoDataFrame,
    x_col: str,
    y_col: str,
    cell_id_col: str
) -> pd.DataFrame:
    """
    Assign transcripts to cells based on spatial overlap.
    
    Parameters
    ----------
    transcripts_df : pd.DataFrame
        Dataframe containing transcripts with x, y coordinates
    cell_shapes : gpd.GeoDataFrame
        GeoDataFrame containing cell polygons
    x_col : str
        Column name for x coordinates
    y_col : str
        Column name for y coordinates  
    cell_id_col : str
        Column name to store cell assignments
        
    Returns
    -------
    pd.DataFrame
        Transcripts dataframe with cell_id_col added
    """
    
    log.info("Assigning transcripts to cells via spatial join")

    with tqdm(total=3, desc="Spatial assignment", unit="step") as pbar:
        # Create GeoDataFrame from transcripts
        pbar.set_description("Creating transcript geometry")
        transcript_gdf = gpd.GeoDataFrame(
            transcripts_df,
            geometry=gpd.points_from_xy(
                transcripts_df[x_col], transcripts_df[y_col])
        )
        pbar.update(1)

        # Spatial join - find which cell each transcript is in
        pbar.set_description("Performing spatial join")
        joined = gpd.sjoin(
            transcript_gdf,
            cell_shapes.reset_index(),
            how="left",
            predicate="within"
        )
        pbar.update(1)

        # Extract cell IDs (from the right index of the join)
        # If transcript is not in any cell, assign 0 (unassigned)
        pbar.set_description("Assigning cell IDs")
        transcripts_df[cell_id_col] = joined["index_right"].fillna(
            0).astype(int)
        pbar.update(1)

    n_assigned = (transcripts_df[cell_id_col] != 0).sum()
    n_total = len(transcripts_df)
    pct_assigned = 100 * n_assigned / n_total

    log.info(
        f"Assigned {n_assigned}/{n_total} transcripts ({pct_assigned:.1f}%) to cells. "
        f"{n_total - n_assigned} transcripts unassigned."
    )

    return transcripts_df


def fix_zarr_metadata(zarr_path, log):
    """
    Comprehensive fix for zarr metadata to ensure SpatialData compatibility.
    """
    
    log.info("Fixing zarr metadata")
    store = zarr.open(str(zarr_path), mode='r+')

    # Fix root level attributes
    log.info("Step 1: Fixing root attributes")
    if 'spatialdata_attrs' not in store.attrs:
        log.info("Adding root spatialdata_attrs")
        store.attrs['spatialdata_attrs'] = {
            'version': '0.2.0',
            'spatialdata': '0.2.0'
        }

    else:
        attrs = dict(store.attrs['spatialdata_attrs'])
        if 'version' not in attrs:
            attrs['version'] = '0.2.0'

        if 'spatialdata' not in attrs:
            attrs['spatialdata'] = '0.2.0'

        store.attrs['spatialdata_attrs'] = attrs
        log.info("Root spatialdata_attrs updated")

    # Fix tables group
    log.info("Step 2: Fixing tables metadata")
    if 'tables' in store:
        tables_group = store['tables']

        for table_name in tables_group.keys():
            table_group = tables_group[table_name]
            log.info(f"Processing table: {table_name}")
            current_attrs = dict(table_group.attrs)

            if 'encoding-type' not in current_attrs:
                log.info("Adding encoding-type")
                current_attrs['encoding-type'] = 'anndata'

            if 'encoding-version' not in current_attrs:
                log.info("Adding encoding-version")
                current_attrs['encoding-version'] = '0.2.0'

            if 'spatialdata_attrs' not in current_attrs:
                log.info("Adding spatialdata_attrs to table")
                current_attrs['spatialdata_attrs'] = {'version': '0.2.0'}

            table_group.attrs.update(current_attrs)
            log.info(f"Table {table_name} metadata updated")

    log.info("All zarr metadata fixed!")


def fix_anndata_table(output_path, log):
    """
    Fix AnnData table by ensuring proper categorical types and spatialdata attributes.
    
    Parameters
    ----------
    output_path : Path
        Base path containing the tables directory
    log : logging.Logger
        Logger instance for output messages
        
    Returns
    -------
    bool
        True if table was found and processed, False otherwise
    """
    
    log.info("Fixing table data")
    table_path = output_path / "tables" / "table"

    if not table_path.exists():
        log.warning(f"Table path does not exist: {table_path}")
        
        return False

    log.info("Reading table from zarr")
    adata = read_anndata_zarr(table_path)
    needs_save = False

    # Fix region column if it exists
    if 'region' in adata.obs.columns:
        log.info(f"Current 'region' dtype: {adata.obs['region'].dtype}")

        if not pd.api.types.is_categorical_dtype(adata.obs['region']):
            log.info("  Converting 'region' to categorical")
            adata.obs['region'] = pd.Categorical(adata.obs['region'])
            needs_save = True
            
        else:
            log.info("Region already categorical")
            
    else:
        log.warning("'region' column not found in adata.obs")

    # Ensure spatialdata_attrs exists in uns
    if 'spatialdata_attrs' not in adata.uns:
        log.info("Adding spatialdata_attrs to uns")
        adata.uns['spatialdata_attrs'] = {
            'region': 'cell_boundaries',
            'region_key': 'region',
            'instance_key': 'cell'  # Note: Proseg uses 'cell' not 'cell_id'
        }
        needs_save = True

    # Save if needed
    if needs_save:
        log.info("Saving updated table")
        temp_path = table_path.parent / "table_temp"

        if temp_path.exists():
            shutil.rmtree(temp_path)

        adata.write_zarr(temp_path)
        shutil.rmtree(table_path)
        shutil.move(str(temp_path), str(table_path))
        log.info("Table saved with fixes")
        
    else:
        log.info("No table changes needed")

    log.info("All fixes applied!")

    return table_path


def load_proseg_components(output_path, log):
    """
    Load Proseg output components including shapes, table, and optionally transcripts.
    
    Parameters
    ----------
    output_path : Path
        Base path containing the Proseg output directories
    log : logging.Logger
        Logger instance for output messages
        
    Returns
    -------
    tuple
        (shapes_gdf, adata, transcripts_df) where transcripts_df may be None if not available
    """
    
    log.info("Loading Proseg output components")

    # Load shapes from parquet
    log.info("Loading shapes from parquet")
    shapes_parquet = output_path / "shapes" / "cell_boundaries" / "shapes.parquet"
    refined_shapes_gdf = gpd.read_parquet(shapes_parquet)
    log.info(f"Loaded {len(refined_shapes_gdf)} shapes")

    # Load table
    log.info("Loading table")
    table_path = output_path / "tables" / "table"
    adata = read_anndata_zarr(table_path)
    log.info(f"Loaded table: {adata.shape}")
    log.info(f"Table obs columns: {list(adata.obs.columns)}")

    # Load points if available
    refined_transcripts_df = None
    points_parquet = output_path / "points" / "transcripts" / "points.parquet"

    if points_parquet.exists():
        log.info("Loading transcripts from parquet")
        refined_transcripts_df = dd.read_parquet(points_parquet).compute()
        log.info(f"Loaded {len(refined_transcripts_df)} transcripts")
        log.info(f"Transcript columns: {list(refined_transcripts_df.columns)}")

        # Check for cell assignment column, Proseg might use 'cell' instead of 'cell_id'
        if 'assignment' in refined_transcripts_df.columns:
            assigned = (refined_transcripts_df['assignment'] != 0).sum()
            log.info(
                f"Assigned (using 'cell' column): {assigned} ({100*assigned/len(refined_transcripts_df):.1f}%)")

        elif 'cell_id' in refined_transcripts_df.columns:
            assigned = (refined_transcripts_df['cell_id'] != '0').sum()
            log.info(
                f"Assigned (using 'cell_id' column): {assigned} ({100*assigned/len(refined_transcripts_df):.1f}%)")

        else:
            log.warning("No cell assignment column found in transcripts")

    else:
        log.warning(f"Transcripts file not found: {points_parquet}")

    return refined_shapes_gdf, adata, refined_transcripts_df


def validate_and_parse_components(output_path, refined_shapes_gdf, adata, log):
    """
    Validate and parse Proseg components with proper SpatialData models.
    
    This function:
    1. Loads shapes with transformations from zarr or parses from GeoDataFrame
    2. Sanitizes the AnnData table with proper region and instance_key mappings
    3. Creates a complete SpatialData object with validated components
    
    Parameters
    ----------
    output_path : Path
        Base path containing the Proseg output
    refined_shapes_gdf : gpd.GeoDataFrame
        GeoDataFrame containing cell boundaries
    adata : AnnData
        AnnData table to validate and sanitize
    log : logging.Logger
        Logger instance for output messages
        
    Returns
    -------
    tuple
        (refined_sdata, adata_sanitized, refined_shapes_parsed, instance_key) where:
        - refined_sdata is the complete SpatialData object
        - adata_sanitized is the validated table (may be None if creation failed)
        - refined_shapes_parsed is the parsed shapes with transformations
        - instance_key is the determined instance key for cells
    """
    
    log.info("Validating and parsing components")

    # Try to load shapes directly from the complete zarr output
    log.info("Attempting to load shapes from complete SpatialData zarr")
    
    try:
        # Try to load the complete SpatialData object
        temp_sdata = sd.read_zarr(output_path)

        if "cell_boundaries" in temp_sdata.shapes:
            refined_shapes_parsed = temp_sdata.shapes["cell_boundaries"]
            log.info("Loaded shapes with transformations from zarr")

            # Also keep as GeoDataFrame for plotting
            refined_shapes_gdf = refined_shapes_parsed
            
        else:
            raise KeyError("cell_boundaries not found in shapes")

    except Exception as e:
        
        log.warning(f"Could not load complete SpatialData: {e}")
        log.info("Falling back to parquet loading and manual parsing")

        # Load from parquet and add transformations manually
        refined_shapes_gdf.crs = None

        # Parse with ShapesModel and add Identity transformation
        refined_shapes_parsed = ShapesModel.parse(
            refined_shapes_gdf,
            transformations={"global": Identity()}
        )
        log.info("Parsed shapes with Identity transformation")

    # Sanitize table with TableModel
    log.info("Sanitizing table")

    # First ensure the table references match
    if 'region' not in adata.obs.columns:
        log.info("  Adding 'region' column to obs")
        adata.obs['region'] = 'cell_boundaries'

    if not pd.api.types.is_categorical_dtype(adata.obs['region']):
        log.info("  Converting 'region' to categorical")
        adata.obs['region'] = pd.Categorical(adata.obs['region'])

    # Determine the instance_key - Proseg uses 'cell' not 'cell_id'
    if 'spatialdata_attrs' in adata.uns:
        instance_key = adata.uns['spatialdata_attrs'].get(
            'instance_key', 'cell')
        
    else:
        instance_key = 'cell'
        adata.uns['spatialdata_attrs'] = {
            'region': 'cell_boundaries',
            'region_key': 'region',
            'instance_key': instance_key
        }

    log.info(f"Using instance_key: {instance_key}")
    log.info(f"Table obs columns: {list(adata.obs.columns)}")
    log.info(f"Table obs index name: {adata.obs.index.name}")

    # Make sure the instance_key column exists in obs
    if instance_key not in adata.obs.columns:
        log.warning(f"Instance key '{instance_key}' not found in obs columns")

        # Check if it's in the index
        if adata.obs.index.name == instance_key:
            log.info(f"Found '{instance_key}' in index, moving to column")
            adata.obs[instance_key] = adata.obs.index

        elif 'cell' in adata.obs.columns:
            log.info("Found 'cell' column, using as instance_key")
            adata.obs[instance_key] = adata.obs['cell']

        else:
            log.info(f"Using obs index as '{instance_key}'")
            adata.obs[instance_key] = adata.obs.index

    # Ensure the instance_key values match the shapes
    log.info(
        f"Instance key '{instance_key}' contains {len(adata.obs[instance_key].unique())} unique values")
    log.info(f"Shapes contains {len(refined_shapes_parsed)} geometries")

    # Get the shape identifiers
    if hasattr(refined_shapes_parsed, 'index'):
        shape_ids = set(refined_shapes_parsed.index.astype(str))
        log.info(f"Shape IDs (first 5): {list(shape_ids)[:5]}")

    elif isinstance(refined_shapes_parsed, gpd.GeoDataFrame):
        shape_ids = set(refined_shapes_parsed.index.astype(str))
        log.info(f"Shape IDs (first 5): {list(shape_ids)[:5]}")

    else:
        log.warning("Could not extract shape IDs")
        shape_ids = None

    # Check for matching IDs
    if shape_ids is not None:
        table_ids = set(adata.obs[instance_key].astype(str))
        matching_ids = shape_ids.intersection(table_ids)
        log.info(f"Matching IDs between table and shapes: {len(matching_ids)}")

        if len(matching_ids) == 0:
            log.warning("No matching IDs found, attempting type conversion")
            # Try converting types
            adata.obs[instance_key] = adata.obs[instance_key].astype(str)

            # Re-check
            table_ids = set(adata.obs[instance_key].astype(str))
            matching_ids = shape_ids.intersection(table_ids)
            log.info(f"After conversion - Matching IDs: {len(matching_ids)}")

    # Create a region-instance mapping for sanitization
    log.info("Creating region-instance mapping")

    # The mapping should be from region -> GeoDataFrame
    # Should be 'cell_boundaries'
    region = adata.obs['region'].cat.categories[0]
    log.info(f"Region: {region}")

    # Try sanitizing with explicit region specification
    try:
        log.info("Attempting to sanitize table")
        adata_sanitized = sanitize_table(
            adata,
            inplace=False
        )

        if adata_sanitized is None:
            raise ValueError("sanitize_table returned None")

        log.info(f"Table sanitized: {adata_sanitized.shape}")

    except Exception as e:
        log.error(f"sanitize_table failed: {e}")
        log.info("Using table without sanitization")
        adata_sanitized = adata

        # Manually ensure the table has proper structure
        if 'spatialdata_attrs' not in adata_sanitized.uns:
            adata_sanitized.uns['spatialdata_attrs'] = {}

        adata_sanitized.uns['spatialdata_attrs'].update({
            'region': region,
            'region_key': 'region',
            'instance_key': instance_key
        })

        log.info(
            f"Using unsanitized table with manual metadata: {adata_sanitized.shape}")

    # Create refined SpatialData manually
    log.info("Creating refined SpatialData")
    try:
        refined_sdata = SpatialData(
            shapes={"cell_boundaries": refined_shapes_parsed},
            tables={"table": adata_sanitized}
        )
        log.info("Refined SpatialData created!")

    except Exception as e:
        log.error(f"Failed to create SpatialData: {e}")
        log.info("Attempting to create without table")

        refined_sdata = SpatialData(
            shapes={"cell_boundaries": refined_shapes_parsed}
        )
        log.info("Refined SpatialData created (without table)")

        # Set adata_sanitized to None so we don't try to use it later
        adata_sanitized = None

    return refined_sdata, adata_sanitized, refined_shapes_parsed, instance_key


def create_integrated_spatialdata(
    sdata,
    output_path,
    refined_shapes_parsed,
    adata_sanitized,
    instance_key,
    transcripts_df=None,
    refined_transcripts_df=None,
    log=None
):
    """
    Create an integrated SpatialData object combining original and refined components.
    
    This function:
    1. Copies all images and labels from the original SpatialData
    2. Adds both original and refined cell boundaries as separate shape layers
    3. Adds original and refined transcripts (if available)
    4. Creates properly sanitized tables for both original and refined data
    5. Combines everything into a single integrated SpatialData object
    
    Parameters
    ----------
    sdata : SpatialData
        Original SpatialData object
    output_path : Path
        Path to Proseg output directory
    refined_shapes_parsed : GeoDataFrame or ShapesModel
        Parsed refined cell boundaries from Proseg
    adata_sanitized : AnnData or None
        Sanitized AnnData table from Proseg (refined)
    instance_key : str
        Column name used as instance identifier (e.g., 'cell' or 'cell_id')
    transcripts_df : pd.DataFrame, optional
        Original transcripts dataframe
    refined_transcripts_df : pd.DataFrame, optional
        Refined transcripts dataframe from Proseg
    log : logging.Logger
        Logger instance for output messages
        
    Returns
    -------
    SpatialData
        Integrated SpatialData object with all components
    """
    
    log.info("Creating integrated SpatialData")

    # Prepare components for new SpatialData
    images_dict = {}
    labels_dict = {}
    shapes_dict = {}
    points_dict = {}
    tables_dict = {}

    # 1. Copy all images from original
    log.info("Step 1: Copying images from original SpatialData")
    for img_key in sdata.images.keys():
        images_dict[img_key] = sdata.images[img_key]
        log.info(f"Copied image: {img_key}")

    # 2. Copy all labels from original
    log.info("Step 2: Copying labels from original SpatialData")
    for label_key in sdata.labels.keys():
        labels_dict[label_key] = sdata.labels[label_key]
        log.info(f"Copied label: {label_key}")

    # 3. Add refined cell boundaries (already parsed)
    log.info("Step 3: Adding refined cell boundaries")
    shapes_dict["cell_boundaries_refined"] = refined_shapes_parsed
    log.info(f"Added {len(refined_shapes_parsed)} refined cell boundaries")

    # Also keep original shapes for comparison
    shape_keys = list(sdata.shapes.keys())
    if len(shape_keys) > 0:
        # Just grab the first shape layer available
        original_shape_key = shape_keys[0]
        original_shapes = sdata.shapes[original_shape_key]
        shapes_dict["cell_boundaries_original"] = original_shapes
        log.info(f"Added {len(original_shapes)} original cell boundaries from layer '{original_shape_key}'")
    else:
        log.warning("No shapes found in original sdata to use for comparison.")
        original_shapes = None

    # 4. Add points
    log.info("Step 4: Adding transcripts")
    if "transcripts" in sdata.points:
        points_dict["transcripts_original"] = sdata.points["transcripts"]
        log.info(
            f"Added {len(transcripts_df) if transcripts_df is not None else 'unknown'} original transcripts")

    # Try to add refined transcripts from the loaded SpatialData
    try:
        # Try to read from the original zarr structure
        refined_sdata_temp = sd.read_zarr(output_path)
        if "transcripts" in refined_sdata_temp.points:
            points_dict["transcripts_refined"] = refined_sdata_temp.points["transcripts"]
            refined_transcripts_compute = refined_sdata_temp.points["transcripts"].compute(
            )
            log.info(
                f"Added {len(refined_transcripts_compute)} refined transcripts")

            # Check for the correct column name
            if 'cell' in refined_transcripts_compute.columns:
                assigned = (refined_transcripts_compute['cell'] != 0).sum()
                log.info(
                    f"Assigned: {assigned} ({100*assigned/len(refined_transcripts_compute):.1f}%)")

            elif 'cell_id' in refined_transcripts_compute.columns:
                assigned = (
                    refined_transcripts_compute['cell_id'] != '0').sum()
                log.info(
                    f"Assigned: {assigned} ({100*assigned/len(refined_transcripts_compute):.1f}%)")

    except Exception as e:
        log.warning(f"Could not load refined transcripts from zarr: {e}")
        log.info("  Attempting to use parquet directly")

        if refined_transcripts_df is not None:
            # Determine coordinate columns
            coord_cols = []
            
            for col in ['x', 'y', 'z']:
                if col in refined_transcripts_df.columns:
                    coord_cols.append(col)

            log.info(f"Found coordinate columns: {coord_cols}")

            # Parse with PointsModel
            try:
                refined_points = PointsModel.parse(
                    refined_transcripts_df,
                    coordinates={"x": "x", "y": "y", "z": "z"} if len(
                        coord_cols) == 3 else {"x": "x", "y": "y"},
                    transformations={"global": Identity()}
                )
                points_dict["transcripts_refined"] = refined_points
                log.info(
                    f"Added {len(refined_transcripts_df)} refined transcripts from parquet")

            except Exception as e2:
                log.warning(
                    f"Could not parse refined transcripts with PointsModel: {e2}")

    # 5. Add tables - CREATE TEMPORARY SPATIALDATA OBJECTS FOR PROPER SANITIZATION
    log.info("Step 5: Adding tables")

    # Refined table
    if adata_sanitized is not None:
        log.info("  Preparing refined table")
        refined_table_copy = adata_sanitized.copy()

        # Update region to point to the refined shapes
        refined_table_copy.obs['region'] = pd.Categorical(
            ['cell_boundaries_refined'] * len(refined_table_copy))

        # Update metadata
        if 'spatialdata_attrs' not in refined_table_copy.uns:
            refined_table_copy.uns['spatialdata_attrs'] = {}

        refined_table_copy.uns['spatialdata_attrs']['region'] = 'cell_boundaries_refined'
        refined_table_copy.uns['spatialdata_attrs']['region_key'] = 'region'

        # Keep the instance_key as is
        if 'instance_key' not in refined_table_copy.uns['spatialdata_attrs']:
            refined_table_copy.uns['spatialdata_attrs']['instance_key'] = instance_key

        # Create temporary SpatialData to properly sanitize
        log.info("  Creating temporary SpatialData for refined table sanitization")
        temp_sdata_refined = SpatialData(
            shapes={"cell_boundaries_refined": refined_shapes_parsed},
            tables={"table_refined": refined_table_copy}
        )

        # Sanitize with context
        try:
            refined_table_sanitized = sanitize_table(
                refined_table_copy,
                sdata=temp_sdata_refined,
                region_key='region',
                instance_key=instance_key
            )
            if refined_table_sanitized is not None:
                tables_dict["table_refined"] = refined_table_sanitized
                log.info(
                    f"Added refined table (sanitized): {refined_table_sanitized.shape}")

            else:
                tables_dict["table_refined"] = refined_table_copy
                log.info(
                    f"Added refined table (unsanitized): {refined_table_copy.shape}")

        except Exception as e:
            log.warning(f"Could not sanitize refined table: {e}")
            tables_dict["table_refined"] = refined_table_copy
            log.info(
                f"Added refined table (unsanitized): {refined_table_copy.shape}")
    else:
        log.warning("No refined table available (adata_sanitized is None)")

    # Original table
    if "table" in sdata.tables:
        log.info("  Preparing original table")
        original_table = sdata.tables["table"].copy()

        # Update region to point to original shapes
        original_table.obs['region'] = pd.Categorical(
            ['cell_boundaries_original'] * len(original_table))

        # Update metadata
        if 'spatialdata_attrs' not in original_table.uns:
            original_table.uns['spatialdata_attrs'] = {}

        original_table.uns['spatialdata_attrs']['region'] = 'cell_boundaries_original'
        original_table.uns['spatialdata_attrs']['region_key'] = 'region'

        # Get the original instance_key
        original_instance_key = original_table.uns.get(
            'spatialdata_attrs', {}).get('instance_key', 'cell_id')

        # Ensure instance_key column exists
        if original_instance_key not in original_table.obs.columns:
            log.warning(
                f"Original instance_key '{original_instance_key}' not in obs columns")
            log.info(
                f"Available columns: {list(original_table.obs.columns)}")

            # Try common alternatives
            if 'cell_id' in original_table.obs.columns:
                original_instance_key = 'cell_id'

            elif 'cells' in original_table.obs.columns:
                original_instance_key = 'cells'

            else:
                # Use index
                original_table.obs[original_instance_key] = original_table.obs.index

            original_table.uns['spatialdata_attrs']['instance_key'] = original_instance_key

        log.info(
            f"Using instance_key for original table: {original_instance_key}")

        # Create temporary SpatialData to properly sanitize
        log.info("  Creating temporary SpatialData for original table sanitization")
        temp_sdata_original = SpatialData(
            shapes={"cell_boundaries_original": original_shapes},
            tables={"table_original": original_table}
        )

        # Sanitize with context
        try:
            original_table_sanitized = sanitize_table(
                original_table,
                sdata=temp_sdata_original,
                region_key='region',
                instance_key=original_instance_key
            )

            if original_table_sanitized is not None:
                tables_dict["table_original"] = original_table_sanitized
                log.info(
                    f"Added original table (sanitized): {original_table_sanitized.shape}")

            else:
                log.warning("sanitize_table returned None for original table")
                tables_dict["table_original"] = original_table
                log.info(
                    f"Added original table (unsanitized): {original_table.shape}")

        except Exception as e:
            log.warning(f"Could not sanitize original table: {e}")
            tables_dict["table_original"] = original_table
            log.info(
                f"Added original table (unsanitized): {original_table.shape}")

    # 6. Create the integrated SpatialData object
    log.info("Step 6: Creating integrated SpatialData object")

    # Only include non-empty dictionaries
    sdata_kwargs = {}
    if images_dict:
        sdata_kwargs['images'] = images_dict
        
    if labels_dict:
        sdata_kwargs['labels'] = labels_dict
        
    if shapes_dict:
        sdata_kwargs['shapes'] = shapes_dict
        
    if points_dict:
        sdata_kwargs['points'] = points_dict
        
    if tables_dict:
        sdata_kwargs['tables'] = tables_dict

    try:
        integrated_sdata = SpatialData(**sdata_kwargs)
        log.info("Integrated SpatialData created!")

    except Exception as e:
        log.error(f"Failed to create integrated SpatialData: {e}")
        log.info("Attempting to create without tables")

        # Try without tables
        sdata_kwargs_no_tables = {k: v for k,
                                  v in sdata_kwargs.items() if k != 'tables'}
        integrated_sdata = SpatialData(**sdata_kwargs_no_tables)
        log.info("Integrated SpatialData created (without tables)")

    log.info("Integrated SpatialData structure:")
    log.info(f"- Images: {list(integrated_sdata.images.keys())}")
    log.info(f"- Labels: {list(integrated_sdata.labels.keys())}")
    log.info(f"- Shapes: {list(integrated_sdata.shapes.keys())}")
    log.info(f"- Points: {list(integrated_sdata.points.keys())}")
    log.info(f"- Tables: {list(integrated_sdata.tables.keys())}")
    
    print("="*60)
    print("TRANSCRIPTS ASSOCIATED TO CELLS")
    print("="*60)
    
    ori_transcript_associated_cells = int(integrated_sdata.tables['table_original'].X.sum())
    ref_transcript_associated_cells = int(integrated_sdata.tables['table_refined'].X.sum())
    ratio_associated_cells = ref_transcript_associated_cells/ori_transcript_associated_cells
    
    print("Original: ", ori_transcript_associated_cells)
    print("Refined: ", ref_transcript_associated_cells)
    print(f"Ratio: {ratio_associated_cells:.1f}")
    print("="*60)
    
    # Save integrated SpatialData
    if output_path.exists():
        log.info(
            f"Removing existing output: {output_path}")
        shutil.rmtree(output_path)

    log.info(
        f"Writing integrated SpatialData to {output_path}")
    integrated_sdata.write(output_path)

    return integrated_sdata


def create_visualizations(integrated_sdata, refined_shapes_gdf, log):
    """
    Create visualizations comparing original and refined segmentations.
    
    Parameters
    ----------
    integrated_sdata : SpatialData
        Integrated SpatialData object containing both original and refined components
    refined_shapes_gdf : gpd.GeoDataFrame
        GeoDataFrame containing refined cell boundaries
    log : logging.Logger
        Logger instance for output messages
        
    Returns
    -------
    dict
        Dictionary with paths to created visualization files
    """
    
    log.info("Creating visualizations")

    output_files = {}

    # Plot 1: Refined segmentation
    log.info("Creating refined segmentation plot")
    fig, ax = plt.subplots(figsize=(12, 12))
    refined_shapes_gdf.plot(
        ax=ax,
        facecolor='lightblue',
        edgecolor='black',
        aspect=1,
        linewidth=0.5,
        alpha=0.7
    )
    ax.set_aspect('equal')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Refined Cell Boundaries ({len(refined_shapes_gdf)} cells)')
    plt.tight_layout()

    refined_seg_path = 'refined_segmentation.png'
    plt.savefig(refined_seg_path, dpi=150, bbox_inches='tight')
    log.info(f"Saved: {refined_seg_path}")
    output_files['refined_segmentation'] = refined_seg_path
    plt.close()

    # Plot 2: Comparison
    if "cell_boundaries_original" in integrated_sdata.shapes:
        log.info("Creating comparison plot")
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Get original as GeoDataFrame for plotting
        orig_gdf = integrated_sdata.shapes["cell_boundaries_original"]

        orig_gdf.plot(
            ax=axes[0],
            facecolor='lightcoral',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7
        )
        axes[0].set_aspect('equal')
        axes[0].set_title(f'Original ({len(orig_gdf)} cells)')

        refined_shapes_gdf.plot(
            ax=axes[1],
            facecolor='lightblue',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7
        )
        axes[1].set_aspect('equal')
        axes[1].set_title(f'Refined ({len(refined_shapes_gdf)} cells)')

        plt.tight_layout()

        comparison_path = 'segmentation_comparison.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved: {comparison_path}")
        output_files['comparison'] = comparison_path
        plt.close()
        
    else:
        log.warning(
            "Original cell boundaries not found in integrated_sdata, skipping comparison plot")

    return output_files


def generate_comparison_statistics(sdata, refined_sdata, integrated_sdata, log):
    """
    Generate and display comparison statistics between original and refined segmentations.
    
    Parameters
    ----------
    sdata : SpatialData
        Original SpatialData object
    refined_sdata : SpatialData
        Refined SpatialData object from Proseg
    integrated_sdata : SpatialData
        Integrated SpatialData object containing both versions
    log : logging.Logger
        Logger instance for output messages
        
    Returns
    -------
    pd.DataFrame or None
        Comparison results dataframe, or None if comparison failed
    """
    
    log.info("Generating comparison statistics")
    
    try:
        comparison = compare_segmentations(
            original_sdata=sdata,
            refined_sdata=refined_sdata,
            original_key="shapes",
            refined_key="cell_boundaries",
            original_table_key="table",
            refined_table_key="table"
        )

        print("" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(comparison.to_string(index=False))

        n_cells_orig = comparison.loc[comparison['metric']
                                      == 'n_cells', 'original'].values[0]
        n_cells_ref = comparison.loc[comparison['metric']
                                     == 'n_cells', 'refined'].values[0]
        n_cells_diff = comparison.loc[comparison['metric']
                                      == 'n_cells', 'difference'].values[0]
        n_cells_pct = comparison.loc[comparison['metric']
                                     == 'n_cells', 'percent_change'].values[0]
        
        print("" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Original cells: {n_cells_orig:.0f}")
        print(f"Refined cells:  {n_cells_ref:.0f}")
        print(f"Change:         {n_cells_diff:.0f} ({n_cells_pct:.1f}%)")

        # Transcript assignment comparison
        if 'transcripts_refined' in integrated_sdata.points and 'transcripts_original' in integrated_sdata.points:
            print("" + "="*60)
            print("TRANSCRIPT ASSIGNMENT")
            print("="*60)

            orig_trans = integrated_sdata.points['transcripts_original'].compute(
            )
            ref_trans = integrated_sdata.points['transcripts_refined'].compute(
            )

            # Handle different column names for original
            if 'cell_id' in orig_trans.columns:
                orig_assigned = (orig_trans['cell_id'] != 0).sum()

            elif 'cell' in orig_trans.columns:
                orig_assigned = (orig_trans['cell'] != 0).sum()

            else:
                orig_assigned = 0

            # Handle different column names for refined
            if 'assignment' in ref_trans.columns:
                ref_assigned = (ref_trans['assignment'] != 0).sum()

            elif 'cell_id' in ref_trans.columns:
                ref_assigned = (ref_trans['cell_id'] != '0').sum()

            else:
                ref_assigned = 0

            if orig_assigned > 0:
                print(
                    f"Original assigned transcripts: {orig_assigned} ({100*orig_assigned/len(orig_trans):.1f}%)")
                print(
                    f"Refined assigned transcripts:  {ref_assigned} ({100*ref_assigned/len(ref_trans):.1f}%)")
                change = ref_assigned - orig_assigned
                pct_change = 100 * change / orig_assigned
                print(f"Change: {change} ({pct_change:.1f}%)")
                
            else:
                log.warning("No assigned transcripts found in original data")
                
        else:
            log.info("Transcript data not available for comparison")

        print("="*60 + "")

        return comparison

    except Exception as e:
        log.warning(f"Could not generate comparison: {e}")
        traceback.print_exc()
        
        return None


def export_data(refined_shapes_gdf, adata, refined_transcripts_df, log):
    """
    Export refined data to various file formats.
    
    Parameters
    ----------
    refined_shapes_gdf : gpd.GeoDataFrame
        Refined cell boundaries to export
    adata : AnnData
        Cell metadata table to export
    refined_transcripts_df : pd.DataFrame or None
        Refined transcripts to export (if available)
    log : logging.Logger
        Logger instance for output messages
        
    Returns
    -------
    dict
        Dictionary with paths to exported files
    """
    
    log.info("Exporting data")

    exported_files = {}

    # Export GeoJSON
    geojson_path = "cell_boundaries_refined.geojson"
    refined_shapes_gdf.to_file(geojson_path, driver="GeoJSON")
    log.info(f"Saved: {geojson_path}")
    exported_files['geojson'] = geojson_path

    # Export Shapefile
    shapefile_path = "cell_boundaries_refined.shp"
    refined_shapes_gdf.to_file(shapefile_path)
    log.info(f"Saved: {shapefile_path}")
    exported_files['shapefile'] = shapefile_path

    # Export metadata
    metadata_path = "cell_metadata_refined.csv"
    adata.obs.to_csv(metadata_path)
    log.info(f"Saved: {metadata_path}")
    exported_files['metadata'] = metadata_path

    # Export transcripts if available
    if refined_transcripts_df is not None:
        transcripts_path = "transcripts_refined.csv"
        refined_transcripts_df.to_csv(transcripts_path, index=False)
        log.info(f"Saved: {transcripts_path}")
        exported_files['transcripts'] = transcripts_path
        
    else:
        log.info("No refined transcripts available to export")

    # Print completion summary
    print("PROCESSING COMPLETE!")
    print("Outputs:")
    print("Visualizations: refined_segmentation.png, segmentation_comparison.png")
    print("Exported files:")
    
    for file_type, file_path in exported_files.items():
        print(f"{file_type:12s}: {file_path}")

    print("Done!")

    return exported_files


def compare_segmentations(
    original_sdata: sd.SpatialData,
    refined_sdata: sd.SpatialData,
    original_key: str = "shapes",
    refined_key: str = "cell_boundaries",
    original_table_key: str = "table",
    refined_table_key: str = "table"
) -> pd.DataFrame:
    """
    Compare original and Proseg-refined segmentations.
    
    Parameters
    ----------
    original_sdata : sd.SpatialData
        Original SpatialData object
    refined_sdata : sd.SpatialData
        Refined SpatialData object from Proseg
    original_key : str
        Key for original cell shapes
    refined_key : str
        Key for refined cell shapes
    original_table_key : str
        Key for original table
    refined_table_key : str
        Key for refined table
        
    Returns
    -------
    pd.DataFrame
        Comparison metrics between original and refined segmentations
        
    Examples
    --------
    >>> comparison = compare_segmentations(original_sdata, refined_sdata)
    >>> print(comparison)
    """
    
    orig_cells = original_sdata.shapes[original_key]
    refined_cells = refined_sdata.shapes[refined_key]

    # Get transcript counts if tables exist
    orig_transcripts = None
    refined_transcripts = None

    if original_table_key in original_sdata.tables:
        orig_table = original_sdata.tables[original_table_key]
        orig_transcripts = np.array(orig_table.X.sum(axis=1)).flatten()

    if refined_table_key in refined_sdata.tables:
        refined_table = refined_sdata.tables[refined_table_key]
        refined_transcripts = np.array(refined_table.X.sum(axis=1)).flatten()

    comparison = pd.DataFrame({
        'metric': [
            'n_cells',
            'mean_area',
            'median_area',
            'total_area',
            'mean_transcripts_per_cell',
            'median_transcripts_per_cell',
            'total_transcripts'
        ],
        'original': [
            len(orig_cells),
            orig_cells.geometry.area.mean(),
            orig_cells.geometry.area.median(),
            orig_cells.geometry.area.sum(),
            orig_transcripts.mean() if orig_transcripts is not None else np.nan,
            np.median(
                orig_transcripts) if orig_transcripts is not None else np.nan,
            orig_transcripts.sum() if orig_transcripts is not None else np.nan,
        ],
        'refined': [
            len(refined_cells),
            refined_cells.geometry.area.mean(),
            refined_cells.geometry.area.median(),
            refined_cells.geometry.area.sum(),
            refined_transcripts.mean() if refined_transcripts is not None else np.nan,
            np.median(
                refined_transcripts) if refined_transcripts is not None else np.nan,
            refined_transcripts.sum() if refined_transcripts is not None else np.nan,
        ]
    })

    comparison['difference'] = comparison['refined'] - comparison['original']
    comparison['percent_change'] = (
        100 * comparison['difference'] /
        comparison['original'].replace(0, np.nan)
    )

    return comparison