## Unveiling cell-cell communication using random walks in spatial transcriptomic networks pipeline

This repository contains a Python-based computational workflow for analysing spatial transcriptomics data, with a focus on unveiling cell-cell communication (CCC) within the tumor microenvironment.

This pipeline integrates multi-omics data (spatial coordinates, transcriptomics and H&E images) to derive systems-level insights into tumor organisation and signalling interactions.

## Pipeline Architecture
The project is structured to process raw spatial data through to spatial cell graph construction and CCC method benchmarking:

* **`01_resegmentation/`**: Scripts for cell boundary resegmentation using Cellpose-Proseg. Scaled to process full 2x1cm histological slides.
* **`02_preprocessing/`**: Quality control, data normalisation and spatial transformation scripts.
* **`03_celltyping/`**: Workflow for marker-based cell type assignment and spatial mapping.
* **`04_network_construction/`**: Graph-based modeling of the spatial microenvironment.
* **`05_random_walks/`**: Implementation of random walk approaches to investigate signalling interactions.
* **`06_benchmarking/`**: An evaluation module comparing multiple CCC methods (`COMMOT`, `NCEM`, `SpaCI`, `SpatialDM`, `Squidpy`, `stLearn`) to assess consistency of inferred interactions.
* **`tools/`**: Custom utility functions.
