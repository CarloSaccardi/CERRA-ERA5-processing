# ERA5 Data Processing Pipeline

This directory contains scripts for downloading and preprocessing ERA5 reanalysis data from the Copernicus Climate Data Store (CDS).

## Overview

The ERA5 pipeline consists of three main components:

1. **Configuration** (`config.py`) - Defines regions, variables, and paths
2. **Download** (`download_era5.py`) - Downloads data from CDS API
3. **Preprocessing** (`preprocess_era5.py`) - Processes NetCDF files to HDF5 format
4. **Pipeline** (`era5_pipeline.py`) - Unified interface for running the full pipeline

## Features

- **Geographic Selection**: Direct API-based geographic area selection (no coordinate files needed)
- **Variable Groups**: Support for `base_variables` (u10, v10, t2m) and `all_variables` (u10, v10, t2m, sshf, zust, sp)
- **Multiple Regions**: Central Europe, Iberia, and Scandinavia
- **HDF5 Output**: Efficient storage format for PyTorch loading
- **Flexible Processing**: Can run individual steps or full pipeline

## Prerequisites

Install required dependencies:

```bash
pip install cdsapi xarray h5py numpy
```

Set up CDS API key (see [CDS API documentation](https://cds.climate.copernicus.eu/api-how-to)).

## Usage

### Full Pipeline

Run the complete pipeline (download + preprocess):

```bash
# Single year
python era5_pipeline.py --region central_europe --years 2021

# Multiple years
python era5_pipeline.py --region iberia --years 2020 2021 2022

# With all variables
python era5_pipeline.py --region scandinavia --years 2021 --variables all_variables

# Year ranges
python era5_pipeline.py --region central_europe --years 2020-2021
```

### Skip Steps

**Download only (skip preprocessing):**
```bash
python era5_pipeline.py --region central_europe --years 2021 --skip-preprocessing
```

**Preprocess only (skip download):**
```bash
python era5_pipeline.py --region central_europe --years 2021 --skip-download
```

### Direct Script Usage

**Download:**
```bash
python download_era5.py --region central_europe --years 2021
python download_era5.py --region iberia --years 2021 --variables all_variables
```

**Preprocess:**
```bash
python preprocess_era5.py --region central_europe --year 2021
python preprocess_era5.py --region iberia --input-file single_2021.nc
```

## Output Structure

```
data/
└── {region}/
    └── ERA5/
        └── samples/
            ├── {region}_{year}_era5.h5    # Main data file
            └── nwp_xy.npy                 # Coordinate grid
```

## Variable Groups

### Base Variables (`base_variables`)
- `u10`: 10m u-component of wind
- `v10`: 10m v-component of wind  
- `t2m`: 2m temperature

### All Variables (`all_variables`)
- `u10`: 10m u-component of wind
- `v10`: 10m v-component of wind
- `t2m`: 2m temperature
- `sshf`: Instantaneous surface sensible heat flux
- `zust`: Friction velocity
- `sp`: Surface pressure

## Regions

- **central_europe**: [60°N, -2°W, 40°N, 18°E]
- **iberia**: [45°N, -10°W, 35°N, 5°E]  
- **scandinavia**: [72°N, 5°E, 55°N, 30°E]

## File Formats

- **Input**: NetCDF files from CDS API
- **Output**: HDF5 files with metadata and coordinate grids
- **Compatible**: Matches CERRA pipeline output format for consistency
