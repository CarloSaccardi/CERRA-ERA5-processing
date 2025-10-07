# CERRA Data Processing Pipeline

This directory contains a clean, modular pipeline for downloading, projecting, and preprocessing CERRA (Copernicus European Regional Reanalysis) data from the Copernicus Climate Data Store (CDS).

## Overview

The pipeline processes CERRA data through three main steps:

1. **Download**: Download CERRA data from CDS in Lambert conformal projection
2. **Projection**: Convert from Lambert to cylindrical (lat-lon) projection for specific regions
3. **Preprocessing**: Extract and compute meteorological variables, save as HDF5 files for efficient PyTorch loading

## Directory Structure

```
CERRA_processing/
├── scripts/                       # Python scripts
│   ├── config.py                 # Configuration file with all settings
│   ├── download_cerra.py         # Download script
│   ├── project_cerra.py          # Projection conversion script
│   ├── preprocess_cerra.py       # Data preprocessing script
│   └── example_usage.py          # Usage examples
├── config/                        # Configuration files
│   ├── cyl.txt                   # Central Europe coordinate file
│   ├── cyl_iberia.txt            # Iberia coordinate file
│   └── cyl_scandinavia.txt       # Scandinavia coordinate file
├── lambert_proj/                  # Downloaded data (Lambert projection)
│   ├── single_levels/            # Time-varying variables (2013.grib, 2014.grib, etc.)
│   ├── single_levels_humidity/   # Humidity data (r2 variable)
│   └── single_levels_static/     # Static variables (orography.grib, etc.)
├── latlon_proj_*/                 # Projected data (cylindrical projection)
│   ├── single_levels/            # Projected time-varying variables
│   ├── single_levels_humidity/   # Projected humidity data
│   └── single_levels_static/     # Projected static variables
├── download_scipts/               # Download helper scripts
├── README.md                      # This documentation
└── requirements.txt               # Python dependencies
```

## Prerequisites

### Required Software

1. **Python 3.7+** with the following packages:
   - `cdsapi` - For downloading from CDS
   - `xarray` - For handling GRIB data
   - `numpy` - For numerical operations
   - `cfgrib` - For reading GRIB files

2. **CDO (Climate Data Operators)** - For projection conversion
   - Install from: https://code.mpimet.mpg.de/projects/cdo
   - Or via conda: `conda install -c conda-forge cdo`

### CDS API Setup

1. Register at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
2. Get your API key from your user profile
3. Create `~/.cdsapirc` file with your credentials:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: <your-uid>:<your-api-key>
   ```

## Quick Start

The pipeline is organized as a multi-step workflow:

### Step 1: Download Data

Download CERRA data from CDS in Lambert projection:

```bash
# Download all variables for specific years
python scripts/download_cerra.py --years 2021 --variables all_variables

# Download with nohup for background processing
nohup python scripts/download_cerra.py --years 2013 --variables all_variables > lambert_proj/2013.log 2>&1 &
```

### Download Static Variables

Static variables like orography need to be downloaded separately:

```bash
# Download orography (run once)
nohup python download_scipts/orography.py > lambert_proj/orography.log 2>&1 &
```

After downloading, move orography to the static variables folder:
```bash
mkdir -p lambert_proj/single_levels_static
mv lambert_proj/single_levels/orography.grib lambert_proj/single_levels_static/
```

### Download and Separate Humidity Data

If you have humidity data, separate it by year:

```bash
# Create humidity directory
mkdir -p lambert_proj/single_levels_humidity

# Split multi-year humidity files into individual years
for year in 2013 2014 2015; do
    grib_copy -w year=$year download_scipts/humidity_2013_15.grib lambert_proj/single_levels_humidity/${year}.grib
done
```

### Step 2: Project to Lat-Lon Grid

Convert from Lambert projection to cylindrical (lat-lon) projection using the `--remap_directories` argument:

```bash
# Project time-varying variables for specific years
python scripts/project_cerra.py \
  --region central_europe \
  --years 2013 2014 2015 \
  --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity

# Project static variables only (no --years needed)
python scripts/project_cerra.py \
  --region central_europe \
  --remap_directories lambert_proj/single_levels_static

# Project everything at once
python scripts/project_cerra.py \
  --region central_europe \
  --years 2013 2014 2015 \
  --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity lambert_proj/single_levels_static

# Project a specific file
python scripts/project_cerra.py \
  --region scandinavia \
  --input-file 2021.grib \
  --remap_directories lambert_proj/single_levels
```

**Key Points:**
- `--remap_directories`: List of directories to process (required)
- `--years`: Process specific years (optional; if omitted, processes all .grib files)
- `--input-file`: Process a specific file by name (optional)
- Output directories are automatically created by replacing `lambert_proj` with `latlon_proj_{region}`

### Step 3: Preprocess Data

Extract and compute meteorological variables, save as HDF5 files:

```bash
# Preprocess for a specific year
python scripts/preprocess_cerra.py --region central_europe --year 2021 --variables base_variables
```

## Configuration

### Regions

The pipeline supports three predefined regions:

- **central_europe**: Central Europe region
- **iberia**: Iberian Peninsula
- **scandinavia**: Scandinavia region

### Variable Groups

Three variable groups are predefined:

- **base_variables**: Core variables (u10, v10, t2m) - 3 variables
- **additional_variables**: Additional variables (momentum fluxes, sensible heat flux, surface pressure) - 4 variables  
- **all_variables**: All available variables (u10, v10, t2m, momentum fluxes, sensible heat flux, surface pressure) - 7 variables

### Custom Variables

You can specify custom variables:

```bash
python cerra_pipeline.py --region central_europe --years 2021 --variables 10m_wind_speed,2m_temperature
```

## Output Data

### Processed Data Format

The preprocessing step generates HDF5 files for efficient PyTorch loading:

1. **Main data file**: `{region}_cerra_data.h5`
   - Contains time series data with shape `(N, height, width, variables)`
   - For `base_variables`: `(N, 368, 368, 3)` with u10, v10, t2m
   - For `all_variables`: `(N, 368, 368, 7)` with all available variables
   - Organized in train/val/test splits

2. **Static data file**: `{region}_cerra_static.h5`
   - Contains coordinate grid: `nwp_xy` with shape `(2, height, width)`
   - Contains normalization parameters: `parameter_mean.pt` and `parameter_std.pt`

### Output Locations

- **Central Europe**: `data/CentralEurope/CERRA/samples/`
- **Iberia**: `data/iberia/CERRA/samples/`
- **Scandinavia**: `data/scandinavia/CERRA/samples/`

### HDF5 Dataset Usage

For PyTorch training, use the provided `CERRAHDF5DataModule`:

```python
from hdf5_dataset import CERRAHDF5DataModule

# Load HDF5 data
data_module = CERRAHDF5DataModule(
    data_dir="data/CentralEurope/CERRA/samples/",
    batch_size=32,
    train_split=0.7,
    val_split=0.15
)

# Get data loaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
```

## Advanced Usage

### Processing Multiple Regions

You can process the same data for different regions:

```bash
# Project for central Europe
python scripts/project_cerra.py \
  --region central_europe \
  --years 2013 2014 \
  --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity

# Project for iberia (uses same source files)
python scripts/project_cerra.py \
  --region iberia \
  --years 2013 2014 \
  --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity
```

### Custom Input Directories

You can specify custom input directories for projection:

```bash
# Use custom directories
python scripts/project_cerra.py \
  --region central_europe \
  --years 2021 \
  --remap_directories /path/to/custom/dir1 /path/to/custom/dir2
```

### Batch Processing

Process multiple years efficiently:

```bash
# Process years 2013-2021 in one command
python scripts/project_cerra.py \
  --region central_europe \
  --years 2013 2014 2015 2016 2017 2018 2019 2020 2021 \
  --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity
```

## Adding New Variables

To add new variables to the pipeline:

1. **Update configuration**: Add variables to `VARIABLE_GROUPS` in `config.py`
2. **Update preprocessing**: Modify `CERRAProcessorHDF5` class in `preprocess_cerra.py` to handle new variables
3. **Update target variables**: Add to `PROCESSING_CONFIG["target_variables"]` in `config.py`

Example:

```python
# In config.py
VARIABLE_GROUPS = {
    "base_variables": [
        "10m_wind_direction",
        "10m_wind_speed", 
        "2m_temperature",
        "new_variable"  # Add here
    ],
    # ...
}

PROCESSING_CONFIG = {
    "target_variables": ["u10", "v10", "t2m", "sshf", "zust", "new_var"],  # Add here
    # ...
}
```

### Workflow for Adding Variables

The pipeline is designed to handle variable additions efficiently:

1. **Download all variables once**: Use `--variables all_variables` to download all available data
2. **Process subsets**: Use `--variables base_variables` or `--variables all_variables` during preprocessing
3. **No re-download needed**: Once downloaded, you can process different variable subsets from the same source files

## Adding New Regions

To add a new region:

1. **Create coordinate file**: Create a new `.txt` file with grid specifications
2. **Update configuration**: Add region to `REGIONS` in `config.py`

Example:

```python
# In config.py
REGIONS = {
    "new_region": {
        "name": "NewRegion",
        "coord_file": "cyl_new_region.txt",
        "description": "New region description"
    },
    # ...
}
```

## Troubleshooting

### Common Issues

1. **CDS API errors**: Check your API key and internet connection
2. **CDO not found**: Install CDO and ensure it's in your PATH
3. **Memory issues**: Process smaller time ranges or use more RAM
4. **File not found**: Check that input files exist in the expected locations

### Logs and Debugging

The scripts provide detailed output during execution. Check the console output for:
- Download progress
- Projection status
- Processing statistics
- Error messages

### Performance Tips

1. **Parallel processing**: Process multiple years in separate terminal sessions
2. **Storage**: Ensure sufficient disk space (CERRA files are large)
3. **Memory**: Use systems with adequate RAM for large datasets
4. **HDF5 efficiency**: HDF5 format provides faster loading and better memory efficiency than individual .npy files
5. **Background processing**: Use `nohup` for long-running downloads to prevent interruption

## File Naming and Organization

### Download Files

- **Lambert projection files**: `{year}.grib` (e.g., `2013.grib`)
- **Log files**: `{year}.log` (e.g., `2013.log`)
- **Location**: `lambert_proj/` directory

### Processed Files

- **HDF5 data files**: `{region}_cerra_data.h5`
- **HDF5 static files**: `{region}_cerra_static.h5`
- **Location**: `data/{region}/CERRA/samples/` directory

### Complete Workflow Example

```bash
# 1. Download all variables for 2013
nohup python scripts/download_cerra.py --years 2013 --variables all_variables > lambert_proj/2013.log 2>&1 &

# 2. Download and organize static variables
nohup python download_scipts/orography.py > lambert_proj/orography.log 2>&1 &
mkdir -p lambert_proj/single_levels_static
mv lambert_proj/single_levels/orography.grib lambert_proj/single_levels_static/

# 3. Download and separate humidity data (if needed)
mkdir -p lambert_proj/single_levels_humidity
grib_copy -w year=2013 download_scipts/humidity_2013_15.grib lambert_proj/single_levels_humidity/2013.grib

# 4. Project for central Europe
python scripts/project_cerra.py \
  --region central_europe \
  --years 2013 \
  --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity lambert_proj/single_levels_static

# 5. Project for iberia (from same downloaded files)
python scripts/project_cerra.py \
  --region iberia \
  --years 2013 \
  --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity lambert_proj/single_levels_static

# 6. Preprocess for each region
python scripts/preprocess_cerra.py --region central_europe --year 2013 --variables base_variables
python scripts/preprocess_cerra.py --region iberia --year 2013 --variables base_variables
```

## Data Quality

### Temporal Coverage

- **Analysis data**: 3-hourly for high-resolution
- **Forecast data**: Hourly for forecast range 1-6 hours
- **Leap years**: Automatically handled (2928 samples vs 2920 for normal years)

### Spatial Resolution

- **Original**: 5.5 km × 5.5 km (Lambert projection)
- **Projected**: Variable resolution based on region (typically ~0.05°)

### Variable Validation

The pipeline includes basic validation:
- Wind component computation from speed/direction
- Physical unit conversions
- Missing data handling

## Support

For issues or questions:

1. Check this documentation
2. Review the script help messages: `python <script>.py --help`
3. Check the CDS documentation: https://cds.climate.copernicus.eu/
4. Verify prerequisites: `python cerra_pipeline.py --check-prerequisites`

## License

This pipeline is provided under the same license as the CERRA dataset (CC-BY 4.0).
