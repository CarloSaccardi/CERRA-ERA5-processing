# CERRA Data Processing Pipeline

This directory contains a clean, modular pipeline for downloading, projecting, and preprocessing CERRA (Copernicus European Regional Reanalysis) data from the Copernicus Climate Data Store (CDS).

## Overview

The pipeline processes CERRA data through three main steps:

1. **Download**: Download CERRA data from CDS in Lambert conformal projection
2. **Projection**: Convert from Lambert to cylindrical (lat-lon) projection for specific regions
3. **Preprocessing**: Extract and compute meteorological variables, save as numpy arrays

## Directory Structure

```
CERRA_download/
├── scripts/                  # Python scripts
│   ├── config.py            # Configuration file with all settings
│   ├── download_cerra.py    # Download script
│   ├── project_cerra.py     # Projection conversion script
│   ├── preprocess_cerra.py  # Data preprocessing script
│   ├── cerra_pipeline.py    # Main pipeline orchestrator
│   └── example_usage.py     # Usage examples
├── config/                   # Configuration files
│   ├── cyl.txt              # Central Europe coordinate file
│   ├── cyl_iberia.txt       # Iberia coordinate file
│   └── cyl_scandinavia.txt  # Scandinavia coordinate file
├── lambert_proj/             # Downloaded data (Lambert projection)
│   └── single_levels/
├── latlon_proj_*/            # Projected data (cylindrical projection)
│   └── single_levels/
├── download_scipts/          # Legacy scripts (can be removed)
├── README.md                 # This documentation
└── requirements.txt          # Python dependencies
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

### Full Pipeline (Recommended)

Run the complete pipeline for a region and year(s):

```bash
# Process central Europe for 2021
python scripts/cerra_pipeline.py --region central_europe --years 2021 --variables base_variables

# Process multiple years
python scripts/cerra_pipeline.py --region iberia --years 2014-2021 --variables base_variables

# Process specific years
python scripts/cerra_pipeline.py --region scandinavia --years 2014,2015,2016 --variables base_variables
```

### Individual Steps

You can also run individual steps:

```bash
# Step 1: Download data
python scripts/download_cerra.py --years 2021 --variables base_variables

# Step 2: Project to cylindrical coordinates
python scripts/project_cerra.py --region central_europe --years 2021

# Step 3: Preprocess data
python scripts/preprocess_cerra.py --region central_europe --year 2021
```

## Configuration

### Regions

The pipeline supports three predefined regions:

- **central_europe**: Central Europe region
- **iberia**: Iberian Peninsula
- **scandinavia**: Scandinavia region

### Variable Groups

Three variable groups are predefined:

- **base_variables**: Core variables (u10, v10, t2m, momentum fluxes, sensible heat flux)
- **additional_variables**: Additional variables (relative humidity, surface pressure)
- **all_variables**: All available variables

### Custom Variables

You can specify custom variables:

```bash
python cerra_pipeline.py --region central_europe --years 2021 --variables 10m_wind_speed,2m_temperature
```

## Output Data

### Processed Data Format

The preprocessing step generates:

1. **Time series files**: `nwp_YYYYMMDDHH.npy` - One file per time step
   - Shape: `(height, width, 5)` where 5 variables are:
     - `u10`: 10m eastward wind component (m/s)
     - `v10`: 10m northward wind component (m/s)  
     - `t2m`: 2m temperature (K)
     - `sshf`: Surface sensible heat flux (W/m²)
     - `zust`: Friction velocity (m/s)

2. **Coordinate grid**: `nwp_xy.npy`
   - Shape: `(2, height, width)`
   - Contains longitude and latitude arrays

### Output Locations

- **Central Europe**: `data/CentralEurope/CERRA/samples/`
- **Iberia**: `data/iberia/CERRA/samples/`
- **Scandinavia**: `data/scandinavia/CERRA/samples/`

## Advanced Usage

### Skip Steps

You can skip certain steps if you already have intermediate data:

```bash
# Skip download (use existing Lambert projection data)
python scripts/cerra_pipeline.py --region central_europe --years 2021 --skip-download

# Skip projection (use existing cylindrical projection data)
python scripts/cerra_pipeline.py --region central_europe --years 2021 --skip-projection
```

### Check Prerequisites

Verify that all required software is installed:

```bash
python scripts/cerra_pipeline.py --region central_europe --check-prerequisites
```

### Custom Input/Output Directories

```bash
# Custom input directory for projection
python scripts/project_cerra.py --region central_europe --years 2021 --input-dir /path/to/custom/input

# Custom input directory for preprocessing
python scripts/preprocess_cerra.py --region central_europe --year 2021 --input-dir /path/to/custom/input
```

## Adding New Variables

To add new variables to the pipeline:

1. **Update configuration**: Add variables to `VARIABLE_GROUPS` in `config.py`
2. **Update preprocessing**: Modify `CERRAProcessor` class in `preprocess_cerra.py` to handle new variables
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
