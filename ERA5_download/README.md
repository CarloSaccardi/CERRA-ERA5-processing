# ERA5 Data Processing Pipeline

This directory contains a complete pipeline for downloading and processing ERA5 reanalysis data from the Copernicus Climate Data Store (CDS). The pipeline is designed to match the CERRA processing workflow and provides a unified interface for meteorological data processing.

## 🏗️ Architecture

The ERA5 pipeline consists of several key components:

```
ERA5_download/
├── scripts/                    # Main pipeline scripts
│   ├── config.py              # Configuration and settings
│   ├── download_era5.py       # CDS API download script
│   ├── preprocess_era5.py     # NetCDF to HDF5 conversion
│   ├── era5_pipeline.py       # Main pipeline orchestrator
│   └── README.md              # Detailed script documentation
├── CentralEurope/             # Region-specific data
├── Iberia/                    # Region-specific data
├── Scandinavia/               # Region-specific data
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install cdsapi xarray h5py numpy
   ```

2. **Set up CDS API key:**
   - Register at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
   - Follow the [CDS API setup guide](https://cds.climate.copernicus.eu/api-how-to)
   - Place your `.cdsapirc` file in your home directory

### Basic Usage

```bash
# Full pipeline for Central Europe 2021
python scripts/era5_pipeline.py --region central_europe --years 2021

# Download only (skip preprocessing)
python scripts/era5_pipeline.py --region central_europe --years 2021 --skip-preprocessing

# Process multiple years
python scripts/era5_pipeline.py --region iberia --years 2020-2021 --variables all_variables
```

## 📊 Supported Regions

| Region | Geographic Coverage | API Area |
|--------|-------------------|----------|
| **Central Europe** | 60°N to 40°N, -2°W to 18°E | `[60, -2, 40, 18]` |
| **Iberia** | 45°N to 35°N, -10°W to 5°E | `[45, -10, 35, 5]` |
| **Scandinavia** | 72°N to 55°N, 5°E to 30°E | `[72, 5, 55, 30]` |

## 🔧 Variable Groups

### Base Variables (`base_variables`)
Essential meteorological variables for basic weather analysis:
- `u10`: 10m u-component of wind (m/s)
- `v10`: 10m v-component of wind (m/s)
- `t2m`: 2m temperature (K)

### All Variables (`all_variables`)
Complete set including surface fluxes and pressure:
- `u10`: 10m u-component of wind (m/s)
- `v10`: 10m v-component of wind (m/s)
- `t2m`: 2m temperature (K)
- `sshf`: Instantaneous surface sensible heat flux (W/m²)
- `zust`: Friction velocity (m/s)
- `sp`: Surface pressure (Pa)

## 📁 Data Flow

```
CDS API → NetCDF Files → HDF5 Files → PyTorch Ready
   ↓           ↓            ↓
Download → Preprocess → Final Output
```

### Input Format
- **Source**: Copernicus Climate Data Store (CDS)
- **Format**: NetCDF files via CDS API
- **Resolution**: 0.25° × 0.25° (approximately 25km)
- **Temporal**: 3-hourly data (8 times per day)

### Output Format
- **Format**: HDF5 files with metadata
- **Structure**: `(time, latitude, longitude, variables)`
- **Compatibility**: Matches CERRA pipeline output format
- **Coordinates**: Saved as `nwp_xy.npy` for consistency

## 🛠️ Pipeline Commands

### Full Pipeline
```bash
# Single year
python scripts/era5_pipeline.py --region central_europe --years 2021

# Multiple years (comma-separated)
python scripts/era5_pipeline.py --region iberia --years 2020,2021,2022

# Year range
python scripts/era5_pipeline.py --region scandinavia --years 2020-2021

# With all variables
python scripts/era5_pipeline.py --region central_europe --years 2021 --variables all_variables
```

### Skip Steps
```bash
# Download only
python scripts/era5_pipeline.py --region central_europe --years 2021 --skip-preprocessing

# Preprocess only (requires existing NetCDF files)
python scripts/era5_pipeline.py --region central_europe --years 2021 --skip-download
```

### Utility Commands
```bash
# Check prerequisites
python scripts/era5_pipeline.py --region central_europe --check-prerequisites

# Custom input directory
python scripts/era5_pipeline.py --region central_europe --years 2021 --skip-download --input-dir /path/to/netcdf/files
```

## 📂 Output Structure

```
data/
└── {region}/
    └── ERA5/
        └── samples/
            ├── {region}_{year}_era5.h5    # Main data file
            └── nwp_xy.npy                 # Coordinate grid
```

### HDF5 File Contents
- **`data`**: Main dataset with shape `(time, lat, lon, variables)`
- **`timestamps`**: Time stamps for each sample
- **`indices`**: Sample indices
- **`metadata`**: Variable names, processing info, and attributes

## 🔄 Comparison with CERRA Pipeline

| Feature | CERRA | ERA5 |
|---------|-------|------|
| **Data Source** | CDS (GRIB) | CDS (NetCDF) |
| **Geographic Selection** | Coordinate files | Direct API areas |
| **Projection** | Lambert → Cylindrical | Direct cylindrical |
| **Steps** | Download → Project → Preprocess | Download → Preprocess |
| **Output Format** | HDF5 | HDF5 (compatible) |
| **Command Interface** | `--skip-preprocessing` | `--skip-preprocessing` |

## ⚡ Performance Notes

- **Download Speed**: Depends on CDS server load and data size
- **Processing**: NetCDF to HDF5 conversion is typically fast
- **Storage**: HDF5 format provides efficient compression
- **Memory**: Processing is optimized for large datasets

## 🐛 Troubleshooting

### Common Issues

1. **CDS API Authentication Error**
   ```bash
   # Check your .cdsapirc file
   cat ~/.cdsapirc
   ```

2. **Missing Dependencies**
   ```bash
   pip install cdsapi xarray h5py numpy
   ```

3. **File Not Found Errors**
   - Ensure input files exist in the specified directory
   - Check file naming convention: `single_{year}.nc`

4. **Memory Issues**
   - Process smaller year ranges
   - Use `--skip-download` to process existing files separately

### Getting Help

- Check the detailed [scripts documentation](scripts/README.md)
- Verify prerequisites with `--check-prerequisites`
- Review error messages for specific guidance

## 📈 Example Workflows

### Research Workflow
```bash
# 1. Download multiple years
python scripts/era5_pipeline.py --region central_europe --years 2020-2021 --skip-preprocessing

# 2. Process with different variable sets
python scripts/era5_pipeline.py --region central_europe --years 2020-2021 --skip-download --variables base_variables
python scripts/era5_pipeline.py --region central_europe --years 2020-2021 --skip-download --variables all_variables
```

### Development Workflow
```bash
# 1. Test with single year
python scripts/era5_pipeline.py --region iberia --years 2021

# 2. Scale to multiple years
python scripts/era5_pipeline.py --region iberia --years 2020-2021
```

## 🔗 Integration

The ERA5 pipeline is designed to work seamlessly with:
- **PyTorch DataLoaders**: HDF5 format is optimized for ML workflows
- **CERRA Pipeline**: Compatible output format and structure
- **Existing Analysis Tools**: Standard meteorological data formats

## 📝 License

This pipeline is part of the CERRA-ERA5 processing project. Please refer to the main project documentation for licensing information.
