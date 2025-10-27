# CERRA Data Splitting Script

This script creates train/validation/test splits from processed CERRA data across multiple regions, following the exact same strategy as the ERA5 data splitting.

## Overview

The script splits CERRA data into three sets:
- **Training**: CE(2013-2015) + Iberia(2015-2017) + Scandinavia(2017-2019) + All regions (odd months 2020)
- **Validation**: All regions (even months 2020)  
- **Test**: All regions (all months 2021)

## File Structure

### Input Structure
```
CERRA_processing/
├── latlon_proj_CentralEurope/single_levels_processed/
│   ├── CentralEurope_2013_cerra.nc
│   ├── CentralEurope_2014_cerra.nc
│   ├── CentralEurope_2015_cerra.nc
│   ├── CentralEurope_2020_cerra.nc
│   ├── CentralEurope_2021_cerra.nc
│   └── CentralEurope_static_cerra.nc
├── latlon_proj_iberia/single_levels_processed/
│   ├── iberia_2015_cerra.nc
│   ├── iberia_2016_cerra.nc
│   ├── iberia_2017_cerra.nc
│   ├── iberia_2020_cerra.nc
│   ├── iberia_2021_cerra.nc
│   └── iberia_static_cerra.nc
└── latlon_proj_scandinavia/single_levels_processed/
    ├── scandinavia_2017_cerra.nc
    ├── scandinavia_2018_cerra.nc
    ├── scandinavia_2019_cerra.nc
    ├── scandinavia_2020_cerra.nc
    ├── scandinavia_2021_cerra.nc
    └── scandinavia_static_cerra.nc
```

### Output Structure
```
zz_processed_data/CERRA/
├── train/
│   ├── CentralEurope.nc
│   ├── Iberia.nc
│   ├── Scandinavia.nc
│   ├── static_CentralEurope.nc
│   ├── static_Iberia.nc
│   └── static_Scandinavia.nc
├── val/
│   ├── CentralEurope.nc
│   ├── Iberia.nc
│   ├── Scandinavia.nc
│   ├── static_CentralEurope.nc
│   ├── static_Iberia.nc
│   └── static_Scandinavia.nc
└── test/
    ├── CentralEurope.nc
    ├── Iberia.nc
    ├── Scandinavia.nc
    ├── static_CentralEurope.nc
    ├── static_Iberia.nc
    └── static_Scandinavia.nc
```

## Usage

### Basic Usage
```bash
# Use the example configuration
python split_cerra_data.py --config split_config_example.json
```

### Advanced Usage
```bash
# Custom input/output directories
python split_cerra_data.py --config split_config_example.json \
    --input-dir /path/to/CERRA_processing \
    --output-dir /path/to/output

# Create only training data
python split_cerra_data.py --config split_config_example.json \
    --skip-val --skip-test --skip-static

# Higher compression
python split_cerra_data.py --config split_config_example.json \
    --compression-level 9

# For forecasting task (will show chunking guidance)
python split_cerra_data.py --config split_config_example.json \
    --task forecasting
```

## Configuration Format

The configuration file follows the same JSON structure as ERA5:

```json
{
  "train": {
    "CentralEurope": {
      "directory": "latlon_proj_CentralEurope/single_levels_processed",
      "years": [2013, 2014, 2015],
      "year_months": {
        "2020": [1, 3, 5, 7, 9, 11]
      }
    },
    "Iberia": {
      "directory": "latlon_proj_iberia/single_levels_processed",
      "years": [2015, 2016, 2017],
      "year_months": {
        "2020": [1, 3, 5, 7, 9, 11]
      }
    },
    "Scandinavia": {
      "directory": "latlon_proj_scandinavia/single_levels_processed",
      "years": [2017, 2018, 2019],
      "year_months": {
        "2020": [1, 3, 5, 7, 9, 11]
      }
    }
  },
  "val": {
    "CentralEurope": {
      "directory": "latlon_proj_CentralEurope/single_levels_processed",
      "year_months": {
        "2020": [2, 4, 6, 8, 10, 12]
      }
    },
    "Iberia": {
      "directory": "latlon_proj_iberia/single_levels_processed",
      "year_months": {
        "2020": [2, 4, 6, 8, 10, 12]
      }
    },
    "Scandinavia": {
      "directory": "latlon_proj_scandinavia/single_levels_processed",
      "year_months": {
        "2020": [2, 4, 6, 8, 10, 12]
      }
    }
  },
  "test": {
    "CentralEurope": {
      "directory": "latlon_proj_CentralEurope/single_levels_processed",
      "years": [2021]
    },
    "Iberia": {
      "directory": "latlon_proj_iberia/single_levels_processed",
      "years": [2021]
    },
    "Scandinavia": {
      "directory": "latlon_proj_scandinavia/single_levels_processed",
      "years": [2021]
    }
  }
}
```

## Key Features

### 1. **Identical Split Strategy to ERA5**
- Same temporal splits across regions
- Same month-based validation strategy
- Same test year (2021)

### 2. **File Naming Convention**
- **Data files**: `{region}_{year}_cerra.nc`
- **Static files**: `{region}_static_cerra.nc`
- **Output files**: `{region}.nc` and `static_{region}.nc`

### 3. **Chunking Strategy**
- **Downscaling**: Time chunk size = 1 (optimal for random access)
- **Forecasting**: Raises error with guidance for proper chunking

### 4. **Compression**
- NetCDF4 format with zlib compression
- Configurable compression level (1-9, default: 4)
- Float32 data type for efficiency

## Differences from ERA5 Script

1. **File naming**: Uses `_cerra.nc` suffix instead of `_era5.nc`
2. **Directory structure**: Uses `latlon_proj_{region}` instead of `{region}`
3. **Region names**: Uses `Iberia` and `Scandinavia` (lowercase in directories)
4. **Output path**: Defaults to `../zz_processed_data/CERRA`

## Supervised Learning Alignment

This script ensures that CERRA data (target) is split **exactly** the same way as ERA5 data (input), which is crucial for supervised learning:

- **Same temporal splits**: Training/validation/test periods match
- **Same regional coverage**: All regions in both datasets
- **Same file structure**: Easy to pair ERA5 and CERRA files
- **Same metadata**: Consistent attributes and naming

## Error Handling

The script includes comprehensive error handling:
- Validates configuration file format
- Checks for required directories and files
- Provides clear error messages
- Graceful handling of missing data

## Performance Considerations

- **Memory efficient**: Processes one region at a time
- **Chunked I/O**: Optimized NetCDF chunking for downscaling
- **Compression**: Reduces file sizes significantly
- **Progress reporting**: Shows processing status and file sizes
