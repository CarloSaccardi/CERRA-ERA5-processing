# ERA5 Data Splitting Tool

## Overview

`split_data.py` is a flexible tool for creating train/validation/test splits from processed ERA5 data across multiple regions. The script combines data from different regions into unified NetCDF files with an added `region` dimension.

## Key Features

- **Explicit Region Control**: Specify exact directories for each region in your configuration
- **Flexible Configuration**: JSON config files with built-in defaults
- **Selective Processing**: Process only the regions you need
- **Future-Proof**: Easy to add new regions without modifying code
- **Configurable Month/Year Selection**: Specify full years or specific months from specific years
- **CF-Compliant Output**: Produces properly formatted NetCDF4 files with metadata
- **Compression**: Configurable compression levels for storage efficiency

## Directory Structure Expected

The script expects your data organized as:

```
ERA5_download/
├── CentralEurope/
│   └── single_levels_processed/
│       ├── CentralEurope_2013_era5.nc
│       ├── CentralEurope_2014_era5.nc
│       ├── ...
│       └── CentralEurope_static_era5.nc
├── Iberia/
│   └── single_levels_processed/
│       ├── Iberia_2015_era5.nc
│       ├── ...
│       └── Iberia_static_era5.nc
└── Scandinavia/
    └── single_levels_processed/
        ├── Scandinavia_2017_era5.nc
        ├── ...
        └── Scandinavia_static_era5.nc
```

## Output Structure

The script creates NetCDF files with the following structure:

- **train.nc**: `(region=N, time=T_train, lat, lon)`
- **val.nc**: `(region=N, time=T_val, lat, lon)`
- **test.nc**: `(region=N, time=T_test, lat, lon)`
- **static.nc**: `(region=N, lat, lon)`

Where `N` is the number of regions and `T` is the number of timesteps (varies by region and split).

## Usage

### Basic Usage (Default Configuration)

```bash
# Use default configuration for CentralEurope, Iberia, Scandinavia
python split_data.py
```

This uses the built-in default configuration:
- **Training**: CE(2013-2015) + Iberia(2015-2017) + Scandinavia(2017-2019) + All regions(odd months 2020)
- **Validation**: All regions (even months 2020)
- **Test**: All regions (all months 2021)

### Custom Configuration File

```bash
# Use custom configuration
python split_data.py --config my_config.json

# Specify custom input/output directories
python split_data.py --config my_config.json \
  --input-dir /path/to/ERA5_download \
  --output-dir /path/to/output
```

To create a custom configuration, copy `split_config_example.json` and modify it for your needs.

### Advanced Options

```bash
# Create only training data
python split_data.py --skip-val --skip-test --skip-static

# Use higher compression (slower but smaller files)
python split_data.py --compression-level 9

# Use lower compression (faster but larger files)
python split_data.py --compression-level 3
```

## Configuration Format

The configuration file is a JSON file with the following structure:

```json
{
  "train": {
    "RegionName": {
      "directory": "RegionName/single_levels_processed",
      "years": [2013, 2014, 2015],
      "year_months": {
        "2020": [1, 3, 5, 7, 9, 11]
      }
    },
    "AnotherRegion": {
      "directory": "AnotherRegion/single_levels_processed",
      "years": [2015, 2016]
    }
  },
  "val": {
    "RegionName": {
      "directory": "RegionName/single_levels_processed",
      "year_months": {
        "2020": [2, 4, 6, 8, 10, 12]
      }
    }
  },
  "test": {
    "RegionName": {
      "directory": "RegionName/single_levels_processed",
      "years": [2021]
    }
  }
}
```

### Configuration Fields

For each region in each split, you must specify:

- **`directory`** (required): Relative path from `--input-dir` to the processed data directory
  - Example: `"CentralEurope/single_levels_processed"`
  - This gives you full control over which directories to process

And at least one of:

- **`years`** (optional): List of years to include in full
  - Example: `[2013, 2014, 2015]` includes all data from these years

- **`year_months`** (optional): Dictionary mapping years to lists of months
  - Example: `{"2020": [1, 3, 5]}` includes only January, March, May from 2020

You can use both fields together, and the data will be concatenated in the order: full years first, then year_months.

## Adding New Regions

To add a new region, simply:

1. Place your processed data in any directory structure you prefer

2. Create or modify your configuration file to include the new region with its directory:
   ```json
   {
     "train": {
       "NewRegion": {
         "directory": "NewRegion/single_levels_processed",
         "years": [2018, 2019, 2020]
       }
     },
     ...
   }
   ```

3. Run the script with your updated configuration

No code changes needed! The explicit `directory` field gives you complete flexibility over where your data is located.

## Example Configurations

### Example 1: Simple Full-Year Split

```json
{
  "train": {
    "CentralEurope": {
      "directory": "CentralEurope/single_levels_processed",
      "years": [2013, 2014, 2015]
    },
    "Iberia": {
      "directory": "Iberia/single_levels_processed",
      "years": [2015, 2016, 2017]
    }
  },
  "val": {
    "CentralEurope": {
      "directory": "CentralEurope/single_levels_processed",
      "years": [2016]
    },
    "Iberia": {
      "directory": "Iberia/single_levels_processed",
      "years": [2018]
    }
  },
  "test": {
    "CentralEurope": {
      "directory": "CentralEurope/single_levels_processed",
      "years": [2021]
    },
    "Iberia": {
      "directory": "Iberia/single_levels_processed",
      "years": [2021]
    }
  }
}
```

### Example 2: Month-Level Split

```json
{
  "train": {
    "MyRegion": {
      "directory": "MyRegion/single_levels_processed",
      "year_months": {
        "2020": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      }
    }
  },
  "val": {
    "MyRegion": {
      "directory": "MyRegion/single_levels_processed",
      "year_months": {
        "2020": [11]
      }
    }
  },
  "test": {
    "MyRegion": {
      "directory": "MyRegion/single_levels_processed",
      "year_months": {
        "2020": [12]
      }
    }
  }
}
```

### Example 3: Single Region

```json
{
  "train": {
    "CentralEurope": {
      "directory": "CentralEurope/single_levels_processed",
      "years": [2013, 2014, 2015, 2016, 2017, 2018, 2019]
    }
  },
  "val": {
    "CentralEurope": {
      "directory": "CentralEurope/single_levels_processed",
      "years": [2020]
    }
  },
  "test": {
    "CentralEurope": {
      "directory": "CentralEurope/single_levels_processed",
      "years": [2021]
    }
  }
}
```

### Example 4: Processing Only Specific Regions

```json
{
  "train": {
    "CentralEurope": {
      "directory": "CentralEurope/single_levels_processed",
      "years": [2013, 2014, 2015]
    }
  },
  "val": {
    "CentralEurope": {
      "directory": "CentralEurope/single_levels_processed",
      "years": [2020]
    }
  },
  "test": {
    "CentralEurope": {
      "directory": "CentralEurope/single_levels_processed",
      "years": [2021]
    }
  }
}
```

Note: Only the regions specified in the configuration will be processed. This allows you to selectively work with subsets of your data.

## Output Metadata

Each output file includes comprehensive metadata:
- Region names and configuration used
- Creation date and script version
- CF-compliant variable attributes
- Coordinate information

## Performance Considerations

- **Compression Level**: Level 6 (default) is a good balance between speed and file size
  - Lower (1-3): Faster processing, larger files
  - Higher (7-9): Slower processing, smaller files
  
- **Chunking**: The script automatically chunks data as `(1, 365, lat, lon)` for efficient region-wise access

- **Memory**: The script processes one split at a time and closes datasets after writing

## Troubleshooting

### "Directory for region not found"
- Check that the `directory` field in your configuration is correct
- Verify the path is relative to `--input-dir`
- Ensure directory permissions are correct
- The directory path is case-sensitive

### "Required file not found"
- Ensure all year files referenced in configuration exist
- Check filename format: `{RegionName}_{Year}_era5.nc`
- Verify the region name in the filename matches your configuration

### "Missing required 'directory' field"
- Each region in the configuration must have a `directory` field
- See `split_config_example.json` for the correct format

### Configuration validation errors
- Ensure your JSON is valid (use a JSON validator)
- Check that all required keys ('train', 'val', 'test') are present
- Each region must have a `directory` field
- Each region must have either `years` or `year_months` (or both)
- Verify month numbers are 1-12
- Verify year values are integers

## Command-Line Reference

```
usage: split_data.py [-h] [--config CONFIG]
                     [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
                     [--compression-level {1,2,3,4,5,6,7,8,9}]
                     [--skip-train] [--skip-val] [--skip-test]
                     [--skip-static]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to JSON configuration file (default: use
                        built-in default config)
  --input-dir INPUT_DIR
                        Root directory containing regional ERA5 data
                        subdirectories (default: ../ERA5_download relative
                        to script)
  --output-dir OUTPUT_DIR
                        Directory to save split datasets (default:
                        INPUT_DIR/splits)
  --compression-level {1,2,3,4,5,6,7,8,9}
                        NetCDF compression level (1-9, default: 6)
  --skip-train          Skip creating training dataset
  --skip-val            Skip creating validation dataset
  --skip-test           Skip creating test dataset
  --skip-static         Skip creating static dataset
```

## Dependencies

- Python 3.7+
- xarray
- numpy
- netCDF4

## See Also

- `preprocess_era5.py`: For preprocessing raw ERA5 data
- `split_config_example.json`: Example configuration file


