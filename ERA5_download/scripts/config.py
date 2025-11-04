"""
Configuration file for ERA5 data processing pipeline.
This file contains all the configuration parameters for downloading and preprocessing ERA5 data.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent  # Go up one level from scripts/ to ERA5_download/
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Download configuration
DOWNLOAD_CONFIG = {
    "dataset": "reanalysis-era5-single-levels",
    "product_type": ["reanalysis"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "time": [
        "01:00", "04:00", "07:00",
        "10:00", "13:00", "16:00",
        "19:00", "22:00"
    ],
    "month": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12"
    ],
    "day": [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
    ]
}

# Variable groups - matching the single_level.py variables
VARIABLE_GROUPS = {
    "base_variables": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind", 
        "2m_temperature",
    ],
    "all_variables": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature", 
        "instantaneous_surface_sensible_heat_flux",
        "friction_velocity",
        "surface_pressure"
    ]
}

# Region configurations with geographic boundaries
REGIONS = {
    "central_europe": {
        "name": "CentralEurope",
        "area": [60, -2, 40, 18],  # [North, West, South, East]
        "description": "Central Europe region"
    },
    "iberia": {
        "name": "Iberia", 
        "area": [45, -15, 25, 5],  # [North, West, South, East]
        "description": "Iberian Peninsula region"
    },
    "scandinavia": {
        "name": "Scandinavia",
        "area": [70, 24, 50, 44],  # [North, West, South, East]
        "description": "Scandinavia region"
    },
    "eurasia": {
        "name": "Eurasia",
        "area": [78, -20, 14, 80],  # [North, West, South, East]
        "description": "Entire Europe region"
    }
}

# Directory structure
DIRECTORIES = {
    "raw_data": BASE_DIR / "{region}" / "single_levels",
    "processed_data": DATA_DIR / "{region}" / "ERA5" / "samples",
    "coordinate_files": BASE_DIR / "config"
}

# Processing configuration
PROCESSING_CONFIG = {
    "target_variables": ["u10", "v10", "t2m", "sshf", "zust", "sp"],
    "time_step_hours": 3,
    "leap_years": [2008, 2012, 2016, 2020, 2024],
    "samples_per_year": {
        "leap": 2928,
        "normal": 2920
    }
}

def get_region_config(region: str) -> Dict:
    """Get configuration for a specific region."""
    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}. Available regions: {list(REGIONS.keys())}")
    return REGIONS[region]

def get_variable_group(group_name: str) -> List[str]:
    """Get variables for a specific group."""
    if group_name not in VARIABLE_GROUPS:
        raise ValueError(f"Unknown variable group: {group_name}. Available groups: {list(VARIABLE_GROUPS.keys())}")
    return VARIABLE_GROUPS[group_name]

def get_output_paths(region: str) -> Dict[str, Path]:
    """Get output paths for a specific region."""
    region_config = get_region_config(region)
    region_name = region_config["name"]
    
    return {
        "raw_data": Path(str(DIRECTORIES["raw_data"]).format(region=region_name)),
        "processed_data": Path(str(DIRECTORIES["processed_data"]).format(region=region_name)),
        "coordinate_files": DIRECTORIES["coordinate_files"]
    }

def get_samples_per_year(year: int) -> int:
    """Get the number of samples for a given year."""
    if year in PROCESSING_CONFIG["leap_years"]:
        return PROCESSING_CONFIG["samples_per_year"]["leap"]
    else:
        return PROCESSING_CONFIG["samples_per_year"]["normal"]
