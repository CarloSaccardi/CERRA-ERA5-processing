"""
Configuration file for CERRA data processing pipeline.
This file contains all the configuration parameters for downloading, 
projecting, and preprocessing CERRA data.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent  # Go up one level from scripts/ to CERRA_download/
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Download configuration
DOWNLOAD_CONFIG = {
    "dataset": "reanalysis-cerra-single-levels",
    "level_type": "surface_or_atmosphere",
    "data_type": ["reanalysis"],
    "product_type": "forecast",
    "leadtime_hour": ["1"],
    "data_format": "grib",
    "time": [
        "00:00", "03:00", "06:00",
        "09:00", "12:00", "15:00",
        "18:00", "21:00"
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

# Variable groups
VARIABLE_GROUPS = {
    "base_variables": [
        "10m_wind_direction",
        "10m_wind_speed", 
        "2m_temperature",
    ],
    "all_variables": [
        "10m_wind_direction",
        "10m_wind_speed",
        "2m_temperature", 
        "momentum_flux_at_the_surface_u_component",
        "momentum_flux_at_the_surface_v_component",
        "surface_sensible_heat_flux",
        "surface_pressure",
        "2m_relative_humidity",
    ]
}

# Region configurations
REGIONS = {
    "central_europe": {
        "name": "CentralEurope",
        "coord_file": "cyl.txt",
        "description": "Central Europe region"
    },
    "iberia": {
        "name": "iberia", 
        "coord_file": "cyl_iberia.txt",
        "description": "Iberian Peninsula region"
    },
    "scandinavia": {
        "name": "scandinavia",
        "coord_file": "cyl_scandinavia.txt", 
        "description": "Scandinavia region"
    }
}

# Directory structure
DIRECTORIES = {
    "lambert_proj": BASE_DIR / "lambert_proj",
    "latlon_proj": BASE_DIR / "latlon_proj_{region}",
    "processed_data": DATA_DIR / "{region}" / "CERRA" / "samples",
    "coordinate_files": BASE_DIR / "config"
}

# Processing configuration
PROCESSING_CONFIG = {
    "target_variables": ["u10", "v10", "t2m", "sshf", "zust"],
    "time_step_hours": 3,
    "leap_years": [2008, 2012, 2016, 2020, 2024],
    "samples_per_year": {
        "leap": 2928,
        "normal": 2920
    }
}

# CDO remapping configuration
CDO_CONFIG = {
    "interpolation_method": "remapbil",
    "input_dir": "lambert_proj",
    "output_dir_template": "latlon_proj_{region}"
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
        "lambert_proj": DIRECTORIES["lambert_proj"],
        "latlon_proj": Path(str(DIRECTORIES["latlon_proj"]).format(region=region_name)),
        "processed_data": Path(str(DIRECTORIES["processed_data"]).format(region=region_name)),
        "coordinate_files": DIRECTORIES["coordinate_files"]
    }

def get_samples_per_year(year: int) -> int:
    """Get the number of samples for a given year."""
    if year in PROCESSING_CONFIG["leap_years"]:
        return PROCESSING_CONFIG["samples_per_year"]["leap"]
    else:
        return PROCESSING_CONFIG["samples_per_year"]["normal"]
