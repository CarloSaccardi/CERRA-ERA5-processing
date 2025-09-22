#!/usr/bin/env python3
"""
CERRA Data Download Script

This script downloads CERRA reanalysis data from the Copernicus Climate Data Store (CDS).
It supports downloading different variable groups and years in a flexible manner.

Usage:
    python download_cerra.py --years 2014 2015 --variables base_variables
    python download_cerra.py --years 2020 --variables additional_variables
    python download_cerra.py --years 2014-2021 --variables all_variables
"""

import argparse
import cdsapi
import sys
from pathlib import Path
from typing import List, Union
import time

# Add the current directory to the path to import config
sys.path.append(str(Path(__file__).parent))
from config import DOWNLOAD_CONFIG, get_variable_group, DIRECTORIES


def parse_years(years_str: str) -> List[str]:
    """Parse years string into a list of year strings.
    
    Args:
        years_str: String like "2014,2015" or "2014-2021" or "2014"
        
    Returns:
        List of year strings
    """
    years = []
    
    # Handle comma-separated years
    if ',' in years_str:
        years = [y.strip() for y in years_str.split(',')]
    # Handle year ranges
    elif '-' in years_str:
        start_year, end_year = years_str.split('-')
        start_year = int(start_year.strip())
        end_year = int(end_year.strip())
        years = [str(year) for year in range(start_year, end_year + 1)]
    # Handle single year
    else:
        years = [years_str.strip()]
    
    return years


def create_download_request(years: List[str], variables: List[str]) -> dict:
    """Create a CDS API request dictionary.
    
    Args:
        years: List of year strings
        variables: List of variable names
        
    Returns:
        Request dictionary for CDS API
    """
    request = DOWNLOAD_CONFIG.copy()
    request["year"] = years
    request["variable"] = variables
    
    return request


def download_cerra_data(years: List[str], variables: List[str], output_dir: Path = None) -> None:
    """Download CERRA data from CDS.
    
    Args:
        years: List of year strings to download
        variables: List of variable names to download
        output_dir: Directory to save the downloaded files (optional)
    """
    if output_dir is None:
        output_dir = DIRECTORIES["lambert_proj"] / "single_levels"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CDS client
    client = cdsapi.Client()
    
    # Create request
    request = create_download_request(years, variables)
    
    print(f"Downloading CERRA data for years: {years}")
    print(f"Variables: {variables}")
    print(f"Output directory: {output_dir}")
    
    # Download data
    try:
        # Change to output directory for download
        original_cwd = Path.cwd()
        os.chdir(output_dir)
        
        # Generate filename based on years
        if len(years) == 1:
            filename = f"{years[0]}.grib"
        else:
            filename = f"{years[0]}-{years[-1]}.grib"
        
        print(f"Starting download to {filename}...")
        start_time = time.time()
        
        client.retrieve(DOWNLOAD_CONFIG["dataset"], request).download(filename)
        
        end_time = time.time()
        print(f"Download completed in {end_time - start_time:.2f} seconds")
        print(f"File saved as: {output_dir / filename}")
        
    except Exception as e:
        print(f"Error during download: {e}")
        raise
    finally:
        # Return to original directory
        os.chdir(original_cwd)


def main():
    """Main function to handle command line arguments and execute download."""
    parser = argparse.ArgumentParser(
        description="Download CERRA reanalysis data from CDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download base variables for 2014
  python download_cerra.py --years 2014 --variables base_variables
  
  # Download additional variables for multiple years
  python download_cerra.py --years 2014,2015,2016 --variables additional_variables
  
  # Download all variables for a year range
  python download_cerra.py --years 2014-2021 --variables all_variables
  
  # Download specific variables
  python download_cerra.py --years 2020 --variables 10m_wind_speed,2m_temperature
        """
    )
    
    parser.add_argument(
        "--years", 
        type=str, 
        required=True,
        help="Years to download (e.g., '2014', '2014,2015', '2014-2021')"
    )
    
    parser.add_argument(
        "--variables",
        type=str,
        default="base_variables",
        help="Variable group or comma-separated list of variables. "
             "Available groups: base_variables, additional_variables, all_variables"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for downloaded files (default: lambert_proj/single_levels)"
    )
    
    args = parser.parse_args()
    
    # Parse years
    years = parse_years(args.years)
    
    # Parse variables
    if args.variables in ["base_variables", "additional_variables", "all_variables"]:
        variables = get_variable_group(args.variables)
    else:
        # Assume comma-separated list of specific variables
        variables = [v.strip() for v in args.variables.split(',')]
    
    # Set output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    
    # Download data
    download_cerra_data(years, variables, output_dir)


if __name__ == "__main__":
    import os
    main()
