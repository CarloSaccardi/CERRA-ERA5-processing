#!/usr/bin/env python3
"""
CERRA Data Projection Script

This script converts CERRA data from Lambert conformal projection to 
cylindrical (lat-lon) projection for specific regions using CDO.

Usage:
    python project_cerra.py --region central_europe --years 2014 2015
    python project_cerra.py --region iberia --years 2020
    python project_cerra.py --region scandinavia --input-file 2021.grib
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import time
import tempfile

# Add the current directory to the path to import config
sys.path.append(str(Path(__file__).parent))
from config import get_region_config, get_output_paths, CDO_CONFIG


def check_cdo_available() -> bool:
    """Check if CDO is available in the system."""
    try:
        subprocess.run(["cdo", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_input_files(input_dir: Path, years: Optional[List[str]] = None, 
                   input_file: Optional[str] = None) -> List[Path]:
    """Get list of input files to process.
    
    Args:
        input_dir: Directory containing input files
        years: List of years to process (optional)
        input_file: Specific file to process (optional)
        
    Returns:
        List of input file paths
    """
    if input_file:
        # Process specific file
        file_path = input_dir / input_file
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        return [file_path]
    
    # Find files based on years
    if years:
        files = []
        for year in years:
            pattern = f"{year}.grib"
            file_path = input_dir / pattern
            if file_path.exists():
                files.append(file_path)
            else:
                print(f"Warning: File not found for year {year}: {file_path}")
        return files
    else:
        # Find all *.grib files
        return list(input_dir.glob("*.grib"))


def run_cdo_remap(input_file: Path, output_file: Path, coord_file: Path) -> None:
    """Run CDO remapbil command to convert projection.
    
    Args:
        input_file: Input GRIB file
        output_file: Output GRIB file
        coord_file: Coordinate file for target grid
    """
    
    remap_argument = f'{CDO_CONFIG["interpolation_method"]},{str(coord_file)}'
    
    cmd = [
        "cdo", 
        remap_argument,
        str(input_file),
        str(output_file)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        print(f"Projection completed in {end_time - start_time:.2f} seconds")
        print(f"Output saved to: {output_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running CDO command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


def project_cerra_data(region: str, years: Optional[List[str]] = None, 
                      input_file: Optional[str] = None, 
                      input_dir: Optional[Path] = None) -> None:
    """Project CERRA data from Lambert to cylindrical projection.
    
    Args:
        region: Target region name
        years: List of years to process (optional)
        input_file: Specific file to process (optional)
        input_dir: Input directory (optional, defaults to lambert_proj/single_levels)
    """
    # Check if CDO is available
    if not check_cdo_available():
        raise RuntimeError("CDO (Climate Data Operators) is not available. Please install CDO.")
    
    # Get region configuration
    region_config = get_region_config(region)
    coord_file_name = region_config["coord_file"]
    
    # Get paths
    paths = get_output_paths(region)
    
    # Set input directory
    if input_dir is None:
        input_dir = paths["lambert_proj"] / "single_levels"
        input_dir_humidity = paths["lambert_proj"] / "single_levels_humidity"
    
    # Get coordinate file
    coord_file = paths["coordinate_files"] / coord_file_name
    if not coord_file.exists():
        raise FileNotFoundError(f"Coordinate file not found: {coord_file}")
    
    # Create output directory
    output_dir = paths["latlon_proj"] / "single_levels"
    output_dir_humidity = paths["latlon_proj"] / "single_levels_humidity"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_humidity.mkdir(parents=True, exist_ok=True)
    
    # Get input files
    input_files = get_input_files(input_dir, years, input_file)
    input_files_humidity = get_input_files(input_dir_humidity, years, input_file)
    
    if not input_files:
        print("No input files found to process.")
        return
    
    print(f"Processing {len(input_files)} files for region: {region}")
    print(f"Coordinate file: {coord_file}")
    print(f"Output directory: {output_dir}")
    
    # Process each file
    for input_file_path in input_files:
        # Generate output filename
        output_filename = input_file_path.name
        output_file_path = output_dir / output_filename
        
        # Skip if output file already exists
        if output_file_path.exists():
            print(f"Skipping {input_file_path.name} (output already exists)")
            continue
        
        print(f"\nProcessing: {input_file_path.name}")
        run_cdo_remap(input_file_path, output_file_path, coord_file)
        
    # Process humidity files if they exist
    for input_file_path in input_files_humidity:
        # Generate output filename
        output_filename = input_file_path.name
        output_file_path = output_dir_humidity / output_filename
        
        # Skip if output file already exists
        if output_file_path.exists():
            print(f"Skipping {input_file_path.name} (output already exists)")
            continue
        
        print(f"\nProcessing humidity file: {input_file_path.name}")
        run_cdo_remap(input_file_path, output_file_path, coord_file)
        
        
def project_static_variables(region: str) -> None:
    """
    Projects the static orography variable to the specified region's grid.
    """
    if not check_cdo_available():
        raise RuntimeError("CDO (Climate Data Operators) is not available. Please install CDO.")
    
    # Get region configuration
    region_config = get_region_config(region)
    paths = get_output_paths(region)

    input_file_path = paths["lambert_proj"] / "single_levels" / "orography.grib"
    output_file_path = paths["latlon_proj"] / "single_levels" / "orography.grib"

    # Get the coordinate file path correctly and robustly
    coord_file_name = region_config["coord_file"]
    coord_file_path = paths["coordinate_files"] / coord_file_name

    # Call CDO wrapper function
    print(f"Projecting orography to {region}...")
    try:
        # Note: Your run_cdo_remap function already handles converting paths to strings
        run_cdo_remap(input_file_path, output_file_path, coord_file_path)
    except Exception as e:
        print(f"An error occurred during the projection of static variables: {e}")


def main():
    """Main function to handle command line arguments and execute projection."""
    parser = argparse.ArgumentParser(
        description="Project CERRA data from Lambert to cylindrical projection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Project data for central Europe for specific years
  python project_cerra.py --region central_europe --years 2014 2015
  
  # Project data for Iberia for a single year
  python project_cerra.py --region iberia --years 2020
  
  # Project a specific file for Scandinavia
  python project_cerra.py --region scandinavia --input-file 2021.grib
  
  # Project all available files for central Europe
  python project_cerra.py --region central_europe
        """
    )
    
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        choices=["central_europe", "iberia", "scandinavia"],
        help="Target region for projection"
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--years",
        nargs="+",
        help="Years to process (e.g., 2014 2015 2016)"
    )
    
    group.add_argument(
        "--input-file",
        type=str,
        help="Specific input file to process"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory (default: lambert_proj/single_levels)"
    )
    
    parser.add_argument(
        '--static-only', 
        type=bool,
        default=False, 
        help='Process only static variables (orography)'
    )
    
    parser.add_argument(
        '--time-varying-only', 
        type=bool,
        default=False, 
        help='Process only time-varying variables'
    )
    
    args = parser.parse_args()
    
    # Set input directory
    input_dir = None
    if args.input_dir:
        input_dir = Path(args.input_dir)
    
    if args.static_only:
        print("Processing only static variables (orography)...")
        project_static_variables(
            region=args.region
            )
        
        
    if args.time_varying_only:
        # Project time-dependent data
        print("Processing only time-varying variables...")
        project_cerra_data(
            region=args.region,
            years=args.years,
            input_file=args.input_file,
            input_dir=input_dir
        )
        
    if not args.static_only and not args.time_varying_only:
        raise ValueError("Please specify either --static-only or --time-varying-only.")
    


if __name__ == "__main__":
    main()
