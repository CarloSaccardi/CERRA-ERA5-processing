#!/usr/bin/env python3
"""
CERRA Data Projection Script

This script converts CERRA data from Lambert conformal projection to 
cylindrical (lat-lon) projection for specific regions using CDO.

Usage:
    # Project time-varying data
    python project_cerra.py --region central_europe --years 2014 2015 \\
        --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity
    
    # Project static variables
    python project_cerra.py --region central_europe \\
        --remap_directories lambert_proj/single_levels_static
    
    # Project everything
    python project_cerra.py --region central_europe --years 2014 2015 \\
        --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity lambert_proj/single_levels_static
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


def determine_output_dir(input_dir: Path, region_name: str, base_path: Path) -> Path:
    """Determine output directory based on input directory structure.
    
    Args:
        input_dir: Input directory path
        region_name: Name of the region (e.g., 'CentralEurope')
        base_path: Base path for lambert_proj
        
    Returns:
        Corresponding output directory path
    """
    # Get relative path from lambert_proj parent
    try:
        relative_path = input_dir.relative_to(base_path.parent)
    except ValueError:
        # If absolute path doesn't contain lambert_proj parent, use input_dir as is
        relative_path = input_dir
    
    # Replace lambert_proj with latlon_proj_{region}
    output_path_str = str(relative_path).replace("lambert_proj", f"latlon_proj_{region_name}")
    output_dir = base_path.parent / output_path_str
    
    return output_dir


def project_cerra_data(region: str, 
                      remap_directories: List[Path],
                      years: Optional[List[str]] = None, 
                      input_file: Optional[str] = None) -> None:
    """Project CERRA data from Lambert to cylindrical projection.
    
    Args:
        region: Target region name
        remap_directories: List of input directories to process
        years: List of years to process (optional)
        input_file: Specific file to process (optional)
    """
    # Check if CDO is available
    if not check_cdo_available():
        raise RuntimeError("CDO (Climate Data Operators) is not available. Please install CDO.")
    
    # Get region configuration
    region_config = get_region_config(region)
    coord_file_name = region_config["coord_file"]
    region_name = region_config["name"]
    
    # Get paths
    paths = get_output_paths(region)
    
    # Get coordinate file
    coord_file = paths["coordinate_files"] / coord_file_name
    if not coord_file.exists():
        raise FileNotFoundError(f"Coordinate file not found: {coord_file}")
    
    print(f"Region: {region}")
    print(f"Coordinate file: {coord_file}")
    print(f"Processing {len(remap_directories)} directories")
    
    # Process each input directory
    for input_dir in remap_directories:
        print(f"\n{'='*60}")
        print(f"Processing directory: {input_dir}")
        print(f"{'='*60}")
        
        # Check if directory exists
        if not input_dir.exists():
            print(f"Warning: Directory not found: {input_dir}")
            continue
        
        # Determine output directory
        output_dir = determine_output_dir(input_dir, region_name, paths["lambert_proj"])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Get input files
        input_files = get_input_files(input_dir, years, input_file)
        
        if not input_files:
            print(f"No input files found in {input_dir}")
            continue
        
        print(f"Found {len(input_files)} file(s) to process")
        
        # Process each file
        for input_file_path in input_files:
            output_filename = input_file_path.name
            output_file_path = output_dir / output_filename
            
            # Skip if output file already exists
            if output_file_path.exists():
                print(f"  Skipping {input_file_path.name} (output already exists)")
                continue
            
            print(f"\n  Processing: {input_file_path.name}")
            run_cdo_remap(input_file_path, output_file_path, coord_file)
        


def main():
    """Main function to handle command line arguments and execute projection."""
    parser = argparse.ArgumentParser(
        description="Project CERRA data from Lambert to cylindrical projection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Project time-varying data for specific years
  python project_cerra.py --region central_europe --years 2014 2015 \\
    --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity
  
  # Project static variables only
  python project_cerra.py --region central_europe \\
    --remap_directories lambert_proj/single_levels_static
  
  # Project everything for specific years
  python project_cerra.py --region central_europe --years 2014 2015 \\
    --remap_directories lambert_proj/single_levels lambert_proj/single_levels_humidity lambert_proj/single_levels_static
  
  # Project a specific file
  python project_cerra.py --region scandinavia --input-file 2021.grib \\
    --remap_directories lambert_proj/single_levels
        """
    )
    
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        choices=["central_europe", "iberia", "scandinavia"],
        help="Target region for projection"
    )
    
    parser.add_argument(
        "--remap_directories",
        nargs="+",
        type=str,
        required=True,
        help="List of input directories to remap (e.g., lambert_proj/single_levels)"
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
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    remap_dirs = [Path(d) for d in args.remap_directories]
    
    # Execute projection
    project_cerra_data(
        region=args.region,
        remap_directories=remap_dirs,
        years=args.years,
        input_file=args.input_file
    )
    


if __name__ == "__main__":
    main()
