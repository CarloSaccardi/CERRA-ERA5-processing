#!/usr/bin/env python3
"""
CERRA Data Projection Script

This script converts CERRA data from Lambert conformal projection to 
cylindrical (lat-lon) projection for specific regions using CDO.

The script automatically handles:
- Multiple regions (if --region not specified, projects all regions)
- Year files (named YYYY.grib, e.g., 2014.grib, 2015.grib)
- Static files (e.g., orography.grib, any non-year named files)

Output files are saved as compressed NetCDF4 format (.nc) with the same base name.

Usage:
    # Project ALL regions for specific years
    python project_cerra.py --years 2014 2015 \\
        --input_directories single_levels single_levels_humidity
    
    # Project specific region for specific years
    python project_cerra.py --region central_europe --years 2014 2015 \\
        --input_directories single_levels single_levels_humidity
    
    # Project ALL regions, all years + static files
    python project_cerra.py \\
        --input_directories single_levels single_levels_static
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import time
import os 

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


def is_year_file(filename: str) -> bool:
    """Check if a filename represents a year file.
    
    Args:
        filename: Name of the file (e.g., '2021.grib', 'orography.grib')
        
    Returns:
        True if filename is a year (1900-2100), False otherwise
    """
    stem = Path(filename).stem
    try:
        year = int(stem)
        return 1900 <= year <= 2100
    except ValueError:
        return False


def get_year_files(input_dir: Path, years: Optional[List[str]] = None) -> List[Path]:
    """Get list of year-based files to process.
    
    Args:
        input_dir: Directory containing input files
        years: List of years to process (optional, if None finds all year files)
        
    Returns:
        List of year file paths
    """
    if years:
        # Process specific years
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
        # Find all year files (files named like YYYY.grib)
        all_grib_files = list(input_dir.glob("*.grib"))
        year_files = [f for f in all_grib_files if is_year_file(f.name)]
        return sorted(year_files)


def get_static_files(input_dir: Path) -> List[Path]:
    """Get list of static (non-year) files to process.
    
    Args:
        input_dir: Directory containing input files
        
    Returns:
        List of static file paths (e.g., orography.grib)
    """
    all_grib_files = list(input_dir.glob("*.grib"))
    static_files = [f for f in all_grib_files if not is_year_file(f.name)]
    return sorted(static_files)


def get_input_files(input_dir: Path, years: Optional[List[str]] = None, 
                   input_file: Optional[str] = None, include_static: bool = True) -> List[Path]:
    """Get list of input files to process.
    
    Args:
        input_dir: Directory containing input files
        years: List of years to process (optional, if None processes all years)
        input_file: Specific file to process (optional)
        include_static: Whether to include static files (default: True)
        
    Returns:
        List of input file paths (year files + static files)
    """
    if input_file:
        # Process specific file
        file_path = input_dir / input_file
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        return [file_path]
    
    # Get year files
    year_files = get_year_files(input_dir, years)
    
    # Get static files if requested
    if include_static:
        static_files = get_static_files(input_dir)
        return year_files + static_files
    else:
        return year_files


def run_cdo_remap(input_file: Path, output_file: Path, coord_file: Path) -> None:
    """Run CDO remapbil command to convert projection.
    
    Args:
        input_file: Input GRIB file
        output_file: Output NetCDF4 file (compressed)
        coord_file: Coordinate file for target grid
    """
    
    remap_argument = f'{CDO_CONFIG["interpolation_method"]},{str(coord_file)}'
    
    cmd = [
        "cdo", 
        "-f", "nc4",  # Force NetCDF4 format (compressed, modern)
        "-z", "zip_4",  # Compression level 4 (good balance of speed and size)
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
    output_path_str = str(relative_path).replace("lambert_proj", f"latlon_proj_{region_name}/remapped")
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
        print(f"Remapping directory: {input_dir}")
        print(f"{'='*60}")
        
        # Check if directory exists
        if not input_dir.exists():
            print(f"Warning: Directory not found: {input_dir}")
            continue
        
        # Determine output directory
        output_dir = determine_output_dir(input_dir, region_name, paths["lambert_proj"])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Get input files (separate year and static files for better reporting)
        if input_file:
            # Specific file requested
            input_files = get_input_files(input_dir, years, input_file)
            year_files = []
            static_files = []
            if input_files and is_year_file(input_files[0].name):
                year_files = input_files
            else:
                static_files = input_files
        else:
            # Get year files and static files separately
            year_files = get_year_files(input_dir, years)
            static_files = get_static_files(input_dir)
        
        # Report what was found
        if year_files:
            print(f"Found {len(year_files)} year file(s): {[f.name for f in year_files]}")
        if static_files:
            print(f"Found {len(static_files)} static file(s): {[f.name for f in static_files]}")
        
        all_files = year_files + static_files
        
        if not all_files:
            print(f"No input files found in {input_dir}")
            continue
        
        # Process each file
        for input_file_path in all_files:
            output_filename = input_file_path.stem + ".nc"
            output_file_path = output_dir / output_filename
            
            # Skip if output file already exists
            if output_file_path.exists():
                print(f"  Skipping {input_file_path.name} (output already exists)")
                continue
            
            file_type = "year" if is_year_file(input_file_path.name) else "static"
            print(f"\n  Processing {file_type} file: {input_file_path.name}")
            run_cdo_remap(input_file_path, output_file_path, coord_file)
        

def get_input_directories_for_region(region: str, dir_suffixes: List[str]) -> List[str]:
    """Construct full directory paths for a region based on suffixes.
    
    Args:
        region: Region name (e.g., 'central_europe')
        dir_suffixes: Directory suffixes (e.g., ['single_levels', 'single_levels_humidity'])
        
    Returns:
        List of full directory paths that exist
    """
    paths = get_output_paths(region)
    base_dir = paths["lambert_proj"]
    
    full_dirs = []
    
    for suffix in dir_suffixes:
        suffix_path = Path(suffix)
        if suffix_path.is_absolute():
            full_path = suffix_path
        else:
            full_path = base_dir / suffix_path
        
        if full_path.exists():
            full_dirs.append(full_path)
        else:
            print(f"  Note: Directory not found (skipping): {full_path}")
    
    return full_dirs


def project_all_regions(dir_suffixes: List[str] = None,
                       years: Optional[List[str]] = None,
                       input_file: Optional[str] = None) -> None:
    """Project CERRA data for all regions.
    
    Args:
        dir_suffixes: Directory suffixes to combine (e.g., ['single_levels', 'single_levels_humidity'])
        years: List of years to process (optional, if not provided processes all years)
        input_file: Specific file to process (optional)
    """
    regions = ["central_europe", "iberia", "scandinavia"]
    
    print(f"\n{'='*70}")
    print(f"Projecting all regions: {regions}")
    if dir_suffixes:
        print(f"Directory suffixes: {dir_suffixes}")
    if years:
        print(f"Processing years: {years}")
    if input_file:
        print(f"Processing file: {input_file}")
    print(f"{'='*70}")
    
    for region in regions:
        print(f"\n{'='*70}")
        print(f"Projecting region: {region}")
        print(f"{'='*70}")
        
        # Get full input directories for this region
        remap_dirs = get_input_directories_for_region(region, dir_suffixes)
        
        if not remap_dirs:
            print(f"No valid input directories found for region {region}")
            continue
        
        # Project this region
        project_cerra_data(
            region=region,
            remap_directories=remap_dirs,
            years=years,
            input_file=input_file
        )
    
    print(f"\n{'='*70}")
    print("All regions projected successfully!")
    print(f"{'='*70}")


def main():
    """Main function to handle command line arguments and execute projection."""
    parser = argparse.ArgumentParser(
        description="Project CERRA data from Lambert to cylindrical projection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Project ALL regions for specific years
  python project_cerra.py --years 2014 2015 \\
    --input_directories single_levels single_levels_humidity
  
  # Project specific region for specific years
  python project_cerra.py --region central_europe --years 2014 2015 \\
    --input_directories single_levels single_levels_humidity
  
  # Project ALL regions, all years + static files
  python project_cerra.py \\
    --input_directories single_levels single_levels_static
  
  # Project specific region, all years + static files
  python project_cerra.py --region iberia \\
    --input_directories single_levels single_levels_static
  
  # Project a specific file for a specific region
  python project_cerra.py --region scandinavia --input-file orography.grib \\
    --input_directories single_levels_static

Notes:
  - If --region is NOT provided, all regions will be projected
  - Year files must be named YYYY.grib (e.g., 2014.grib, 2015.grib)
  - Static files can have any other name (e.g., orography.grib)
  - Static files are automatically included when processing directories
  - Output files are saved as compressed NetCDF4 (.nc) format
  - Files are saved to latlon_proj_{Region}/remapped/{directory_name}/
        """
    )
    
    parser.add_argument(
        "--region",
        type=str,
        required=False,
        choices=["central_europe", "iberia", "scandinavia"],
        help="Target region for projection (if not provided, projects all regions)"
    )
    
    parser.add_argument(
        "--input_directories",
        nargs="+",
        type=str,
        default=["single_levels"],
        help="Directory suffixes to combine (e.g., single_levels single_levels_humidity). "
             "Will be combined with region-specific base path automatically. "
             "Default: ['single_levels']"
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
    
    # Execute projection
    if args.region:
        # Project specific region
        remap_dirs = get_input_directories_for_region(args.region, args.input_directories)
        
        project_cerra_data(
            region=args.region,
            remap_directories=remap_dirs,
            years=args.years,
            input_file=args.input_file
        )
    else:
        # Project all regions
        project_all_regions(
            dir_suffixes=args.input_directories,
            years=args.years,
            input_file=args.input_file
        )
    


if __name__ == "__main__":
    main()
