#!/usr/bin/env python3
"""
Example usage of the CERRA data processing pipeline.

This script demonstrates how to use the pipeline programmatically
and provides examples for common use cases.
"""

import subprocess
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))
from config import get_region_config, get_output_paths


def example_full_pipeline():
    """Example: Run full pipeline for central Europe, year 2021."""
    print("Example 1: Full pipeline for Central Europe, 2021")
    print("-" * 50)
    
    cmd = [
        sys.executable,
        "cerra_pipeline.py",
        "--region", "central_europe",
        "--years", "2021",
        "--variables", "base_variables"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("This will:")
    print("1. Download CERRA data for 2021")
    print("2. Project to cylindrical coordinates for Central Europe")
    print("3. Preprocess and extract variables (u10, v10, t2m, sshf, zust)")
    print("4. Save as numpy arrays in data/CentralEurope/CERRA/samples/")
    print()


def example_multiple_years():
    """Example: Process multiple years."""
    print("Example 2: Multiple years (2014-2016)")
    print("-" * 50)
    
    cmd = [
        sys.executable,
        "cerra_pipeline.py",
        "--region", "iberia",
        "--years", "2014-2016",
        "--variables", "base_variables"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("This will process years 2014, 2015, and 2016 for the Iberia region.")
    print()


def example_skip_download():
    """Example: Skip download step (use existing data)."""
    print("Example 3: Skip download (use existing Lambert projection data)")
    print("-" * 50)
    
    cmd = [
        sys.executable,
        "cerra_pipeline.py",
        "--region", "scandinavia",
        "--years", "2021",
        "--variables", "base_variables",
        "--skip-download"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("This will skip the download step and use existing data in lambert_proj/")
    print()


def example_custom_variables():
    """Example: Download specific variables only."""
    print("Example 4: Custom variables")
    print("-" * 50)
    
    cmd = [
        sys.executable,
        "cerra_pipeline.py",
        "--region", "central_europe",
        "--years", "2020",
        "--variables", "10m_wind_speed,2m_temperature"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("This will download only wind speed and temperature variables.")
    print()


def example_individual_steps():
    """Example: Run individual steps."""
    print("Example 5: Individual steps")
    print("-" * 50)
    
    print("Step 1 - Download:")
    print("python download_cerra.py --years 2021 --variables base_variables")
    print()
    
    print("Step 2 - Project:")
    print("python project_cerra.py --region central_europe --years 2021")
    print()
    
    print("Step 3 - Preprocess:")
    print("python preprocess_cerra.py --region central_europe --year 2021")
    print()


def example_check_prerequisites():
    """Example: Check prerequisites."""
    print("Example 6: Check prerequisites")
    print("-" * 50)
    
    cmd = [
        sys.executable,
        "cerra_pipeline.py",
        "--region", "central_europe",
        "--check-prerequisites"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("This will check if all required software is installed.")
    print()


def show_output_structure():
    """Show the expected output structure."""
    print("Expected Output Structure:")
    print("-" * 50)
    
    regions = ["central_europe", "iberia", "scandinavia"]
    
    for region in regions:
        region_config = get_region_config(region)
        paths = get_output_paths(region)
        
        print(f"\n{region.upper()} ({region_config['description']}):")
        print(f"  Processed data: {paths['processed_data']}")
        print(f"  Coordinate grid: {paths['processed_data'] / 'nwp_xy.npy'}")
        print(f"  Time series files: {paths['processed_data'] / 'nwp_YYYYMMDDHH.npy'}")


def main():
    """Main function to display all examples."""
    print("CERRA Data Processing Pipeline - Usage Examples")
    print("=" * 60)
    print()
    
    example_full_pipeline()
    example_multiple_years()
    example_skip_download()
    example_custom_variables()
    example_individual_steps()
    example_check_prerequisites()
    show_output_structure()
    
    print("\n" + "=" * 60)
    print("To run any of these examples, copy the command and execute it in the terminal.")
    print("Make sure you have:")
    print("1. CDS API credentials set up (~/.cdsapirc)")
    print("2. CDO installed and in PATH")
    print("3. Required Python packages installed")
    print("4. Sufficient disk space for the data")
    print("=" * 60)


if __name__ == "__main__":
    main()
