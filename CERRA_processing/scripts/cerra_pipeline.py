#!/usr/bin/env python3
"""
CERRA Data Processing Pipeline

This script orchestrates the complete CERRA data processing workflow:
1. Download data from CDS
2. Project from Lambert to cylindrical coordinates
3. Preprocess and extract meteorological variables

Usage:
    python cerra_pipeline.py --region central_europe --years 2020 2021 --variables base_variables
    python cerra_pipeline.py --region iberia --years 2014-2021 --skip-download
    python cerra_pipeline.py --region scandinavia --years 2020 --skip-projection
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import time

# Add the current directory to the path to import config
sys.path.append(str(Path(__file__).parent))
from config import get_region_config, get_output_paths, get_variable_group


class CERRAPipeline:
    """Main pipeline class for CERRA data processing."""
    
    def __init__(self, region: str):
        """Initialize pipeline for a specific region.
        
        Args:
            region: Region name (central_europe, iberia, scandinavia)
        """
        self.region = region
        self.region_config = get_region_config(region)
        self.paths = get_output_paths(region)
        self.script_dir = Path(__file__).parent
        
    def run_download(self, years: List[str], variables: List[str]) -> None:
        """Run the download step.
        
        Args:
            years: List of years to download
            variables: List of variables to download (always download all_variables)
        """
        print("=" * 60)
        print("STEP 1: DOWNLOADING CERRA DATA")
        print("=" * 60)
        
        # Always download all variables since base_variables are included
        # The user can choose which subset to process later
        download_variables = get_variable_group("all_variables")
        
        print(f"Downloading all variables: {download_variables}")
        print("Note: You can choose which variables to process later in the preprocessing step")
        
        # Prepare command - pass the variable group name, not individual variables
        cmd = [
            sys.executable,
            str(self.script_dir / "download_cerra.py"),
            "--years", ",".join(years),
            "--variables", "all_variables"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, check=True)
            end_time = time.time()
            print(f"Download completed in {end_time - start_time:.2f} seconds")
            
        except subprocess.CalledProcessError as e:
            print(f"Download failed with error: {e}")
            raise
    
    def run_projection(self, years: List[str] = None, input_dir: Path = None) -> None:
        """Run the projection step.
        
        Args:
            years: List of years to project (optional, if not provided, process all files in input_dir)
            input_dir: Input directory containing GRIB files (optional, defaults to lambert_proj)
        """
        print("=" * 60)
        print("STEP 2: PROJECTING TO CYLINDRICAL COORDINATES")
        print("=" * 60)
        
        # Set input directory
        if input_dir is None:
            input_dir = self.paths["lambert_proj"] / "single_levels"
        
        print(f"Input directory: {input_dir}")
        
        # Prepare command
        cmd = [
            sys.executable,
            str(self.script_dir / "project_cerra.py"),
            "--region", self.region,
            "--input-dir", str(input_dir)
        ]
        
        # Add years if provided, otherwise process all files in directory
        if years:
            cmd.extend(["--years"] + years)
        
        print(f"Running: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            end_time = time.time()
            print(f"Projection completed in {end_time - start_time:.2f} seconds")
            print("Projection output:")
            print(result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"Projection failed with error: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
    def run_preprocessing(self, years: List[str] = None, variables: List[str] = None, 
                         input_dir: Path = None) -> None:
        """Run the preprocessing step.
        
        Args:
            years: List of years to preprocess (optional, if not provided, process all files in input_dir)
            variables: List of variables to process (optional, defaults to base_variables)
            input_dir: Input directory containing projected GRIB files (optional, defaults to latlon_proj)
        """
        print("=" * 60)
        print("STEP 3: PREPROCESSING DATA")
        print("=" * 60)
        
        # Set default variables if not provided
        if variables is None:
            variables = ["base_variables"]
        
        # Set input directory
        if input_dir is None:
            input_dir = self.paths["latlon_proj"] / "single_levels"
        
        print(f"Input directory: {input_dir}")
        print(f"Processing variables: {variables}")
        
        # Process each year
        if years:
            for year in years:
                print(f"\nProcessing year: {year}")
                
                # Prepare command
                cmd = [
                    sys.executable,
                    str(self.script_dir / "preprocess_cerra.py"),
                    "--region", self.region,
                    "--year", year,
                    "--input-dir", str(input_dir),
                    "--variables", variables[0] if isinstance(variables, list) else variables
                ]
                
                print(f"Running: {' '.join(cmd)}")
                start_time = time.time()
                
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    end_time = time.time()
                    print(f"Preprocessing for {year} completed in {end_time - start_time:.2f} seconds")
                    print("Preprocessing output:")
                    print(result.stdout)
                    
                except subprocess.CalledProcessError as e:
                    print(f"Preprocessing failed for {year} with error: {e}")
                    print(f"STDOUT: {e.stdout}")
                    print(f"STDERR: {e.stderr}")
                    raise
        else:
            # Process all files in the input directory
            print(f"\nProcessing all files in directory: {input_dir}")
            
            # Prepare command
            cmd = [
                sys.executable,
                str(self.script_dir / "preprocess_cerra.py"),
                "--region", self.region,
                "--input-dir", str(input_dir),
                "--variables", variables[0] if isinstance(variables, list) else variables
            ]
            
            print(f"Running: {' '.join(cmd)}")
            start_time = time.time()
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                end_time = time.time()
                print(f"Preprocessing completed in {end_time - start_time:.2f} seconds")
                print("Preprocessing output:")
                print(result.stdout)
                
            except subprocess.CalledProcessError as e:
                print(f"Preprocessing failed with error: {e}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                raise
    
    def run_full_pipeline(self, years: List[str], variables: List[str], 
                         skip_download: bool = False, skip_projection: bool = False, 
                         skip_preprocessing: bool = False, input_dir: Path = None) -> None:
        """Run the complete pipeline.
        
        Args:
            years: List of years to process
            variables: List of variables to process (for preprocessing step)
            skip_download: Skip the download step
            skip_projection: Skip the projection step
            skip_preprocessing: Skip the preprocessing step
            input_dir: Input directory for projection/preprocessing (optional)
        """
        print("=" * 60)
        print(f"CERRA DATA PROCESSING PIPELINE - {self.region.upper()}")
        print("=" * 60)
        print(f"Region: {self.region}")
        print(f"Years: {years}")
        print(f"Variables: {variables}")
        print(f"Skip download: {skip_download}")
        print(f"Skip projection: {skip_projection}")
        print(f"Skip preprocessing: {skip_preprocessing}")
        print("=" * 60)
        
        total_start_time = time.time()
        
        try:
            # Step 1: Download
            if not skip_download:
                self.run_download(years, variables)
            else:
                print("Skipping download step...")
            
            # Step 2: Projection
            if not skip_projection:
                self.run_projection(years, input_dir)
            else:
                print("Skipping projection step...")
            
            # Step 3: Preprocessing
            if not skip_preprocessing:
                self.run_preprocessing(years, variables, input_dir)
                total_end_time = time.time()
                print("=" * 60)
                print("PIPELINE COMPLETED SUCCESSFULLY!")
                print(f"Total processing time: {total_end_time - total_start_time:.2f} seconds")
                print("=" * 60)
                
                # Print output locations
                print("\nOutput locations:")
                print(f"Processed data: {self.paths['processed_data']}")
                print(f"Coordinate grid: {self.paths['processed_data'] / 'nwp_xy.npy'}")
            else:
                print("Skipping preprocessing step...")
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met.
        
        Returns:
            True if all prerequisites are met, False otherwise
        """
        print("Checking prerequisites...")
        
        # Check if scripts exist
        scripts = [
            "download_cerra.py",
            "project_cerra.py", 
            "preprocess_cerra.py"
        ]
        
        for script in scripts:
            script_path = self.script_dir / script
            if not script_path.exists():
                print(f"Error: Required script not found: {script_path}")
                return False
        
        # Check if coordinate files exist
        coord_file = self.paths["coordinate_files"] / self.region_config["coord_file"]
        if not coord_file.exists():
            print(f"Error: Coordinate file not found: {coord_file}")
            return False
        
        # Check if CDO is available (for projection step)
        try:
            subprocess.run(["cdo", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: CDO (Climate Data Operators) not found. Projection step will fail.")
            print("Please install CDO: https://code.mpimet.mpg.de/projects/cdo")
        
        print("Prerequisites check completed.")
        return True


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


def main():
    """Main function to handle command line arguments and execute pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete CERRA data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline for central Europe, years 2020-2021
  python cerra_pipeline.py --region central_europe --years 2020-2021 --variables base_variables
  
  # Skip download, only project and preprocess
  python cerra_pipeline.py --region iberia --years 2014,2015 --skip-download
  
  # Skip projection, only preprocess existing data
  python cerra_pipeline.py --region scandinavia --years 2020 --skip-projection
  
  # Skip preprocessing, only download and project
  python cerra_pipeline.py --region central_europe --years 2021 --skip-preprocessing
  
  # Process downloaded files with specific input directory
  python cerra_pipeline.py --region central_europe --years 2021 --skip-download --input-dir /path/to/lambert_proj
  
  # Process different variable subsets from same downloaded data
  python cerra_pipeline.py --region central_europe --years 2021 --skip-download --variables base_variables
  python cerra_pipeline.py --region central_europe --years 2021 --skip-download --variables all_variables
        """
    )
    
    parser.add_argument(
        "--region",
        type=str,
        required=False,
        choices=["central_europe", "iberia", "scandinavia"],
        help="Target region for processing"
    )
    
    parser.add_argument(
        "--years",
        type=str,
        required=False,
        help="Years to process (e.g., '2014', '2014,2015', '2014-2021')"
    )
    
    parser.add_argument(
        "--variables",
        type=str,
        default="base_variables",
        help="Variable group or comma-separated list of variables. "
             "Available groups: base_variables, additional_variables, all_variables"
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step"
    )
    
    parser.add_argument(
        "--skip-projection",
        action="store_true",
        help="Skip the projection step"
    )
    
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip the preprocessing step"
    )
    
    parser.add_argument(
        "--check-prerequisites",
        action="store_true",
        help="Only check prerequisites and exit"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory for projection/preprocessing steps (default: lambert_proj/single_levels)"
    )
    
    args = parser.parse_args()
    
    # Check if this is a download-only operation
    download_only = args.skip_projection and args.skip_preprocessing
    
    # For download-only operations, region is not required
    if download_only and not args.region:
        args.region = "central_europe"  # Use default region for download
        print("No region specified for download-only operation. Using default region: central_europe")
    
    # Check prerequisites first
    if args.check_prerequisites:
        if not args.region:
            print("Error: --region is required for --check-prerequisites")
            sys.exit(1)
        pipeline = CERRAPipeline(args.region)
        if not pipeline.check_prerequisites():
            print("Prerequisites check failed. Please fix the issues and try again.")
            sys.exit(1)
        print("Prerequisites check completed successfully.")
        sys.exit(0)
    
    # Parse years
    if not args.years:
        print("Error: --years argument is required when not using --check-prerequisites")
        sys.exit(1)
    
    years = parse_years(args.years)
    
    # Initialize pipeline
    if not args.region:
        print("Error: --region is required for projection and preprocessing steps")
        sys.exit(1)
    
    pipeline = CERRAPipeline(args.region)
    
    # Check prerequisites
    if not pipeline.check_prerequisites():
        print("Prerequisites check failed. Please fix the issues and try again.")
        sys.exit(1)
    
    # Parse variables
    if args.variables in ["base_variables", "additional_variables", "all_variables"]:
        variables = args.variables
    else:
        # Assume comma-separated list of specific variables
        variables = [v.strip() for v in args.variables.split(',')]
    
    # Set input directory
    input_dir = None
    if args.input_dir:
        input_dir = Path(args.input_dir)
    
    # Run pipeline
    pipeline.run_full_pipeline(
        years=years,
        variables=variables,
        skip_download=args.skip_download,
        skip_projection=args.skip_projection,
        skip_preprocessing=args.skip_preprocessing,
        input_dir=input_dir
    )


if __name__ == "__main__":
    main()
