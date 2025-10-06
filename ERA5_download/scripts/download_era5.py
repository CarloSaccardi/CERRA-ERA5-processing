#!/usr/bin/env python3
"""
ERA5 Data Download Script

This script downloads ERA5 reanalysis data from the Copernicus Climate Data Store (CDS)
for specified regions and time periods.

Usage:
    python download_era5.py --region central_europe --year 2021
    python download_era5.py --region iberia --year 2021 --variables all_variables
"""

import argparse
import cdsapi
import sys
from pathlib import Path
from typing import Optional, List
import time

# Add the current directory to the path to import config
sys.path.append(str(Path(__file__).parent))
from config import get_region_config, get_output_paths, DOWNLOAD_CONFIG, get_variable_group


class ERA5Downloader:
    """Class to handle ERA5 data downloads from CDS."""
    
    def __init__(self, region: str, variables: str = "base_variables"):
        """Initialize downloader for a specific region.
        
        Args:
            region: Region name (central_europe, iberia, scandinavia)
            variables: Variable group to download ("base_variables" or "all_variables")
        """
        self.region = region
        self.region_config = get_region_config(region)
        self.paths = get_output_paths(region)
        self.variables = get_variable_group(variables)
        
        # Initialize CDS client
        self.client = cdsapi.Client()
        
    def create_download_request(self, year: int) -> dict:
        """Create a CDS API request for downloading ERA5 data.
        
        Args:
            year: Year to download
            
        Returns:
            Dictionary containing the CDS API request parameters
        """
        request = {
            "product_type": DOWNLOAD_CONFIG["product_type"],
            "variable": self.variables,
            "year": [str(year)],
            "month": DOWNLOAD_CONFIG["month"],
            "day": DOWNLOAD_CONFIG["day"],
            "time": DOWNLOAD_CONFIG["time"],
            "data_format": DOWNLOAD_CONFIG["data_format"],
            "download_format": DOWNLOAD_CONFIG["download_format"],
            "area": self.region_config["area"]  # [North, West, South, East]
        }
        
        return request
    
    def download_year(self, year: int, output_filename: Optional[str] = None) -> Path:
        """Download ERA5 data for a specific year.
        
        Args:
            year: Year to download
            output_filename: Optional custom filename for the downloaded file
            
        Returns:
            Path to the downloaded file
        """
        print(f"Downloading ERA5 data for {self.region} region, year {year}")
        print(f"Variables: {', '.join(self.variables)}")
        
        # Create output directory
        self.paths["raw_data"].mkdir(parents=True, exist_ok=True)
        
        # Set output filename
        if output_filename is None:
            if DOWNLOAD_CONFIG["data_format"] == "netcdf":
                output_filename = f"{year}.nc"
            else:
                output_filename = f"{year}.grib"
        
        output_path = self.paths["raw_data"] / output_filename
        
        # Create request
        request = self.create_download_request(year)
        
        print(f"Request parameters:")
        for key, value in request.items():
            if key != "area":
                print(f"  {key}: {value}")
        print(f"  area: {request['area']} (North, West, South, East)")
        
        # Check if file already exists
        if output_path.exists():
            print(f"File already exists: {output_path}")
            response = input("Do you want to overwrite it? (y/N): ")
            if response.lower() != 'y':
                print("Download cancelled.")
                return output_path
        
        print(f"Starting download to: {output_path}")
        print("This may take a while...")
        
        try:
            # Download the data
            start_time = time.time()
            self.client.retrieve(DOWNLOAD_CONFIG["dataset"], request).download(str(output_path))
            end_time = time.time()
            
            # Check if download was successful
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"Download completed successfully!")
                print(f"File: {output_path}")
                print(f"Size: {file_size_mb:.1f} MB")
                print(f"Download time: {end_time - start_time:.1f} seconds")
            else:
                print("Error: Download failed - file not found")
                
        except Exception as e:
            print(f"Error during download: {e}")
            # Clean up partial file if it exists
            if output_path.exists():
                output_path.unlink()
                print("Cleaned up partial download file.")
            raise
        
        return output_path
    
    def download_multiple_years(self, years: List[int]) -> List[Path]:
        """Download ERA5 data for multiple years.
        
        Args:
            years: List of years to download
            
        Returns:
            List of paths to downloaded files
        """
        downloaded_files = []
        
        for year in years:
            try:
                file_path = self.download_year(year)
                downloaded_files.append(file_path)
                
                # Add a small delay between downloads to be respectful to the API
                if len(years) > 1:
                    print("Waiting 5 seconds before next download...")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Failed to download year {year}: {e}")
                continue
        
        return downloaded_files


def download_era5_data(region: str, years: List[int],
                      variables: str = "base_variables",
                      output_filename: Optional[str] = None) -> None:
    """Download ERA5 data for a specific region and time period.
    
    Args:
        region: Region name
        years: List of years to download
        variables: Variable group to download ("base_variables" or "all_variables")
        output_filename: Optional custom filename for single year downloads
    """
    downloader = ERA5Downloader(region, variables)
    
    if len(years) == 1:
        # Download single year
        downloader.download_year(years[0], output_filename)
    else:
        # Download multiple years
        downloader.download_multiple_years(years)


def main():
    """Main function to handle command line arguments and execute download."""
    parser = argparse.ArgumentParser(
        description="Download ERA5 reanalysis data from CDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data for central Europe for 2021
  python download_era5.py --region central_europe --years 2021
  
  # Download data for Iberia for 2021 with all variables
  python download_era5.py --region iberia --years 2021 --variables all_variables
  
  # Download data for multiple years
  python download_era5.py --region scandinavia --years 2020 2021 2022
        """
    )
    
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        choices=["central_europe", "iberia", "scandinavia"],
        help="Target region for download"
    )
    
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="Years to download"
    )
    
    parser.add_argument(
        "--variables",
        type=str,
        default="base_variables",
        choices=["base_variables", "all_variables"],
        help="Variable group to download (default: base_variables)"
    )
    
    parser.add_argument(
        "--output-filename",
        type=str,
        help="Custom filename for single year downloads (optional)"
    )
    
    args = parser.parse_args()
    
    # Process arguments
    years = args.years
    
    # Download data
    download_era5_data(
        region=args.region,
        years=years,
        variables=args.variables,
        output_filename=args.output_filename
    )


if __name__ == "__main__":
    main()
