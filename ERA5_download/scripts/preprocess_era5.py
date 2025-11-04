#!/usr/bin/env python3
"""
ERA5 Data Preprocessing Script - NetCDF Output Version

This script processes ERA5 NetCDF files and saves them as processed NetCDF files
with proper CF-compliant metadata. This maintains compatibility with the climate
data ecosystem while providing efficient access.

Usage:
    python preprocess_era5.py --region central_europe --year 2021
    python preprocess_era5.py --region iberia --input-file 2021.nc
"""

import argparse
import xarray as xr
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import time

# Add the current directory to the path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_region_config, get_output_paths, PROCESSING_CONFIG, get_samples_per_year


class ERA5ProcessorNetCDF:
    """Class to handle ERA5 data processing with NetCDF output."""
    
    def __init__(self, region: str, variables: str = "base_variables"):
        """Initialize processor for a specific region.
        
        Args:
            region: Region name (central_europe, iberia, scandinavia)
            variables: Variable group to process ("base_variables" or "all_variables")
        """
        self.region = region
        self.region_config = get_region_config(region)
        self.paths = get_output_paths(region)
        self.variables = variables
        
        # Set variable selection based on input
        if variables == "base_variables":
            # Only process u10, v10, t2m
            self.variable_indices = [0, 1, 2]  # u10, v10, t2m
            self.variable_names = ['u10', 'v10', 't2m']
        elif variables == "all_variables":
            # Process all variables: u10, v10, t2m, sshf, zust, sp
            self.variable_indices = [0, 1, 2, 3, 4, 5]  # u10, v10, t2m, sshf, zust, sp
            self.variable_names = ['u10', 'v10', 't2m', 'sshf', 'zust', 'sp']
        else:
            raise ValueError(f"Unknown variable group: {variables}. Use 'base_variables' or 'all_variables'")
        
    def load_dataset(self, input_file: str) -> xr.Dataset:
        """Load ERA5 dataset from NetCDF file.
        
        Args:
            input_file: Path to input NetCDF file
            
        Returns:
            ERA5 dataset
        """
        print(f"Loading dataset from: {input_file}")
        
        # Load ERA5 dataset
        ds = xr.open_dataset(input_file)
        
        print(f"Dataset loaded successfully")
        print(f"Variables: {list(ds.data_vars)}")
        print(f"Dimensions: {dict(ds.dims)}")
        
        return ds
    
    def extract_variables(self, ds: xr.Dataset) -> Tuple[np.ndarray, ...]:
        """Extract and process variables from ERA5 dataset.
        
        Args:
            ds: ERA5 dataset
            
        Returns:
            Tuple of processed variable arrays
        """
        print("Extracting and processing variables...")
        
        # Extract basic variables
        u10 = ds['u10'].values  # 10m u-component of wind
        v10 = ds['v10'].values  # 10m v-component of wind
        t2m = ds['t2m'].values  # 2m temperature
        
        # Initialize other variables with zeros if not in all_variables mode
        if self.variables == "all_variables":
            # Extract additional variables for all_variables mode
            sshf = ds['ishf'].values  # Instantaneous surface sensible heat flux (W/m²)
            zust = ds['zust'].values  # Friction velocity (m/s)
            sp = ds['sp'].values      # Surface pressure (Pa)
        else:
            # Create dummy arrays for base_variables mode
            sshf = np.zeros_like(t2m)
            zust = np.zeros_like(t2m)
            sp = np.zeros_like(t2m)
        
        print(f"Variable shapes:")
        print(f"  u10: {u10.shape}")
        print(f"  v10: {v10.shape}")
        print(f"  t2m: {t2m.shape}")
        if self.variables == "all_variables":
            print(f"  sshf: {sshf.shape}")
            print(f"  zust: {zust.shape}")
            print(f"  sp: {sp.shape}")
        
        return u10, v10, t2m, sshf, zust, sp
    
    def process_file_to_netcdf(self, input_file: str, output_dir: str, year: int) -> None:
        """Process a single ERA5 file and save as NetCDF.
        
        Args:
            input_file: Path to input NetCDF file
            output_dir: Output directory for processed data
            year: Year for timestamp generation
        """
        print(f"Processing file: {os.path.basename(input_file)}")
        
        # Load dataset
        ds = self.load_dataset(input_file)
        
        # Extract variables
        u10, v10, t2m, sshf, zust, sp = self.extract_variables(ds)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamps
        num_samples = get_samples_per_year(year)
        start_time = datetime(year, 1, 1, 1)  # Start: 01-01 01:00
        time_step = timedelta(hours=PROCESSING_CONFIG["time_step_hours"])
        time_values = [start_time + i * time_step for i in range(num_samples)]
        
        # Create time coordinate
        time_coord = xr.cftime_range(
            start=start_time,
            periods=num_samples,
            freq=f"{PROCESSING_CONFIG['time_step_hours']}H",
            calendar='standard'
        )
        
        # Create NetCDF filename
        netcdf_filename = f"{self.region_config['name']}_{year}_era5.nc"
        netcdf_filepath = os.path.join(output_dir, netcdf_filename)
        
        print(f"Saving to NetCDF: {netcdf_filepath}")
        
        # Create data arrays for selected variables
        data_vars = {}
        for i, var_name in enumerate(self.variable_names):
            var_index = self.variable_indices[i]
            if var_index == 0:
                data = u10
            elif var_index == 1:
                data = v10
            elif var_index == 2:
                data = t2m
            elif var_index == 3:
                data = sshf
            elif var_index == 4:
                data = zust
            elif var_index == 5:
                data = sp
            
            # Create DataArray with proper coordinates and attributes
            data_vars[var_name] = xr.DataArray(
                data,
                coords={
                    'time': time_coord,
                    'latitude': ds.latitude,
                    'longitude': ds.longitude
                },
                dims=['time', 'latitude', 'longitude'],
                attrs=self._get_variable_attrs(var_name)
            )
        
        # Create dataset
        processed_ds = xr.Dataset(
            data_vars,
            coords={
                'time': time_coord,
                'latitude': ds.latitude,
                'longitude': ds.longitude
            },
            attrs=self._get_global_attrs(year)
        )
        
        # Save to NetCDF with compression
        encoding = {}
        for var_name in self.variable_names:
            encoding[var_name] = {
                'zlib': True,
                'complevel': 6,
                'chunksizes': (1, processed_ds[var_name].shape[1], processed_ds[var_name].shape[2])
            }
        
        processed_ds.to_netcdf(
            netcdf_filepath,
            encoding=encoding,
            format='NETCDF4'
        )
        
        # Print file size information
        file_size_mb = os.path.getsize(netcdf_filepath) / (1024 * 1024)
        print(f"NetCDF file created: {netcdf_filepath}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Saved {len(time_values)} time steps in NetCDF format")
    
    def _get_variable_attrs(self, var_name: str) -> dict:
        """Get CF-compliant attributes for variables."""
        attrs_map = {
            'u10': {
                'long_name': '10 metre U wind component',
                'standard_name': 'eastward_wind',
                'units': 'm s-1',
                'description': '10m u-component of wind'
            },
            'v10': {
                'long_name': '10 metre V wind component', 
                'standard_name': 'northward_wind',
                'units': 'm s-1',
                'description': '10m v-component of wind'
            },
            't2m': {
                'long_name': '2 metre temperature',
                'standard_name': 'air_temperature',
                'units': 'K',
                'description': '2m temperature'
            },
            'sshf': {
                'long_name': 'Surface sensible heat flux',
                'standard_name': 'surface_upward_sensible_heat_flux',
                'units': 'W m-2',
                'description': 'Instantaneous surface sensible heat flux'
            },
            'zust': {
                'long_name': 'Friction velocity',
                'standard_name': 'friction_velocity',
                'units': 'm s-1',
                'description': 'Friction velocity'
            },
            'sp': {
                'long_name': 'Surface pressure',
                'standard_name': 'surface_air_pressure',
                'units': 'Pa',
                'description': 'Surface pressure'
            }
        }
        return attrs_map.get(var_name, {})
    
    def _get_global_attrs(self, year: int) -> dict:
        """Get global attributes for the dataset."""
        return {
            'title': f'Processed ERA5 data for {self.region_config["name"]} - {year}',
            'description': f'Processed ERA5 reanalysis data for region {self.region}',
            'source': 'ERA5 reanalysis',
            'institution': 'ECMWF',
            'processing_date': datetime.now().isoformat(),
            'region': self.region,
            'year': year,
            'variables': self.variable_names,
            'conventions': 'CF-1.8',
            'history': f'Processed from ERA5 NetCDF files on {datetime.now().isoformat()}'
        }
        
    def save_static_data(self, geopotential_file: str, output_dir: str) -> None:
        """Save static variables (geopotential) to NetCDF file."""
        print("Processing static variables...")
        
        # Load geopotential
        ds = xr.open_dataset(geopotential_file)

        # Check dimensions and extract the 2D numpy array
        if ds['z'].ndim == 2:
            print("Data is 2D (static).")
            geop = ds['z'].values  # Shape is already (latitude, longitude)
        else:
            print("Data is 3D (or more). Slicing 'valid_time=0'.")
            geop = ds['z'].isel(valid_time=0).values
        
        # Create static NetCDF file
        static_filename = f"{self.region_config['name']}_static_era5.nc"
        static_filepath = os.path.join(output_dir, static_filename)
        
        # Create DataArray for geopotential
        geopotential_da = xr.DataArray(
            geop,
            coords={
                'latitude': ds.latitude,
                'longitude': ds.longitude,
            },
            dims=['latitude', 'longitude'],
            attrs={
                'long_name': 'Geopotential',
                'standard_name': 'geopotential',
                'units': 'm2 s-2',
                'description': 'Surface geopotential'
            }
        )
        
        # Create dataset
        static_ds = xr.Dataset(
            {'geopotential': geopotential_da},
            coords={
                'latitude': ds.latitude,
                'longitude': ds.longitude
            },
            attrs={
                'title': f'Static ERA5 data for {self.region_config["name"]}',
                'description': 'Static ERA5 reanalysis data (geopotential)',
                'source': 'ERA5 reanalysis',
                'institution': 'ECMWF',
                'processing_date': datetime.now().isoformat(),
                'region': self.region,
                'conventions': 'CF-1.8',
                'history': f'Processed from ERA5 NetCDF files on {datetime.now().isoformat()}'
            }
        )
        
        # Save to NetCDF with compression
        encoding = {
            'geopotential': {
                'zlib': True,
                'complevel': 6
            }
        }
        
        static_ds.to_netcdf(
            static_filepath,
            encoding=encoding,
            format='NETCDF4'
        )
        
        file_size_mb = os.path.getsize(static_filepath) / (1024 * 1024)
        print(f"Static data saved to: {static_filepath}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"geopotential shape: {geop.shape}")
    


def process_era5_data_netcdf(region: str, year: Optional[int] = None, 
                            input_file: Optional[str] = None,
                            input_dir: Optional[str] = None,
                            variables: str = "base_variables") -> None:
    """Process ERA5 data for a specific region and save as NetCDF.
    
    Args:
        region: Region name
        year: Year to process (optional)
        input_file: Specific file to process (optional)
        input_dir: Input directory (optional)
        variables: Variable group to process ("base_variables" or "all_variables")
    """
    processor = ERA5ProcessorNetCDF(region, variables)
    paths = processor.paths
    
    # Set input directory
    if input_dir is None:
        input_dir = str(paths["raw_data"])
        
    output_dir = str(paths["raw_data"].with_name("single_levels_processed"))
    
    # Get input files
    if input_file:
        input_files = [os.path.join(input_dir, input_file)]
    elif year:
        input_files = [os.path.join(input_dir, f"{year}.nc")]
    else:
        # Find all {year}.nc files
        import glob
        pattern = os.path.join(input_dir, "[0-9][0-9][0-9][0-9].nc")
        input_files = glob.glob(pattern)
        input_files.sort()  # Sort by year
    
    if not input_files:
        print("No input files found to process.")
        return
    
    print(f"Processing {len(input_files)} files for region: {region}")
    print(f"Output directory: {output_dir}")
    
    # Process each file
    for input_file_path in input_files:
        if not os.path.exists(input_file_path):
            print(f"Warning: File not found: {input_file_path}")
            continue
        
        # Extract year from filename
        filename = os.path.splitext(os.path.basename(input_file_path))[0]
        try:
            file_year = int(filename)
        except ValueError:
            print(f"Warning: Could not extract year from filename: {filename}")
            continue
        
        # Process the file
        processor.process_file_to_netcdf(input_file_path, output_dir, file_year)
        
    # Process static data (geopotential)
    geopotential_file = os.path.join(input_dir, "geopotential.nc")
    if os.path.exists(geopotential_file):
        processor.save_static_data(geopotential_file, output_dir)
    else:
        print(f"Warning: geopotential file not found: {geopotential_file}")
    
    # Note: Coordinates are now included in the NetCDF files themselves
    
    print(f"\nProcessing completed for region: {region}")
    print(f"Output files saved to: {output_dir}")
        
            
            
            
def process_all_regions(year: Optional[int] = None, 
                       input_file: Optional[str] = None,
                       variables: str = "base_variables") -> None:
    """Process ERA5 data for all regions.
    
    Args:
        year: Year to process (optional, if not provided processes all years)
        input_file: Specific file to process (optional)
        variables: Variable group to process ("base_variables" or "all_variables")
    """
    regions = ["central_europe", "iberia", "scandinavia"]
    
    print(f"Processing all regions: {regions}")
    if year:
        print(f"Processing year: {year}")
    if input_file:
        print(f"Processing file: {input_file}")
    
    for region in regions:
        print(f"\n{'='*50}")
        print(f"Processing region: {region}")
        print(f"{'='*50}")
        process_era5_data_netcdf(region, year, input_file, None, variables)
    
    print(f"\n{'='*50}")
    print("All regions processed successfully!")
    print(f"{'='*50}")


def main():
    """Main function to handle command line arguments and execute processing."""
    parser = argparse.ArgumentParser(
        description="Process ERA5 data to NetCDF format (u10, v10, t2m for base_variables; u10, v10, t2m, sshf, zust, sp for all_variables)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process data for central Europe for 2021
  python preprocess_era5.py --region central_europe --year 2021
  
  # Process a specific file for Iberia
  python preprocess_era5.py --region iberia --input-file single_2021.nc
  
  # Process all available files for Scandinavia
  python preprocess_era5.py --region scandinavia
        """
    )
    
    parser.add_argument(
        "--region",
        type=str,
        required=False,
        choices=["central_europe", "iberia", "scandinavia", "eurasia"],
        help="Target region for processing"
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--year",
        type=int,
        help="Year to process"
    )
    
    group.add_argument(
        "--input-file",
        type=str,
        help="Specific input file to process"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory (default: {region}/single_levels)"
    )
    
    parser.add_argument(
        "--variables",
        type=str,
        default="all_variables",
        choices=["base_variables", "all_variables"],
        help="Variable group to process (default: base_variables)"
    )
    
    args = parser.parse_args()
    
    # Set input directory
    input_dir = None
    if args.input_dir:
        input_dir = args.input_dir
    
    if args.region:
        # Process specific region
        process_era5_data_netcdf(
            region=args.region,
            year=args.year,
            input_file=args.input_file,
            input_dir=input_dir,
            variables=args.variables
        )
        
    else:
        # Process all regions
        process_all_regions(
            year=args.year,
            input_file=args.input_file,
            variables=args.variables
        )


if __name__ == "__main__":
    main()
