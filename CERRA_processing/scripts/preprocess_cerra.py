#!/usr/bin/env python3
"""
CERRA Data Preprocessing Script - NetCDF Output Version

This script processes CERRA GRIB files from multiple directories and saves them 
as NetCDF files with proper CF-compliant metadata. This maintains compatibility 
with the climate data ecosystem while providing efficient access.

Usage:
    # Process specific region with multiple input directories
    python preprocess_cerra.py --region central_europe --year 2021 \\
        --input_directories single_levels single_levels_humidity
    
    # Process all regions with humidity data
    python preprocess_cerra.py --year 2021 \\
        --input_directories single_levels single_levels_humidity
    
    # Process all regions, all years
    python preprocess_cerra.py \\
        --input_directories single_levels single_levels_humidity
"""

import argparse
import xarray as xr
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import time

# Add the current directory to the path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_region_config, get_output_paths, PROCESSING_CONFIG, get_samples_per_year


def get_input_directories_for_region(region: str, dir_suffixes: List[str]) -> List[str]:
    """Construct full directory paths for a region based on suffixes.
    
    Args:
        region: Region name (e.g., 'central_europe')
        dir_suffixes: Directory suffixes (e.g., ['single_levels', 'single_levels_humidity'])
        
    Returns:
        List of full directory paths that exist
    """
    paths = get_output_paths(region)
    base_dir = paths["latlon_proj"] / "remapped"
    
    full_dirs = []
    for suffix in dir_suffixes:
        # Check if it's an absolute path
        if os.path.isabs(suffix):
            full_path = suffix
        else:
            # Relative suffix - combine with region base
            full_path = os.path.join(base_dir, suffix)
        
        if os.path.exists(full_path):
            full_dirs.append(full_path)
        else:
            print(f"  Note: Directory not found (skipping): {full_path}")
    
    return full_dirs


def collect_year_files(year: int, input_directories: List[str]) -> List[str]:
    """Collect all files for a given year across multiple directories.
    
    Args:
        year: Year to find files for
        input_directories: List of directories to search
        
    Returns:
        List of file paths for the given year
    """
    files = []
    for directory in input_directories:
        year_file = os.path.join(directory, f"{year}.nc")
        if os.path.exists(year_file):
            files.append(year_file)
    return files


def find_static_file(input_directories: List[str], filename: str) -> Optional[str]:
    """Find a static file (like orography.nc) across directories.
    
    Args:
        input_directories: List of directories to search
        filename: Name of file to find
        
    Returns:
        Path to file if found, None otherwise
    """
    for directory in input_directories:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return filepath
    return None


def find_available_years(input_directories: List[str]) -> List[int]:
    """Find all available years across multiple directories.
    
    Args:
        input_directories: List of directories to search
        
    Returns:
        Sorted list of years found
    """
    years = set()
    for directory in input_directories:
        if not os.path.exists(directory):
            continue
        
        # Find all .nc files that are 4-digit years
        for filename in os.listdir(directory):
            if filename.endswith('.nc'):
                year_str = os.path.splitext(filename)[0]
                try:
                    year = int(year_str)
                    if 1900 <= year <= 2100:  # Sanity check
                        years.add(year)
                except ValueError:
                    pass
    
    return sorted(list(years))


class CERRAProcessorNetCDF:
    """Class to handle CERRA data processing with NetCDF output."""
    
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
        
        # Physical constants
        self.Rd = 287.05  # Gas constant for dry air (J/kg/K)
        self.eps = 0.622  # Ratio of molecular weights of water vapor and dry air
        
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
        
        
    # Variable name mapping from NetCDF to standard names
    NETCDF_VAR_MAP = {
        '2t': 't2m',           # 2m temperature
        '2r': 'r2',            # 2m relative humidity
        '10si': 'si10',        # 10m wind speed
        '10wdir': 'wdir10',    # 10m wind direction
        'sp': 'sp',            # surface pressure (no change)
        'sshf': 'sshf',        # sensible heat flux (no change)
        'tisemf': 'tisemf',    # U-momentum flux (no change)
        'tisnmf': 'tisnmf',    # V-momentum flux (no change)
    }
    
    def standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize NetCDF dataset by renaming variables and squeezing single-level heights.
        
        Args:
            ds: Input xarray Dataset from NetCDF file
            
        Returns:
            Standardized dataset with renamed variables and squeezed dimensions
        """
        # Rename variables according to mapping
        rename_dict = {}
        for nc_name, std_name in self.NETCDF_VAR_MAP.items():
            if nc_name in ds:
                rename_dict[nc_name] = std_name
        
        ds = ds.rename(rename_dict)
        
        # Standardize coordinate names (CDO uses 'lat'/'lon', we want 'latitude'/'longitude')
        coord_rename = {}
        if 'lat' in ds.coords:
            coord_rename['lat'] = 'latitude'
        if 'lon' in ds.coords:
            coord_rename['lon'] = 'longitude'
        if coord_rename:
            ds = ds.rename(coord_rename)
        
        # Squeeze single-level height dimensions
        for var_name in ds.data_vars:
            var = ds[var_name]
            # Find height-like dimensions with size 1
            for dim in var.dims:
                if 'height' in dim.lower() and var.sizes[dim] == 1:
                    ds[var_name] = var.squeeze(dim, drop=True)
        
        return ds
    
    def load_datasets(self, input_files: List[str]) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, 
                                                             xr.Dataset]:
        """Load CERRA datasets from multiple NetCDF files and merge them.
        
        Args:
            input_files: List of paths to input NetCDF files (for the same year)
            
        Returns:
            Tuple of datasets: (combined_ds with all variables)
            Note: Returns single merged dataset since NetCDF files contain all variables
        """
        print(f"Loading datasets from {len(input_files)} file(s):")
        for f in input_files:
            print(f"  - {f}")
        
        # Load and merge all NetCDF files
        datasets = []
        for input_file in input_files:
            try:
                ds = xr.open_dataset(input_file)
                ds = self.standardize_dataset(ds)
                datasets.append(ds)
                print(f"  Loaded: {list(ds.data_vars.keys())}")
            except Exception as e:
                print(f"  Warning: Could not load {input_file}: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded from input files")
        
        # Merge all datasets
        merged_ds = xr.merge(datasets)
        
        print(f"Merged dataset variables: {list(merged_ds.data_vars.keys())}")
        
        # Check for required variables
        required_vars = {
            'wind_components': ['si10', 'wdir10'],
            '2m_data': ['t2m', 'r2'],
            'surface_fluxes': ['sshf', 'tisemf', 'tisnmf'],
            'surface_pressure': ['sp']
        }
        
        for var_group, var_list in required_vars.items():
            missing = [v for v in var_list if v not in merged_ds]
            if missing:
                print(f"Warning: Missing {var_group}: {missing}")
        
        return merged_ds
    
    def compute_wind_components(self, ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Compute u and v wind components from wind speed and direction.
        
        Args:
            ds: Dataset containing 10m wind data (si10, wdir10)
            
        Returns:
            Tuple of (u10, v10) wind components
        """
        wspd = ds["si10"].values       # Wind speed (m/s)
        wdir = ds["wdir10"].values     # Wind direction (degrees, meteorological)
        
        # Convert direction to radians
        phi = np.deg2rad(wdir)
        
        # Compute wind components
        u10 = -wspd * np.sin(phi)  # Eastward component
        v10 = -wspd * np.cos(phi)  # Northward component
        
        return u10, v10
    
    def compute_friction_velocity(self, ds: xr.Dataset) -> np.ndarray:
        """Compute friction velocity from momentum fluxes and air density.
        
        Args:
            ds: Dataset containing required variables (sp, t2m, r2, tisemf, tisnmf)
            
        Returns:
            Friction velocity array
        """
        
        # Get variables
        p = ds['sp'].values                   # Surface pressure (Pa)
        t2m = ds['t2m'].values                # 2m temperature (K)
        RH = ds['r2'].values                  # 2m relative humidity (%)
        tau_u = ds['tisemf'].values / 3600.0  # U-component momentum flux
        tau_v = ds['tisnmf'].values / 3600.0  # V-component momentum flux
        
        # Compute air density
        Tc = t2m - 273.15  # Temperature in Celsius
        es = 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5)) * 100.0  # Saturation vapor pressure
        e = (RH / 100.0) * es  # Actual vapor pressure
        q = self.eps * e / (p - (1 - self.eps) * e)  # Specific humidity
        Tv = t2m * (1 + 0.61 * q)  # Virtual temperature
        rho = p / (self.Rd * Tv)  # Air density
        
        # Compute the magnitude of the surface stress
        stress_magnitude = np.sqrt(tau_u**2 + tau_v**2)
        
        # Compute friction velocity
        friction_velocity = np.sqrt(stress_magnitude / rho)
        
        return friction_velocity
    
    def compute_sensible_heat_flux(self, ds: xr.Dataset) -> np.ndarray:
        """Compute surface sensible heat flux in W/m².
        
        Args:
            ds: Dataset containing sensible heat flux (sshf)
            
        Returns:
            Sensible heat flux array in W/m²
        """
        sensible_heat_flux = ds['sshf'].values  # J/m²
        # Convert from J/m² to W/m² (divide by 3600 seconds)
        ishf = sensible_heat_flux / 3600
        return ishf
    
    def process_file_to_netcdf(self, input_files: List[str], output_dir: str, year: int) -> None:
        """Process CERRA files (from multiple directories) and save as NetCDF.
        
        Args:
            input_files: List of paths to input GRIB files for the same year
            output_dir: Output directory for processed data
            year: Year for timestamp generation
        """
        if len(input_files) == 1:
            print(f"Processing file: {os.path.basename(input_files[0])}")
        else:
            print(f"Processing {len(input_files)} files for year {year}")
        
        # Load and merge datasets from multiple files
        merged_ds = self.load_datasets(input_files)
        
        # Compute variables
        u10, v10 = self.compute_wind_components(merged_ds)
        t2m = merged_ds['t2m'].values
        sshf = self.compute_sensible_heat_flux(merged_ds)
        zust = self.compute_friction_velocity(merged_ds)
        
        # Get surface pressure if available
        if 'sp' in merged_ds:
            sp = merged_ds['sp'].values
        else:
            print("Warning: Surface pressure not available, using zeros")
            sp = np.zeros_like(t2m)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamps
        num_samples = get_samples_per_year(year)
        start_time = datetime(year, 1, 1, 1)  # Start: 01-01 01:00
        time_step = timedelta(hours=PROCESSING_CONFIG["time_step_hours"])
        
        # Create time coordinate
        time_coord = xr.cftime_range(
            start=start_time,
            periods=num_samples,
            freq=f"{PROCESSING_CONFIG['time_step_hours']}H",
            calendar='standard'
        )
        
        # Get coordinates from one of the datasets
        latitude = merged_ds.latitude
        longitude = merged_ds.longitude
    
        
        # Create NetCDF filename
        netcdf_filename = f"{self.region_config['name']}_{year}_cerra.nc"
        netcdf_filepath = os.path.join(output_dir, netcdf_filename)
        
        print(f"Saving to NetCDF: {netcdf_filepath}")
        
        # Create data arrays for selected variables
        data_vars = {}
        var_data_map = {'u10': u10, 'v10': v10, 't2m': t2m, 'sshf': sshf, 'zust': zust, 'sp': sp}
        
        for var_name in self.variable_names:
            data = var_data_map[var_name]
            
            # Create DataArray with proper coordinates and attributes
            data_vars[var_name] = xr.DataArray(
                data,
                coords={
                    'time': time_coord,
                    'latitude': latitude,
                    'longitude': longitude
                },
                dims=['time', 'latitude', 'longitude'],
                attrs=self._get_variable_attrs(var_name)
            )
        
        # Create dataset
        processed_ds = xr.Dataset(
            data_vars,
            coords={
                'time': time_coord,
                'latitude': latitude,
                'longitude': longitude
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
        print(f"Saved {num_samples} time steps in NetCDF format")
    
    def save_static_data(self, orography_file: str, output_dir: str) -> None:
        """Save static variables (orography) to NetCDF file."""
        print("Processing static variables...")
        
        # Load orography from NetCDF
        ds = xr.open_dataset(orography_file)
        ds = self.standardize_dataset(ds)
        
        # Handle variable name (could be 'orog' from GRIB or 'orography' from already processed)
        if 'orog' in ds:
            orog = ds['orog'].values  # (latitude, longitude)
        elif 'orography' in ds:
            orog = ds['orography'].values  # (latitude, longitude)
        else:
            raise ValueError(f"Orography variable not found in {orography_file}. Available: {list(ds.data_vars.keys())}")
        
        # Create static NetCDF file
        static_filename = f"{self.region_config['name']}_static_cerra.nc"
        static_filepath = os.path.join(output_dir, static_filename)
        
        # Create DataArray for orography
        orography_da = xr.DataArray(
            orog,
            coords={
                'time': ds.time,
                'latitude': ds.latitude,
                'longitude': ds.longitude
            },
            dims=['time', 'latitude', 'longitude'],
            attrs={
                'long_name': 'Surface orography',
                'standard_name': 'surface_altitude',
                'units': 'm',
                'description': 'Surface orography (height above sea level)'
            }
        )
        
        # Create dataset
        static_ds = xr.Dataset(
            {'orography': orography_da},
            coords={
                'time': ds.time,
                'latitude': ds.latitude,
                'longitude': ds.longitude
            },
            attrs={
                'title': f'Static CERRA data for {self.region_config["name"]}',
                'description': 'Static CERRA reanalysis data (orography)',
                'source': 'CERRA reanalysis',
                'institution': 'ECMWF',
                'processing_date': datetime.now().isoformat(),
                'region': self.region,
                'conventions': 'CF-1.8',
                'history': f'Processed from CERRA GRIB files on {datetime.now().isoformat()}'
            }
        )
        
        # Save to NetCDF with compression
        encoding = {
            'orography': {
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
        print(f"Orography shape: {orog.shape}")
    
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
                'description': 'Surface sensible heat flux'
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
            'title': f'Processed CERRA data for {self.region_config["name"]} - {year}',
            'description': f'Processed CERRA reanalysis data for region {self.region}',
            'source': 'CERRA reanalysis',
            'institution': 'ECMWF',
            'processing_date': datetime.now().isoformat(),
            'region': self.region,
            'year': year,
            'variables': self.variable_names,
            'conventions': 'CF-1.8',
            'history': f'Processed from CERRA GRIB files on {datetime.now().isoformat()}'
        }


def process_cerra_data_netcdf(region: str, 
                             dir_suffixes: List[str] = None,
                             year: Optional[int] = None, 
                             input_file: Optional[str] = None,
                             variables: str = "all_variables") -> None:
    """Process CERRA data from multiple input directories and save to NetCDF format.
    
    Args:
        region: Region name (central_europe, iberia, scandinavia)
        dir_suffixes: Directory suffixes to combine (e.g., ['single_levels', 'single_levels_humidity'])
        year: Year to process (optional, if not provided processes all years)
        input_file: Specific file to process (optional)
        variables: Variable group to process ("base_variables" or "all_variables")
    """
    # Default directory suffixes if none provided
    if dir_suffixes is None:
        dir_suffixes = ["single_levels"]
    
    # Initialize processor
    processor = CERRAProcessorNetCDF(region, variables)
    paths = processor.paths
    
    # Get full input directories for this region
    input_directories = get_input_directories_for_region(region, dir_suffixes)
    
    if not input_directories:
        print(f"Error: No valid input directories found for region {region}")
        return
    
    print(f"\nInput directories for region {region}:")
    for d in input_directories:
        print(f"  - {d}")
    
    # Set output directory
    output_dir = os.path.join(str(paths["latlon_proj"]), "single_levels_processed")
    
    # Determine which years to process
    if year:
        years_to_process = [year]
    elif input_file:
        # Extract year from input_file if it's a year file
        try:
            file_year = int(os.path.splitext(input_file)[0])
            years_to_process = [file_year]
        except ValueError:
            print(f"Error: Could not extract year from filename: {input_file}")
            return
    else:
        # Find all available years across all directories
        years_to_process = find_available_years(input_directories)
    
    print(f"Processing {len(years_to_process)} year(s): {years_to_process}")
    print(f"Output directory: {output_dir}")
    
    # Process each year
    for year_to_process in years_to_process:
        print(f"\n{'='*60}")
        print(f"Processing year: {year_to_process}")
        print(f"{'='*60}")
        
        # Collect all files for this year across all directories
        year_files = collect_year_files(year_to_process, input_directories)
        
        if not year_files:
            print(f"Warning: No files found for year {year_to_process}")
            continue
        
        # Process the year
        processor.process_file_to_netcdf(year_files, output_dir, year_to_process)
    
    # Process static data (orography) - look across all directories
    orography_file = find_static_file(input_directories, "orography.nc")
    if orography_file:
        print(f"\nProcessing static data (orography) from: {orography_file}")
        processor.save_static_data(orography_file, output_dir)
    else:
        print("Warning: Orography file not found in any input directory")
    
    # Coordinates are now saved within each NetCDF file, no separate coordinate grid needed
    
    print(f"\n{'='*60}")
    print(f"Processing completed for region: {region}")
    print(f"Output files saved to: {output_dir}")
    print(f"{'='*60}")


def process_all_regions(dir_suffixes: List[str] = None,
                       year: Optional[int] = None, 
                       input_file: Optional[str] = None,
                       variables: str = "base_variables") -> None:
    """Process CERRA data for all regions.
    
    Args:
        dir_suffixes: Directory suffixes to combine (e.g., ['single_levels', 'single_levels_humidity'])
        year: Year to process (optional, if not provided processes all years)
        input_file: Specific file to process (optional)
        variables: Variable group to process ("base_variables" or "all_variables")
    """
    regions = ["central_europe", "iberia", "scandinavia"]
    
    print(f"\n{'='*70}")
    print(f"Processing all regions: {regions}")
    if dir_suffixes:
        print(f"Directory suffixes: {dir_suffixes}")
    if year:
        print(f"Processing year: {year}")
    if input_file:
        print(f"Processing file: {input_file}")
    print(f"{'='*70}")
    
    for region in regions:
        print(f"\n{'='*70}")
        print(f"Processing region: {region}")
        print(f"{'='*70}")
        process_cerra_data_netcdf(region, dir_suffixes, year, input_file, variables)
    
    print(f"\n{'='*70}")
    print("All regions processed successfully!")
    print(f"{'='*70}")


def main():
    """Main function to handle command line arguments and execute processing."""
    parser = argparse.ArgumentParser(
        description="Process CERRA data to NetCDF format with CF-compliant metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific region with humidity data
  python preprocess_cerra.py --region central_europe --year 2021 \\
    --input_directories single_levels single_levels_humidity
  
  # Process all regions with humidity data for specific year
  python preprocess_cerra.py --year 2021 \\
    --input_directories single_levels single_levels_humidity
  
  # Process all regions, all years (with humidity)
  python preprocess_cerra.py \\
    --input_directories single_levels single_levels_humidity
  
  # Process with base variables only
  python preprocess_cerra.py --region iberia --year 2021 \\
    --input_directories single_levels --variables base_variables

Output files (per region):
  - {RegionName}_{year}_cerra.nc   # Year data (e.g., CentralEurope_2021_cerra.nc)
  - {RegionName}_static_cerra.nc   # Static data (orography)
        """
    )
    
    parser.add_argument(
        "--region",
        type=str,
        required=False,
        choices=["central_europe", "iberia", "scandinavia"],
        help="Target region for processing (if not provided, processes all regions)"
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--year",
        type=int,
        help="Year to process (if not provided, processes all years)"
    )
    
    group.add_argument(
        "--input-file",
        type=str,
        help="Specific input file to process (e.g., 2021.grib)"
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
    
    parser.add_argument(
        "--variables",
        type=str,
        default="all_variables",
        choices=["base_variables", "all_variables"],
        help="Variable group to process (default: all_variables)"
    )
    
    args = parser.parse_args()
    
    # Process data
    if args.region:
        # Process specific region
        process_cerra_data_netcdf(
            region=args.region,
            dir_suffixes=args.input_directories,
            year=args.year,
            input_file=args.input_file,
            variables=args.variables
        )
    else:
        # Process all regions
        process_all_regions(
            dir_suffixes=args.input_directories,
            year=args.year,
            input_file=args.input_file,
            variables=args.variables
        )


if __name__ == "__main__":
    main()
