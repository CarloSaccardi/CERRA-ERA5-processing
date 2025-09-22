#!/usr/bin/env python3
"""
CERRA Data Preprocessing Script - HDF5 Output Version

This script processes CERRA GRIB files and saves them directly as HDF5 files
instead of individual .npy files. This is much more efficient for PyTorch loading.

Usage:
    python preprocess_cerra_hdf5.py --region central_europe --year 2021
    python preprocess_cerra_hdf5.py --region iberia --input-file single_2021.grib
"""

import argparse
import xarray as xr
import numpy as np
import h5py
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import time

# Add the current directory to the path to import config
sys.path.append(str(Path(__file__).parent))
from config import get_region_config, get_output_paths, PROCESSING_CONFIG, get_samples_per_year


class CERRAProcessorHDF5:
    """Class to handle CERRA data processing with direct HDF5 output."""
    
    def __init__(self, region: str, variables: str = "base_variables"):
        """Initialize processor for a specific region.
        
        Args:
            region: Region name (central_europe, iberia, scandinavia)
            variables: Variable group to process ("base_variables" or "all_variables")
        """
        self.region = region
        self.region_config = get_region_config(region)
        self.paths = get_output_paths(region)
        
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
        
    def load_datasets(self, input_file: Path) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, 
                                                      xr.Dataset, xr.Dataset]:
        """Load CERRA datasets from GRIB file.
        
        Args:
            input_file: Path to input GRIB file
            
        Returns:
            Tuple of datasets: (surface, 2m, 10m, rho_surface, rho_2m)
        """
        print(f"Loading datasets from: {input_file}")
        
        # Load surface level data
        dssurface = xr.open_dataset(
            input_file,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {
                "typeOfLevel": "surface",
            }},
        )
        
        # Load 2m height data
        ds2 = xr.open_dataset(
            input_file,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {
                "typeOfLevel": "heightAboveGround",
                "level": 2
            }},
        )
        
        # Load 10m height data
        ds10 = xr.open_dataset(
            input_file,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {
                "typeOfLevel": "heightAboveGround",
                "level": 10
            }},
        )
        
        # Load rho variables (surface pressure and 2m relative humidity)
        try:
            drho_sp = xr.open_dataset(
                input_file,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {
                    "typeOfLevel": "surface",
                    "shortName": "sp"  # surface pressure
                }},
            )
        except:
            print("Warning: Surface pressure not found in the file")
            drho_sp = None
            
        try:
            drho_r2 = xr.open_dataset(
                input_file,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {
                    "typeOfLevel": "heightAboveGround",
                    "level": 2,
                    "shortName": "r2"  # 2m relative humidity
                }},
            )
        except:
            print("Warning: 2m relative humidity not found in the file")
            drho_r2 = None
        
        return dssurface, ds2, ds10, drho_sp, drho_r2
    
    def compute_wind_components(self, ds10: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Compute u and v wind components from wind speed and direction.
        
        Args:
            ds10: Dataset containing 10m wind data
            
        Returns:
            Tuple of (u10, v10) wind components
        """
        wspd = ds10["si10"]       # Wind speed (m/s)
        wdir = ds10["wdir10"]     # Wind direction (degrees, meteorological)
        
        # Convert direction to radians
        phi = np.deg2rad(wdir)
        
        # Compute wind components
        u10 = -wspd * np.sin(phi)  # Eastward component
        v10 = -wspd * np.cos(phi)  # Northward component
        
        return u10.values, v10.values
    
    def compute_friction_velocity(self, dssurface: xr.Dataset, ds2: xr.Dataset,
                                 drho_sp: xr.Dataset, drho_r2: xr.Dataset) -> np.ndarray:
        """Compute friction velocity from momentum fluxes and air density.
        
        Args:
            dssurface: Surface dataset
            ds2: 2m dataset
            drho_sp: Surface pressure dataset
            drho_r2: 2m relative humidity dataset
            
        Returns:
            Friction velocity array
        """
        if drho_sp is None or drho_r2 is None:
            print("Warning: Cannot compute friction velocity - missing density variables")
            return np.zeros_like(dssurface['tisemf'].values)
        
        # Get variables
        p = drho_sp['sp']             # Surface pressure (Pa)
        t2m = ds2['t2m']              # 2m temperature (K)
        RH = drho_r2['r2']            # 2m relative humidity (%)
        tau_u = dssurface['tisemf']   # U-component momentum flux
        tau_v = dssurface['tisnmf']   # V-component momentum flux
        
        # Compute air density
        Tc = t2m - 273.15  # Temperature in Celsius
        es = 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5)) * 100.0  # Saturation vapor pressure
        e = (RH / 100.0) * es  # Actual vapor pressure
        q = self.eps * e / (p - (1 - self.eps) * e)  # Specific humidity
        Tv = t2m * (1 + 0.61 * q)  # Virtual temperature
        rho = p / (self.Rd * Tv)  # Air density
        
        # Compute friction velocity
        friction_velocity = np.sqrt(np.sqrt(tau_u**2 + tau_v**2) / rho)
        friction_velocity = friction_velocity / 3600
        
        return friction_velocity.values
    
    def compute_sensible_heat_flux(self, dssurface: xr.Dataset) -> np.ndarray:
        """Compute surface sensible heat flux in W/m².
        
        Args:
            dssurface: Surface dataset
            
        Returns:
            Sensible heat flux array in W/m²
        """
        sensible_heat_flux = dssurface['sshf']  # J/m²
        # Convert from J/m² to W/m² (divide by 3600 seconds)
        ishf = sensible_heat_flux / 3600
        return ishf.values
    
    def process_file_to_hdf5(self, input_file: Path, output_dir: Path, year: int) -> None:
        """Process a single CERRA file and save as HDF5.
        
        Args:
            input_file: Path to input GRIB file
            output_dir: Output directory for processed data
            year: Year for timestamp generation
        """
        print(f"Processing file: {input_file.name}")
        
        # Load datasets
        dssurface, ds2, ds10, drho_sp, drho_r2 = self.load_datasets(input_file)
        
        # Compute variables
        u10, v10 = self.compute_wind_components(ds10)
        t2m = ds2['t2m'].values
        sshf = self.compute_sensible_heat_flux(dssurface)
        zust = self.compute_friction_velocity(dssurface, ds2, drho_sp, drho_r2)
        
        # Get surface pressure if available
        if drho_sp is not None:
            sp = drho_sp['sp'].values
        else:
            print("Warning: Surface pressure not available, using zeros")
            sp = np.zeros_like(t2m)
        
        # Stack all data first (we'll extract only what we need later)
        all_data_list = [u10, v10, t2m, sshf, zust, sp]
        all_var_names = ['u10', 'v10', 't2m', 'sshf', 'zust', 'sp']
        all_stacked_data = np.stack(all_data_list, axis=-1)  # (time, height, width, 6)
        
        # Extract selected variables based on configuration
        selected_data = all_stacked_data[:, :, :, self.variable_indices]  # (time, height, width, n_vars)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamps
        num_samples = get_samples_per_year(year)
        start_time = datetime(year, 1, 1, 1)  # Start: 01-01 01:00
        time_step = timedelta(hours=PROCESSING_CONFIG["time_step_hours"])
        time_values = [start_time + i * time_step for i in range(num_samples)]
        
        # Create HDF5 file
        hdf5_filename = f"{self.region_config['name']}_{year}_cerra.h5"
        hdf5_filepath = output_dir / hdf5_filename
        
        print(f"Saving to HDF5: {hdf5_filepath}")
        print(f"Data shape: {selected_data.shape}")
        
        with h5py.File(hdf5_filepath, 'w') as f:
            # Create main dataset
            data_dset = f.create_dataset(
                'data',
                data=selected_data,
                dtype=np.float32,
                chunks=(1, selected_data.shape[1], selected_data.shape[2], selected_data.shape[3]),
                compression='gzip',
                compression_opts=6
            )
            
            # Create timestamps dataset
            timestamps = [ts.strftime("%Y%m%d%H") for ts in time_values]
            f.create_dataset(
                'timestamps',
                data=[ts.encode('utf-8') for ts in timestamps],
                dtype='S10',
                compression='gzip'
            )
            
            # Create indices dataset
            f.create_dataset(
                'indices',
                data=np.arange(len(time_values)),
                dtype=np.int64,
                compression='gzip'
            )
            
            # Store metadata
            metadata_group = f.create_group('metadata')
            metadata_group.create_dataset('variable_names', 
                                        data=[name.encode('utf-8') for name in self.variable_names])
            metadata_group.create_dataset('variable_indices', data=self.variable_indices)
            metadata_group.create_dataset('all_variable_names', 
                                        data=[name.encode('utf-8') for name in all_var_names])
            
            # Store conversion metadata
            metadata_group.attrs['conversion_date'] = datetime.now().isoformat()
            metadata_group.attrs['year'] = year
            metadata_group.attrs['region'] = self.region
            metadata_group.attrs['n_samples'] = len(time_values)
            metadata_group.attrs['data_shape'] = selected_data.shape
        
        # Print file size information
        file_size_mb = hdf5_filepath.stat().st_size / (1024 * 1024)
        print(f"HDF5 file created: {hdf5_filepath}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Saved {len(time_values)} time steps in HDF5 format")
    
    def save_coordinate_grid(self, input_file: Path, output_dir: Path) -> None:
        """Extract and save the longitude and latitude grid.
        
        Args:
            input_file: Path to input GRIB file
            output_dir: Output directory for coordinate file
        """
        print("Extracting coordinate grid...")
        
        ds = xr.open_dataset(
            input_file,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {
                "typeOfLevel": "surface",
            }},
        )
        
        lon = ds['longitude'].values
        lat = ds['latitude'].values
        
        if len(lon.shape) == 1:
            lon, lat = np.meshgrid(lon, lat)
        
        xy = np.array([lon, lat])  # shape = (2, n_lon, n_lat)
        save_path = output_dir / "nwp_xy.npy"
        np.save(save_path, xy)
        print(f"Coordinate grid saved to: {save_path}")


def process_cerra_data_hdf5(region: str, year: Optional[int] = None, 
                           input_file: Optional[str] = None,
                           input_dir: Optional[Path] = None,
                           variables: str = "base_variables") -> None:
    """Process CERRA data for a specific region and save as HDF5.
    
    Args:
        region: Region name
        year: Year to process (optional)
        input_file: Specific file to process (optional)
        input_dir: Input directory (optional)
        variables: Variable group to process ("base_variables" or "all_variables")
    """
    processor = CERRAProcessorHDF5(region, variables)
    paths = processor.paths
    
    # Set input directory
    if input_dir is None:
        input_dir = paths["latlon_proj"] / "single_levels"
    
    # Get input files
    if input_file:
        input_files = [input_dir / input_file]
    elif year:
        input_files = [input_dir / f"single_{year}.grib"]
    else:
        # Find all single_*.grib files
        input_files = list(input_dir.glob("single_*.grib"))
    
    if not input_files:
        print("No input files found to process.")
        return
    
    # Create output directory
    output_dir = paths["processed_data"]
    
    print(f"Processing {len(input_files)} files for region: {region}")
    print(f"Output directory: {output_dir}")
    
    # Process each file
    for input_file_path in input_files:
        if not input_file_path.exists():
            print(f"Warning: File not found: {input_file_path}")
            continue
        
        # Extract year from filename if not provided
        file_year = year
        if file_year is None:
            # Try to extract year from filename
            filename = input_file_path.stem
            if 'single_' in filename:
                year_str = filename.replace('single_', '').split('.')[0]
                try:
                    file_year = int(year_str)
                except ValueError:
                    print(f"Warning: Could not extract year from filename: {filename}")
                    continue
        
        if file_year is None:
            print(f"Warning: Could not determine year for file: {input_file_path}")
            continue
        
        # Process the file
        processor.process_file_to_hdf5(input_file_path, output_dir, file_year)
        
        # Save coordinate grid (only for the first file)
        if input_file_path == input_files[0]:
            processor.save_coordinate_grid(input_file_path, output_dir)


def main():
    """Main function to handle command line arguments and execute processing."""
    parser = argparse.ArgumentParser(
        description="Process CERRA data to HDF5 format (u10, v10, t2m for base_variables; u10, v10, t2m, sshf, zust, sp for all_variables)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process data for central Europe for 2021
  python preprocess_cerra.py --region central_europe --year 2021
  
  # Process a specific file for Iberia
  python preprocess_cerra.py --region iberia --input-file single_2021.grib
  
  # Process all available files for Scandinavia
  python preprocess_cerra.py --region scandinavia
        """
    )
    
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        choices=["central_europe", "iberia", "scandinavia"],
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
        help="Input directory (default: latlon_proj_{region}/single_levels)"
    )
    
    parser.add_argument(
        "--variables",
        type=str,
        default="base_variables",
        choices=["base_variables", "all_variables"],
        help="Variable group to process (default: base_variables)"
    )
    
    args = parser.parse_args()
    
    # Set input directory
    input_dir = None
    if args.input_dir:
        input_dir = Path(args.input_dir)
    
    # Process data
    process_cerra_data_hdf5(
        region=args.region,
        year=args.year,
        input_file=args.input_file,
        input_dir=input_dir,
        variables=args.variables
    )


if __name__ == "__main__":
    main()
