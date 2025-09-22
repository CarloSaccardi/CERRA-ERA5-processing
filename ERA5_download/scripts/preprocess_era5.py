#!/usr/bin/env python3
"""
ERA5 Data Preprocessing Script - HDF5 Output Version

This script processes ERA5 NetCDF files and saves them directly as HDF5 files
instead of individual .npy files. This is much more efficient for PyTorch loading.

Usage:
    python preprocess_era5.py --region central_europe --year 2021
    python preprocess_era5.py --region iberia --input-file single_2021.nc
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


class ERA5ProcessorHDF5:
    """Class to handle ERA5 data processing with direct HDF5 output."""
    
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
        
    def load_dataset(self, input_file: Path) -> xr.Dataset:
        """Load ERA5 dataset from NetCDF file.
        
        Args:
            input_file: Path to input NetCDF file
            
        Returns:
            ERA5 dataset
        """
        print(f"Loading dataset from: {input_file}")
        
        # Load ERA5 data
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
        if "all_variables" in str(self.variable_names):
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
        if "all_variables" in str(self.variable_names):
            print(f"  sshf: {sshf.shape}")
            print(f"  zust: {zust.shape}")
            print(f"  sp: {sp.shape}")
        
        return u10, v10, t2m, sshf, zust, sp
    
    def process_file_to_hdf5(self, input_file: Path, output_dir: Path, year: int) -> None:
        """Process a single ERA5 file and save as HDF5.
        
        Args:
            input_file: Path to input NetCDF file
            output_dir: Output directory for processed data
            year: Year for timestamp generation
        """
        print(f"Processing file: {input_file.name}")
        
        # Load dataset
        ds = self.load_dataset(input_file)
        
        # Extract variables
        u10, v10, t2m, sshf, zust, sp = self.extract_variables(ds)
        
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
        hdf5_filename = f"{self.region_config['name']}_{year}_era5.h5"
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
            input_file: Path to input NetCDF file
            output_dir: Output directory for coordinate file
        """
        print("Extracting coordinate grid...")
        
        ds = xr.open_dataset(input_file)
        
        lon = ds['longitude'].values
        lat = ds['latitude'].values
        
        if len(lon.shape) == 1:
            lon, lat = np.meshgrid(lon, lat)
        
        xy = np.array([lon, lat])  # shape = (2, n_lon, n_lat)
        save_path = output_dir / "nwp_xy.npy"
        np.save(save_path, xy)
        print(f"Coordinate grid saved to: {save_path}")


def process_era5_data_hdf5(region: str, year: Optional[int] = None, 
                          input_file: Optional[str] = None,
                          input_dir: Optional[Path] = None,
                          variables: str = "base_variables") -> None:
    """Process ERA5 data for a specific region and save as HDF5.
    
    Args:
        region: Region name
        year: Year to process (optional)
        input_file: Specific file to process (optional)
        input_dir: Input directory (optional)
        variables: Variable group to process ("base_variables" or "all_variables")
    """
    processor = ERA5ProcessorHDF5(region, variables)
    paths = processor.paths
    
    # Set input directory
    if input_dir is None:
        input_dir = paths["raw_data"]
    
    # Get input files
    if input_file:
        input_files = [input_dir / input_file]
    elif year:
        input_files = [input_dir / f"single_{year}.nc"]
    else:
        # Find all single_*.nc files
        input_files = list(input_dir.glob("single_*.nc"))
    
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
        description="Process ERA5 data to HDF5 format (u10, v10, t2m for base_variables; u10, v10, t2m, sshf, zust, sp for all_variables)",
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
        help="Input directory (default: {region}/single_levels)"
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
    process_era5_data_hdf5(
        region=args.region,
        year=args.year,
        input_file=args.input_file,
        input_dir=input_dir,
        variables=args.variables
    )


if __name__ == "__main__":
    main()
