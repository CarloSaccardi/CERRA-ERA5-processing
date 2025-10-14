#!/usr/bin/env python3
"""
ERA5 Data Splitting Script

This script creates train/validation/test splits from processed ERA5 data across
multiple regions. The script combines data from different regions into single files
with an added 'region' dimension.

The script automatically discovers available regions by scanning the input directory
for subdirectories containing 'single_levels_processed' folders.

Split configuration can be provided via:
1. A JSON configuration file (--config)
2. Command-line arguments for the default split strategy

Output Structure:
- train.nc: (region=N, time, lat, lon)
- val.nc: (region=N, time, lat, lon)
- test.nc: (region=N, time, lat, lon)
- static.nc: (region=N, lat, lon)

Usage:
    # Use default split strategy
    python split_data.py --input-dir /path/to/ERA5_download --output-dir /path/to/output
    
    # Use custom configuration file
    python split_data.py --config split_config.json --input-dir /path/to/ERA5_download
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import xarray as xr
import numpy as np


class ERA5DataSplitter:
    """Class to handle splitting ERA5 data across regions into train/val/test sets."""
    
    # Month definitions
    ODD_MONTHS = [1, 3, 5, 7, 9, 11]
    EVEN_MONTHS = [2, 4, 6, 8, 10, 12]
    
    def __init__(self, input_dir: Path, output_dir: Path, split_config: Dict, 
                 compression_level: int = 6, task: str = 'downscaling'):
        """Initialize the data splitter.
        
        Args:
            input_dir: Root directory containing regional data subdirectories
            output_dir: Directory to save split datasets
            split_config: Configuration dictionary defining the split strategy
            compression_level: NetCDF compression level (1-9)
            task: Task type ('downscaling' or 'forecasting')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.split_config = split_config
        self.compression_level = compression_level
        self.task = task
        
        # Extract region directories from configuration
        self.regions = self._extract_region_directories()
        
        # Validate configuration
        self._validate_config()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized ERA5 Data Splitter")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Configured regions: {list(self.regions.keys())}")
        print(f"Task type: {self.task}")
        print(f"Compression level: {self.compression_level}")
    
    def _extract_region_directories(self) -> Dict[str, Path]:
        """Extract region directories from configuration.
        
        Returns:
            Dictionary mapping region names to their processed data directories
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        regions = {}
        
        # Collect all regions from all splits and extract their directories
        for split_name in ['train', 'val', 'test']:
            if split_name in self.split_config:
                for region, region_config in self.split_config[split_name].items():
                    if region not in regions:
                        # Get directory from config
                        if 'directory' not in region_config:
                            raise ValueError(
                                f"Region '{region}' in '{split_name}' missing required 'directory' field"
                            )
                        
                        region_dir = self.input_dir / region_config['directory']
                        
                        # Validate directory exists
                        if not region_dir.exists():
                            raise FileNotFoundError(
                                f"Directory for region '{region}' not found: {region_dir}"
                            )
                        
                        regions[region] = region_dir
                        print(f"✓ Region '{region}': {region_dir}")
        
        if not regions:
            raise ValueError("No regions found in configuration")
        
        return regions
    
    def _get_available_years(self, region: str) -> List[int]:
        """Get available years for a specific region.
        
        Args:
            region: Region name
            
        Returns:
            Sorted list of available years
        """
        region_dir = self.regions[region]
        years = []
        
        # Look for files matching pattern {region}_{year}_era5.nc
        for file in region_dir.glob(f"{region}_*_era5.nc"):
            # Extract year from filename
            try:
                parts = file.stem.split('_')
                if len(parts) >= 2 and parts[-1] == 'era5':
                    year = int(parts[-2])
                    years.append(year)
            except (ValueError, IndexError):
                continue
        
        return sorted(years)
    
    def _validate_config(self) -> None:
        """Validate that the split configuration is valid."""
        # Check required keys
        required_keys = ['train', 'val', 'test']
        for key in required_keys:
            if key not in self.split_config:
                raise ValueError(f"Split configuration missing required key: '{key}'")
        
        # Validate that each region config has required fields
        for split_name in required_keys:
            split_data = self.split_config[split_name]
            for region, region_config in split_data.items():
                # Check directory field exists (already checked in _extract_region_directories, but good to be explicit)
                if 'directory' not in region_config:
                    raise ValueError(
                        f"Region '{region}' in '{split_name}' missing required 'directory' field"
                    )
                
                # Check that at least 'years' or 'year_months' is specified
                if 'years' not in region_config and 'year_months' not in region_config:
                    raise ValueError(
                        f"Region '{region}' in '{split_name}' must specify either 'years' or 'year_months'"
                    )
        
        print("✓ Configuration validated")
    
    def _get_file_path(self, region: str, year: int) -> Path:
        """Get the file path for a specific region and year.
        
        Args:
            region: Region name
            year: Year
            
        Returns:
            Path to the NetCDF file
        """
        region_dir = self.regions[region]
        filename = f"{region}_{year}_era5.nc"
        filepath = region_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
        
        return filepath
    
    def _get_static_file_path(self, region: str) -> Path:
        """Get the static file path for a specific region.
        
        Args:
            region: Region name
            
        Returns:
            Path to the static NetCDF file
        """
        region_dir = self.regions[region]
        filename = f"{region}_static_era5.nc"
        filepath = region_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Required static file not found: {filepath}")
        
        return filepath
    
    def _extract_months(self, ds: xr.Dataset, months: List[int]) -> xr.Dataset:
        """Extract specific months from a dataset.
        
        Args:
            ds: Dataset with time dimension
            months: List of month numbers to extract
            
        Returns:
            Dataset containing only the specified months
        """
        # Get month from time coordinate
        time_months = ds.time.dt.month.values
        
        # Create mask for specified months
        mask = np.isin(time_months, months)
        
        # Select data for specified months
        ds_filtered = ds.isel(time=mask)
        
        return ds_filtered
    
    def _load_year_data(self, region: str, year: int) -> xr.Dataset:
        """Load data for a specific region and year.
        
        Args:
            region: Region name
            year: Year to load
            
        Returns:
            Dataset for the specified year
        """
        filepath = self._get_file_path(region, year)
        print(f"  Loading {region} {year}: {filepath.name}")
        
        ds = xr.open_dataset(filepath)
        return ds
    
    def _concatenate_years(self, region: str, years: List[int]) -> xr.Dataset:
        """Concatenate data from multiple years for a region.
        
        Args:
            region: Region name
            years: List of years to concatenate
            
        Returns:
            Concatenated dataset
        """
        datasets = []
        for year in years:
            ds = self._load_year_data(region, year)
            datasets.append(ds)
        
        # Concatenate along time dimension
        combined = xr.concat(datasets, dim='time')
        
        return combined
    
    def _process_region_for_split(self, region: str, region_config: Dict) -> xr.Dataset:
        """Process data for a single region according to split configuration.
        
        Args:
            region: Region name
            region_config: Configuration dict with keys like 'years', 'months', 'year_months'
            
        Returns:
            Dataset for this region
        """
        datasets_to_concat = []
        
        # Process full years
        if 'years' in region_config and region_config['years']:
            years = region_config['years']
            print(f"  Loading full years: {years}")
            ds_years = self._concatenate_years(region, years)
            datasets_to_concat.append(ds_years)
        
        # Process year-month combinations (e.g., {2020: [1,3,5,7,9,11]})
        if 'year_months' in region_config:
            for year, months in region_config['year_months'].items():
                year = int(year)  # Ensure year is int (JSON keys are strings)
                print(f"  Loading {year} (months: {months})")
                ds_year = self._load_year_data(region, year)
                ds_filtered = self._extract_months(ds_year, months)
                datasets_to_concat.append(ds_filtered)
        
        # Concatenate all datasets for this region
        if len(datasets_to_concat) == 1:
            return datasets_to_concat[0]
        else:
            return xr.concat(datasets_to_concat, dim='time')
    
    def create_training_data(self) -> List[Tuple[str, xr.Dataset]]:
        """Create training datasets for all regions according to configuration.
        
        Returns:
            List of (region_name, dataset) tuples
        """
        print("\n" + "="*60)
        print("Creating Training Datasets")
        print("="*60)
        
        train_config = self.split_config['train']
        region_datasets = []
        
        for region, region_config in train_config.items():
            print(f"\nProcessing {region}:")
            ds_region = self._process_region_for_split(region, region_config)
            print(f"  Total timesteps for {region}: {len(ds_region.time)}")
            
            # Add metadata
            ds_region.attrs.update({
                'title': f'ERA5 Training Dataset - {region}',
                'description': f'Training data for {region}',
                'region': region,
                'split': 'train',
                'creation_date': datetime.now().isoformat(),
                'creator': 'split_data.py',
                'configuration': json.dumps(region_config)
            })
            
            region_datasets.append((region, ds_region))
        
        print(f"\n✓ Training datasets created for {len(region_datasets)} regions")
        
        return region_datasets
    
    def create_validation_data(self) -> List[Tuple[str, xr.Dataset]]:
        """Create validation datasets for all regions according to configuration.
        
        Returns:
            List of (region_name, dataset) tuples
        """
        print("\n" + "="*60)
        print("Creating Validation Datasets")
        print("="*60)
        
        val_config = self.split_config['val']
        region_datasets = []
        
        for region, region_config in val_config.items():
            print(f"\nProcessing {region}:")
            ds_region = self._process_region_for_split(region, region_config)
            print(f"  Total timesteps for {region}: {len(ds_region.time)}")
            
            # Add metadata
            ds_region.attrs.update({
                'title': f'ERA5 Validation Dataset - {region}',
                'description': f'Validation data for {region}',
                'region': region,
                'split': 'validation',
                'creation_date': datetime.now().isoformat(),
                'creator': 'split_data.py',
                'configuration': json.dumps(region_config)
            })
            
            region_datasets.append((region, ds_region))
        
        print(f"\n✓ Validation datasets created for {len(region_datasets)} regions")
        
        return region_datasets
    
    def create_test_data(self) -> List[Tuple[str, xr.Dataset]]:
        """Create test datasets for all regions according to configuration.
        
        Returns:
            List of (region_name, dataset) tuples
        """
        print("\n" + "="*60)
        print("Creating Test Datasets")
        print("="*60)
        
        test_config = self.split_config['test']
        region_datasets = []
        
        for region, region_config in test_config.items():
            print(f"\nProcessing {region}:")
            ds_region = self._process_region_for_split(region, region_config)
            print(f"  Total timesteps for {region}: {len(ds_region.time)}")
            
            # Add metadata
            ds_region.attrs.update({
                'title': f'ERA5 Test Dataset - {region}',
                'description': f'Test data for {region}',
                'region': region,
                'split': 'test',
                'creation_date': datetime.now().isoformat(),
                'creator': 'split_data.py',
                'configuration': json.dumps(region_config)
            })
            
            region_datasets.append((region, ds_region))
        
        print(f"\n✓ Test datasets created for {len(region_datasets)} regions")
        
        return region_datasets
    
    def create_static_data(self) -> List[Tuple[str, xr.Dataset]]:
        """Load static data for all regions in configuration.
        
        Returns:
            List of (region_name, dataset) tuples
        """
        print("\n" + "="*60)
        print("Loading Static Datasets")
        print("="*60)
        
        # Collect all regions from all splits
        all_regions = set()
        for split_name in ['train', 'val', 'test']:
            if split_name in self.split_config:
                all_regions.update(self.split_config[split_name].keys())
        
        region_datasets = []
        region_names = sorted(all_regions)  # Sort for consistent ordering
        
        for region in region_names:
            print(f"\nProcessing {region}:")
            filepath = self._get_static_file_path(region)
            print(f"  Loading: {filepath.name}")
            
            ds = xr.open_dataset(filepath)
            
            # Add metadata
            ds.attrs.update({
                'title': f'ERA5 Static Data - {region}',
                'description': f'Static variables (geopotential) for {region}',
                'region': region,
                'creation_date': datetime.now().isoformat(),
                'creator': 'split_data.py'
            })
            
            region_datasets.append((region, ds))
        
        print(f"\n✓ Static datasets loaded for {len(region_datasets)} regions")
        
        return region_datasets
    
    def _save_region_dataset(self, ds: xr.Dataset, region_name: str, split_name: str) -> None:
        """Save dataset for a single region to NetCDF with compression.
        
        Args:
            ds: Dataset to save
            region_name: Name of the region
            split_name: Name of split (train, val, test)
        """
        # Create split subdirectory
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output path
        output_path = split_dir / f"{region_name}.nc"
        
        print(f"\nSaving to: {output_path}")
        
        # Create encoding with compression for all variables
        encoding = {}
        for var in ds.data_vars:
            # Determine chunk sizes based on dimensions and task type
            chunks = []
            for dim in ds[var].dims:
                if dim == 'time':
                    if self.task == 'downscaling':
                        # Each timestep is independent sample (optimal for random access)
                        chunks.append(1)
                    elif self.task == 'forecasting':
                        raise NotImplementedError(
                            "Forecasting task chunking not yet configured. "
                            "Please refer to the chunking documentation to set appropriate "
                            "time chunk sizes for your forecasting workflow. "
                            "Typical values: 30-90 for batch-based forecasting, "
                            "or 365 for large sequential reads."
                        )
                    else:
                        raise ValueError(f"Unknown task type: {self.task}")
                else:
                    # Keep spatial dimensions unchunked for better access patterns
                    chunks.append(ds.dims[dim])
            
            encoding[var] = {
                'zlib': True,
                'complevel': self.compression_level,
                'dtype': 'float32',
                'chunksizes': tuple(chunks)
            }
        
        # Save to NetCDF
        ds.to_netcdf(
            output_path,
            encoding=encoding,
            format='NETCDF4'
        )
        
        # Report file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ File saved: {output_path.relative_to(self.output_dir)}")
        print(f"  Size: {file_size_mb:.1f} MB")
    
    def _save_static_datasets(self, static_datasets: List[Tuple[str, xr.Dataset]], 
                             split_dirs: List[str]) -> None:
        """Save static datasets to all split directories.
        
        Args:
            static_datasets: List of (region_name, dataset) tuples
            split_dirs: List of split directory names to save to
        """
        print("\n" + "="*60)
        print("Saving Static Datasets")
        print("="*60)
        
        for split_name in split_dirs:
            # Create split subdirectory
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nSaving to {split_name}/ directory:")
            
            for region_name, ds in static_datasets:
                output_path = split_dir / f"static_{region_name}.nc"
                
                # Create encoding with compression
                encoding = {}
                for var in ds.data_vars:
                    encoding[var] = {
                        'zlib': True,
                        'complevel': self.compression_level,
                        'dtype': 'float32'
                    }
                
                # Save to NetCDF
                ds.to_netcdf(
                    output_path,
                    encoding=encoding,
                    format='NETCDF4'
                )
                
                # Report file size
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {output_path.relative_to(self.output_dir)} ({file_size_mb:.1f} MB)")
        
        print(f"\n✓ Static datasets saved to {len(split_dirs)} directories")
    
    def run(self, skip_train: bool = False, skip_val: bool = False, 
            skip_test: bool = False, skip_static: bool = False) -> None:
        """Run the complete data splitting pipeline.
        
        Args:
            skip_train: Skip creating training data
            skip_val: Skip creating validation data
            skip_test: Skip creating test data
            skip_static: Skip creating static data
        """
        print("\n" + "="*60)
        print("ERA5 Data Splitting Pipeline")
        print("="*60)
        
        try:
            # Track which splits were created
            created_splits = []
            
            # Create training data
            if not skip_train:
                train_datasets = self.create_training_data()
                for region_name, ds in train_datasets:
                    self._save_region_dataset(ds, region_name, 'train')
                    ds.close()
                created_splits.append('train')
            else:
                print("\n⊘ Skipping training data")
            
            # Create validation data
            if not skip_val:
                val_datasets = self.create_validation_data()
                for region_name, ds in val_datasets:
                    self._save_region_dataset(ds, region_name, 'val')
                    ds.close()
                created_splits.append('val')
            else:
                print("\n⊘ Skipping validation data")
            
            # Create test data
            if not skip_test:
                test_datasets = self.create_test_data()
                for region_name, ds in test_datasets:
                    self._save_region_dataset(ds, region_name, 'test')
                    ds.close()
                created_splits.append('test')
            else:
                print("\n⊘ Skipping test data")
            
            # Create static data and save to all created split directories
            if not skip_static and created_splits:
                static_datasets = self.create_static_data()
                self._save_static_datasets(static_datasets, created_splits)
                # Close all static datasets
                for _, ds in static_datasets:
                    ds.close()
            elif not skip_static:
                print("\n⊘ No splits created, skipping static data")
            else:
                print("\n⊘ Skipping static data")
            
            print("\n" + "="*60)
            print("✓ Data splitting completed successfully!")
            print("="*60)
            print(f"\nOutput directory structure:")
            print(f"  {self.output_dir}/")
            for split in created_splits:
                print(f"    {split}/")
                print(f"      {{region}}.nc")
                if not skip_static:
                    print(f"      static_{{region}}.nc")
            
        except Exception as e:
            print(f"\n✗ Error during data splitting: {e}")
            raise


def load_config(config_file: Optional[Path]) -> Dict:
    """Load configuration from file.
    
    Args:
        config_file: Path to JSON configuration file (required)
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config_file is None or doesn't exist
    """
    if config_file is None:
        raise FileNotFoundError(
            "Configuration file is required. "
            "Please provide a configuration file using --config option. "
            "See split_config_example.json for an example."
        )
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    print(f"Loading configuration from: {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def main():
    """Main function to handle command line arguments and execute splitting."""
    parser = argparse.ArgumentParser(
        description="Split ERA5 data into train/validation/test sets across multiple regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Format (JSON):
{
  "train": {
    "RegionName": {
      "directory": "RegionName/single_levels_processed",
      "years": [2013, 2014, 2015],
      "year_months": {"2020": [1, 3, 5]}
    }
  },
  "val": { ... },
  "test": { ... }
}

Each region must specify:
  - directory: Relative path from input-dir to the processed data directory
  - years (optional): List of full years to include
  - year_months (optional): Dictionary mapping years to lists of months

See split_config_example.json for a complete example.

Default Split Strategy:
  Training:   CE(2013-2015) + Iberia(2015-2017) + Scandinavia(2017-2019) + All(odd months 2020)
  Validation: All regions (even months 2020)
  Test:       All regions (all months 2021)

Output Structure (default: processed_data/ERA5/):
  train/
    CentralEurope.nc
    Iberia.nc
    Scandinavia.nc
    static_CentralEurope.nc
    static_Iberia.nc
    static_Scandinavia.nc
  val/
    CentralEurope.nc
    Iberia.nc
    Scandinavia.nc
    static_CentralEurope.nc
    static_Iberia.nc
    static_Scandinavia.nc
  test/
    CentralEurope.nc
    Iberia.nc
    Scandinavia.nc
    static_CentralEurope.nc
    static_Iberia.nc
    static_Scandinavia.nc

Examples:
  # Use example configuration for downscaling (default)
  python split_data.py --config split_config_example.json
  
  # Use custom configuration file
  python split_data.py --config my_split_config.json
  
  # For forecasting task (will show error with chunking guidance)
  python split_data.py --config my_config.json --task forecasting
  
  # Specify custom paths
  python split_data.py --config my_config.json --input-dir /path/to/ERA5_download --output-dir /path/to/output
  
  # Create only training data
  python split_data.py --config my_config.json --skip-val --skip-test --skip-static
  
  # Use higher compression
  python split_data.py --config my_config.json --compression-level 9
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file (required, see split_config_example.json)"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Root directory containing regional ERA5 data subdirectories (default: ../ERA5_download relative to script)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save split datasets (default: INPUT_DIR/processed_data/ERA5)"
    )
    
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        choices=range(1, 10),
        help="NetCDF compression level (1-9, default: 4)"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="downscaling",
        choices=["downscaling", "forecasting"],
        help="Task type: 'downscaling' (chunk=1) or 'forecasting' (see docs, default: downscaling)"
    )
    
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip creating training dataset"
    )
    
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Skip creating validation dataset"
    )
    
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip creating test dataset"
    )
    
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip creating static dataset"
    )
    
    args = parser.parse_args()
    
    # Set default input directory (parent of scripts directory)
    if args.input_dir is None:
        script_dir = Path(__file__).parent
        input_dir = script_dir.parent  # ERA5_download directory
    else:
        input_dir = Path(args.input_dir)
    
    # Set default output directory
    if args.output_dir is None:
        output_dir = input_dir / "processed_data" / "ERA5"
    else:
        output_dir = Path(args.output_dir)
    
    # Load configuration
    try:
        config_file = Path(args.config) if args.config else None
        split_config = load_config(config_file)
    except Exception as e:
        print(f"\n✗ Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create splitter and run
    try:
        splitter = ERA5DataSplitter(
            input_dir=input_dir,
            output_dir=output_dir,
            split_config=split_config,
            compression_level=args.compression_level,
            task=args.task
        )
        
        splitter.run(
            skip_train=args.skip_train,
            skip_val=args.skip_val,
            skip_test=args.skip_test,
            skip_static=args.skip_static
        )
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

