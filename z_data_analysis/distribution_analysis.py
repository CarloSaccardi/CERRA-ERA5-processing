#!/usr/bin/env python3
"""
Temperature Distribution Analysis Script

This script analyzes the temperature distributions across different regions
(Central Europe, Iberia, and Scandinavia) using test data from processed_data/ERA5/test.

It creates normalized histograms and statistical comparisons to understand
how temperature patterns differ between regions.

Usage:
    python analyze_temperature_distributions.py --data-dir processed_data/ERA5/test
    python analyze_temperature_distributions.py --data-dir processed_data/ERA5/test --output-dir results
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VariableAnalyzer:
    """Class to analyze variable distributions across regions."""
    
    def __init__(self, data_dir: Path, output_dir: Path, variable: str = 't2m', 
                 clip_percentiles: bool = True):
        """Initialize the analyzer.
        
        Args:
            data_dir: Directory containing region NetCDF files
            output_dir: Directory to save analysis results
            variable: Variable name to analyze (e.g., 't2m', 'u10', 'v10')
            clip_percentiles: Whether to clip 1st and 99th percentiles
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.variable = variable
        self.clip_percentiles = clip_percentiles
        
        # Expected regions
        self.regions = ['CentralEurope', 'Iberia', 'Scandinavia']
        
        print(f"Variable Distribution Analyzer")
        print(f"Variable: {self.variable}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Regions: {self.regions}")
        print(f"Percentile clipping: {self.clip_percentiles}")
    
    def load_region_data(self, region: str) -> xr.Dataset:
        """Load variable data for a specific region.
        
        Args:
            region: Region name
            
        Returns:
            Dataset containing variable data
        """
        file_path = self.data_dir / f"{region}.nc"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        print(f"Loading {region} data from: {file_path}")
        ds = xr.open_dataset(file_path)
        
        # Check if variable exists
        if self.variable not in ds.data_vars:
            available_vars = list(ds.data_vars.keys())
            raise ValueError(f"Variable '{self.variable}' not found in {region} data. Available: {available_vars}")
        
        var_data = ds[self.variable]
        print(f"  Shape: {var_data.shape}")
        print(f"  Time range: {ds.time.min().values} to {ds.time.max().values}")
        print(f"  {self.variable} range: {var_data.min().values:.2f} to {var_data.max().values:.2f}")
        
        return ds
    
    def extract_variable_values(self, ds: xr.Dataset) -> np.ndarray:
        """Extract variable values as flat array.
        
        Args:
            ds: Dataset containing variable data
            
        Returns:
            Flattened array of variable values
        """
        # Convert to numpy array and flatten
        var_values = ds[self.variable].values.flatten()
        
        # Remove any NaN values
        var_values = var_values[~np.isnan(var_values)]
        
        # Apply percentile clipping if requested
        if self.clip_percentiles:
            p1 = np.percentile(var_values, 1)
            p99 = np.percentile(var_values, 99)
            print(f"  Clipping percentiles: {p1:.2f} to {p99:.2f}")
            var_values = np.clip(var_values, p1, p99)
        
        return var_values
    
    def normalize_variable(self, var_values: np.ndarray) -> np.ndarray:
        """Normalize variable values to [0, 1] range.
        
        Args:
            var_values: Raw variable values
            
        Returns:
            Normalized variable values
        """
        var_min = var_values.min()
        var_max = var_values.max()
        
        normalized = (var_values - var_min) / (var_max - var_min)
        
        return normalized, var_min, var_max
    
    def calculate_statistics(self, var_values: np.ndarray) -> Dict:
        """Calculate statistical measures for variable data.
        
        Args:
            var_values: Variable values
            
        Returns:
            Dictionary of statistical measures
        """
        stats_dict = {
            'count': len(var_values),
            'mean': np.mean(var_values),
            'std': np.std(var_values),
            'min': np.min(var_values),
            'max': np.max(var_values),
            'median': np.median(var_values),
            'q25': np.percentile(var_values, 25),
            'q75': np.percentile(var_values, 75),
            'skewness': stats.skew(var_values),
            'kurtosis': stats.kurtosis(var_values)
        }
        
        return stats_dict
    
    def create_histogram_plot(self, region_data: Dict[str, np.ndarray], 
                            normalized_data: Dict[str, np.ndarray],
                            var_ranges: Dict[str, Tuple[float, float]]) -> None:
        """Create histogram plots for variable distributions.
        
        Args:
            region_data: Raw variable data for each region
            normalized_data: Normalized variable data for each region
            var_ranges: Variable ranges (min, max) for each region
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.variable.upper()} Distribution Analysis Across Regions', fontsize=16, fontweight='bold')
        
        # Plot 1: Raw variable histograms
        ax1 = axes[0, 0]
        for region in self.regions:
            if region in region_data:
                ax1.hist(region_data[region], bins=50, alpha=0.7, 
                        label=f'{region}', density=True)
        ax1.set_xlabel(f'{self.variable.upper()}')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Raw {self.variable.upper()} Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Normalized variable histograms
        ax2 = axes[0, 1]
        for region in self.regions:
            if region in normalized_data:
                ax2.hist(normalized_data[region], bins=50, alpha=0.7, 
                        label=f'{region} (norm)', density=True)
        ax2.set_xlabel(f'Normalized {self.variable.upper()} [0, 1]')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Normalized {self.variable.upper()} Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Box plots for raw variables
        ax3 = axes[1, 0]
        box_data = [region_data[region] for region in self.regions if region in region_data]
        box_labels = [region for region in self.regions if region in region_data]
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_ylabel(f'{self.variable.upper()}')
        ax3.set_title(f'{self.variable.upper()} Distribution Box Plots')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overlayed normalized distributions
        ax4 = axes[1, 1]
        for region in self.regions:
            if region in normalized_data:
                ax4.hist(normalized_data[region], bins=50, alpha=0.6, 
                        label=f'{region}', density=True, histtype='step', linewidth=2)
        ax4.set_xlabel(f'Normalized {self.variable.upper()} [0, 1]')
        ax4.set_ylabel('Density')
        ax4.set_title('Overlayed Normalized Distributions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / f'{self.variable}_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Histogram plot saved: {output_path}")
        
        # plt.show()
    
    def create_statistics_table(self, statistics: Dict[str, Dict]) -> None:
        """Create and save statistics table.
        
        Args:
            statistics: Statistics for each region
        """
        # Create DataFrame
        df = pd.DataFrame(statistics).T
        
        # Round numerical values
        numeric_cols = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'skewness', 'kurtosis']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(3)
        
        # Save to CSV
        csv_path = self.output_dir / f'{self.variable}_statistics.csv'
        df.to_csv(csv_path)
        print(f"✓ Statistics table saved: {csv_path}")
        
        # Print table
        print("\n" + "="*80)
        print(f"{self.variable.upper()} STATISTICS BY REGION")
        print("="*80)
        print(df.to_string())
        print("="*80)
    
    def create_comparison_plot(self, normalized_data: Dict[str, np.ndarray]) -> None:
        """Create detailed comparison plot.
        
        Args:
            normalized_data: Normalized variable data for each region
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Detailed {self.variable.upper()} Distribution Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: KDE plots
        ax1 = axes[0]
        for region in self.regions:
            if region in normalized_data:
                sns.kdeplot(normalized_data[region], label=region, ax=ax1)
        ax1.set_xlabel(f'Normalized {self.variable.upper()} [0, 1]')
        ax1.set_ylabel('Density')
        ax1.set_title('Kernel Density Estimation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Violin plots
        ax2 = axes[1]
        violin_data = []
        violin_labels = []
        for region in self.regions:
            if region in normalized_data:
                violin_data.append(normalized_data[region])
                violin_labels.append(region)
        
        parts = ax2.violinplot(violin_data, positions=range(len(violin_labels)))
        ax2.set_xticks(range(len(violin_labels)))
        ax2.set_xticklabels(violin_labels)
        ax2.set_ylabel(f'Normalized {self.variable.upper()} [0, 1]')
        ax2.set_title('Distribution Shape Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative distribution
        ax3 = axes[2]
        for region in self.regions:
            if region in normalized_data:
                sorted_data = np.sort(normalized_data[region])
                y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax3.plot(sorted_data, y, label=region, linewidth=2)
        ax3.set_xlabel(f'Normalized {self.variable.upper()} [0, 1]')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Functions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / f'{self.variable}_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved: {output_path}")
        
        plt.show()
    
    def run_analysis(self) -> None:
        """Run the complete variable analysis."""
        print("\n" + "="*60)
        print(f"Starting {self.variable.upper()} Distribution Analysis")
        print("="*60)
        
        # Load data for all regions
        region_data = {}
        normalized_data = {}
        var_ranges = {}
        statistics = {}
        
        for region in self.regions:
            print(f"\nProcessing {region}...")
            try:
                # Load data
                ds = self.load_region_data(region)
                
                # Extract variable values
                var_values = self.extract_variable_values(ds)
                region_data[region] = var_values
                
                # Normalize variable
                norm_values, var_min, var_max = self.normalize_variable(var_values)
                normalized_data[region] = norm_values
                var_ranges[region] = (var_min, var_max)
                
                # Calculate statistics
                stats_dict = self.calculate_statistics(var_values)
                statistics[region] = stats_dict
                
                print(f"  ✓ Loaded {len(var_values):,} {self.variable} values")
                print(f"  ✓ Range: {var_min:.2f} to {var_max:.2f}")
                print(f"  ✓ Mean: {stats_dict['mean']:.2f}, Std: {stats_dict['std']:.2f}")
                
                # Close dataset
                ds.close()
                
            except Exception as e:
                print(f"  ✗ Error processing {region}: {e}")
                continue
        
        if not region_data:
            print("\n✗ No region data loaded successfully!")
            return
        
        print(f"\n✓ Successfully loaded data for {len(region_data)} regions")
        
        # Create plots
        print("\nCreating visualization plots...")
        self.create_histogram_plot(region_data, normalized_data, var_ranges)
        self.create_comparison_plot(normalized_data)
        
        # Create statistics table
        print("\nGenerating statistics...")
        self.create_statistics_table(statistics)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        for region in self.regions:
            if region in statistics:
                stats = statistics[region]
                print(f"\n{region}:")
                print(f"  {self.variable.upper()} range: {stats['min']:.2f} to {stats['max']:.2f}")
                print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                print(f"  Skewness: {stats['skewness']:.3f}, Kurtosis: {stats['kurtosis']:.3f}")
                print(f"  Data points: {stats['count']:,}")
        
        print(f"\n✓ Analysis complete! Results saved to: {self.output_dir}")
        print("✓ Generated files:")
        print(f"  - {self.variable}_distributions.png")
        print(f"  - {self.variable}_comparison.png") 
        print(f"  - {self.variable}_statistics.csv")


def main():
    """Main function to handle command line arguments and execute analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze variable distributions across regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze temperature distributions (default)
  python analyze_temperature_distributions.py --data-dir processed_data/ERA5/test
  
  # Analyze wind speed (u10) with percentile clipping
  python analyze_temperature_distributions.py --data-dir processed_data/ERA5/test --variable u10
  
  # Analyze without percentile clipping
  python analyze_temperature_distributions.py --data-dir processed_data/ERA5/test --variable v10 --no-clip
  
  # Specify custom output directory
  python analyze_temperature_distributions.py --data-dir processed_data/ERA5/test --output-dir results --variable t2m
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing region NetCDF files (e.g., processed_data/ERA5/test)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save analysis results (default: DATA_DIR/analysis)"
    )
    
    parser.add_argument(
        "--variable",
        type=str,
        default="t2m",
        help="Variable to analyze (e.g., 't2m', 'u10', 'v10', 'sshf', 'zust', 'sp')"
    )
    
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable percentile clipping (1st and 99th percentiles)"
    )
    
    args = parser.parse_args()
    
    # Set paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        sys.exit(1)
    
    if args.output_dir is None:
        output_dir = data_dir / "analysis"
    else:
        output_dir = Path(args.output_dir)
    
    # Run analysis
    try:
        analyzer = VariableAnalyzer(
            data_dir=data_dir, 
            output_dir=output_dir,
            variable=args.variable,
            clip_percentiles=not args.no_clip
        )
        analyzer.run_analysis()
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
