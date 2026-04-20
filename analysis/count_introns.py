#!/usr/bin/env python3
"""
Count valid introns (Average_gene_depth > 10) and introns with retention ratio > 0.0
for each species, broken down by mating_type and autosomal regions.
"""
import argparse
import pandas as pd
import glob
import os


def extract_species_name(filepath):
    """Extract species name from filename (e.g., 'Mpusilla' from 'Mpusilla.features.branchpoints.tsv')."""
    basename = os.path.basename(filepath)
    return basename.replace('.features.branchpoints.tsv', '')


def process_species_file(filepath, min_depth=10):
    """
    Process a single species TSV file and compute statistics.
    
    Parameters:
    -----------
    filepath : str
        Path to the TSV file
    min_depth : float
        Minimum average gene depth threshold (default: 10)
    
    Returns:
    --------
    dict : Dictionary with species name and statistics
    """
    species = extract_species_name(filepath)
    
    # Load data
    df = pd.read_csv(filepath, sep="\t")
    
    # Filter for valid expressed introns (Average_gene_depth > min_depth)
    # Handle NaN values by excluding them (NaN > min_depth is False)
    valid_df = df[df["Average_gene_depth"] > min_depth].copy()
    
    if len(valid_df) == 0:
        return {
            "Species": species,
            "Mating_type_valid": 0,
            "Autosomal_valid": 0,
            "Mating_type_retained": 0,
            "Autosomal_retained": 0,
            "MT_retained_proportion": 0.0,
            "Auto_retained_proportion": 0.0,
        }
    
    # Count valid introns by region
    mating_type_valid = len(valid_df[valid_df["In_region"] == 1])
    autosomal_valid = len(valid_df[valid_df["In_region"] == 0])
    
    # Count introns with retention ratio > 0.0 by region
    mating_type_retained = len(valid_df[(valid_df["In_region"] == 1) & 
                                         (valid_df["rMATS_retention_ratio"] > 0.0)])
    autosomal_retained = len(valid_df[(valid_df["In_region"] == 0) & 
                                      (valid_df["rMATS_retention_ratio"] > 0.0)])
    
    # Proportion retained = retained / valid (0.0 when valid is 0)
    mt_prop = mating_type_retained / mating_type_valid if mating_type_valid else 0.0
    auto_prop = autosomal_retained / autosomal_valid if autosomal_valid else 0.0
    
    return {
        "Species": species,
        "Mating_type_valid": mating_type_valid,
        "Autosomal_valid": autosomal_valid,
        "Mating_type_retained": mating_type_retained,
        "Autosomal_retained": autosomal_retained,
        "MT_retained_proportion": round(mt_prop, 4),
        "Auto_retained_proportion": round(auto_prop, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Count valid introns and retained introns by species and region type"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing *.features.branchpoints.tsv files (default: data)"
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=10.0,
        help="Minimum average gene depth threshold (default: 10.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output TSV file path. If not specified, prints to stdout only."
    )
    args = parser.parse_args()
    
    # Find all TSV files
    pattern = os.path.join(args.data_dir, "*.features.branchpoints.tsv")
    tsv_files = sorted(glob.glob(pattern))
    
    if not tsv_files:
        print(f"Error: No *.features.branchpoints.tsv files found in {args.data_dir}")
        return 1
    
    print(f"Found {len(tsv_files)} species files")
    print(f"Using minimum depth threshold: {args.min_depth}")
    print()
    
    # Process each file
    results = []
    for filepath in tsv_files:
        stats = process_species_file(filepath, min_depth=args.min_depth)
        results.append(stats)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print formatted table
    print("=" * 100)
    print(f"{'Species':<14} {'MT Valid':<10} {'Auto Valid':<10} {'MT Retained':<12} {'Auto Retained':<12} {'MT Prop':<10} {'Auto Prop':<10}")
    print("=" * 100)
    for _, row in results_df.iterrows():
        print(f"{row['Species']:<14} {row['Mating_type_valid']:<10} {row['Autosomal_valid']:<10} "
              f"{row['Mating_type_retained']:<12} {row['Autosomal_retained']:<12} "
              f"{row['MT_retained_proportion']:<10.4f} {row['Auto_retained_proportion']:<10.4f}")
    print("=" * 100)
    print()
    
    # Print summary statistics
    print("Summary:")
    print(f"  Total species: {len(results_df)}")
    print(f"  Total mating_type valid introns: {results_df['Mating_type_valid'].sum()}")
    print(f"  Total autosomal valid introns: {results_df['Autosomal_valid'].sum()}")
    print(f"  Total mating_type retained introns: {results_df['Mating_type_retained'].sum()}")
    print(f"  Total autosomal retained introns: {results_df['Autosomal_retained'].sum()}")
    
    # Save to TSV if output path specified
    if args.output:
        results_df.to_csv(args.output, sep="\t", index=False)
        print(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
