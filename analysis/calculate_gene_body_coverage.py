#!/usr/bin/env python3
"""
Calculate mean read coverage along gene bodies in three bins (5' quartile, middle 50%, 3' quartile)
for Mammiellales species, comparing mating type (MT) vs autosomal genes.
"""
import argparse
import glob
import os
import re
import subprocess
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm

# Species mapping: (short name, display name)
SPECIES_NAMES = {
    "Mpusilla": "M. pusilla",
    "Mcommoda": "M. commoda",
    "Otauri": "O. tauri",
    "Bprasinos": "B. prasinos",
}


def parse_gtf_attributes(attr_string):
    """Parse GTF attribute string into dictionary."""
    attributes = {}
    parts = re.findall(r'(\w+)\s+"([^"]+)"', attr_string)
    for key, value in parts:
        attributes[key] = value
    return attributes


def parse_gtf_genes(gtf_file):
    """
    Parse GTF file to extract gene features.
    Returns dict: gene_id -> {chrom, start, end, strand}
    """
    genes = {}
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature == "gene":
                chrom = fields[0]
                start = int(fields[3])  # GTF is 1-based inclusive
                end = int(fields[4])   # GTF is 1-based inclusive
                strand = fields[6]
                attrs = parse_gtf_attributes(fields[8])
                if 'gene_id' in attrs:
                    gene_id = attrs['gene_id']
                    genes[gene_id] = {
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'strand': strand
                    }
    return genes


def load_bam_depths(bam_files, verbose=False):
    """
    Load BAM file(s) and compute genome-wide coverage using samtools depth.
    Returns dict: chrom -> numpy array of coverage values (0-based indexing)
    If multiple BAM files are provided, coverage is summed across files.
    """
    if isinstance(bam_files, str):
        bam_files = [bam_files]
    
    combined_arrays = {}
    
    for bam_file in bam_files:
        if verbose:
            print(f"[INFO] Processing BAM {bam_file} with samtools depth...", flush=True)
        cmd = ["samtools", "depth", "-aa", bam_file]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        current_chrom = None
        depths = []
        chrom_lengths = {}
        
        for line in tqdm(proc.stdout, desc=f"Reading depth from {os.path.basename(bam_file)}", unit="lines", disable=not verbose):
            chrom, pos, cov = line.strip().split('\t')
            pos, cov = int(pos) - 1, int(cov)  # Convert to 0-based
            if current_chrom != chrom:
                if current_chrom is not None:
                    # Store previous chromosome
                    arr = np.array(depths, dtype=int)
                    chrom_lengths[current_chrom] = len(arr)
                    if current_chrom in combined_arrays:
                        # Extend if needed and sum
                        if len(combined_arrays[current_chrom]) < len(arr):
                            # Pad existing array
                            padded = np.zeros(len(arr), dtype=int)
                            padded[:len(combined_arrays[current_chrom])] = combined_arrays[current_chrom]
                            combined_arrays[current_chrom] = padded
                        combined_arrays[current_chrom] += arr
                    else:
                        combined_arrays[current_chrom] = arr
                current_chrom = chrom
                depths = []
            depths.append(cov)
        
        # Handle last chromosome
        if current_chrom is not None:
            arr = np.array(depths, dtype=int)
            if current_chrom in combined_arrays:
                # Extend if needed and sum
                if len(combined_arrays[current_chrom]) < len(arr):
                    padded = np.zeros(len(arr), dtype=int)
                    padded[:len(combined_arrays[current_chrom])] = combined_arrays[current_chrom]
                    combined_arrays[current_chrom] = padded
                combined_arrays[current_chrom] += arr
            else:
                combined_arrays[current_chrom] = arr
    
    return combined_arrays


def calculate_gene_coverage_bins(chrom, gene_start, gene_end, strand, depth_array):
    """
    Calculate mean coverage in three bins along gene body.
    
    Bins:
    - 5' quartile: positions 0-25% of gene length
    - Middle 50%: positions 25-75% of gene length
    - 3' quartile: positions 75-100% of gene length
    
    For - strand genes, 5' is at the end, so we reverse the order.
    
    Args:
        chrom: chromosome name
        gene_start: gene start position (1-based inclusive, from GTF)
        gene_end: gene end position (1-based inclusive, from GTF)
        strand: '+' or '-'
        depth_array: dict of chrom -> numpy array (0-based indexing)
    
    Returns: (coverage_5p, coverage_middle, coverage_3p)
    """
    if chrom not in depth_array:
        return np.nan, np.nan, np.nan
    
    # Convert 1-based GTF coordinates to 0-based array indices
    # GTF: 1-based inclusive [gene_start, gene_end]
    # samtools depth outputs 1-based positions, stored in 0-based array
    # Position N (1-based) is at index N-1 (0-based)
    # To get positions gene_start to gene_end (inclusive), use [gene_start-1:gene_end]
    chrom_len = len(depth_array[chrom])
    start_idx = max(0, min(gene_start - 1, chrom_len - 1))  # Convert to 0-based
    end_idx = max(start_idx + 1, min(gene_end, chrom_len))  # End is exclusive in slice, so gene_end gives positions up to gene_end-1
    
    if start_idx >= end_idx:
        return np.nan, np.nan, np.nan
    
    # Extract gene body coverage
    gene_coverage = depth_array[chrom][start_idx:end_idx].astype(float)
    gene_length = len(gene_coverage)
    
    if gene_length < 4:
        # Too short to divide into bins
        mean_cov = np.mean(gene_coverage)
        return mean_cov, mean_cov, mean_cov
    
    # Calculate bin boundaries
    q1_end = int(gene_length * 0.25)
    q3_start = int(gene_length * 0.75)
    
    if strand == '-':
        # For - strand: 5' is at end, 3' is at start
        # Reverse the array to get 5' -> 3' orientation
        gene_coverage = gene_coverage[::-1]
    
    # Calculate mean coverage for each bin
    coverage_5p = np.mean(gene_coverage[:q1_end]) if q1_end > 0 else np.mean(gene_coverage)
    coverage_middle = np.mean(gene_coverage[q1_end:q3_start]) if q3_start > q1_end else np.mean(gene_coverage)
    coverage_3p = np.mean(gene_coverage[q3_start:]) if q3_start < gene_length else np.mean(gene_coverage)
    
    return coverage_5p, coverage_middle, coverage_3p


def process_species(branchpoints_file, gtf_file, bam_files, output_prefix, verbose=False):
    """
    Process a single species: calculate coverage bins for all genes.
    
    Returns:
    - per_gene_df: DataFrame with per-gene coverage results
    - summary_stats: dict with summary statistics
    """
    if verbose:
        print(f"\n[INFO] Processing species: {output_prefix}")
        print(f"  Branchpoints file: {branchpoints_file}")
        print(f"  GTF file: {gtf_file}")
        print(f"  BAM files: {bam_files}")
    
    # Load branchpoints.tsv
    bp_df = pd.read_csv(branchpoints_file, sep='\t')
    if verbose:
        print(f"  Loaded {len(bp_df)} intron records")
    
    # Get unique gene_ids with In_region flag
    gene_info = bp_df.groupby('gene_id').agg({
        'In_region': 'first',  # Should be same for all introns of a gene
        'Gene_length': 'first'
    }).reset_index()
    
    if verbose:
        print(f"  Found {len(gene_info)} unique genes")
        print(f"    Autosomal (In_region=0): {len(gene_info[gene_info['In_region']==0])}")
        print(f"    Mating-type (In_region=1): {len(gene_info[gene_info['In_region']==1])}")
    
    # Parse GTF to get gene coordinates
    genes_gtf = parse_gtf_genes(gtf_file)
    if verbose:
        print(f"  Found {len(genes_gtf)} genes in GTF")
    
    # Load BAM depths
    depth_arrays = load_bam_depths(bam_files, verbose=verbose)
    if verbose:
        print(f"  Loaded coverage for {len(depth_arrays)} chromosomes")
    
    # Calculate coverage bins for each gene
    results = []
    genes_not_found = 0
    genes_no_coverage = 0
    
    for _, row in tqdm(gene_info.iterrows(), total=len(gene_info), desc="Calculating coverage", disable=not verbose):
        gene_id = row['gene_id']
        in_region = row['In_region']
        gene_length = row['Gene_length']
        
        if gene_id not in genes_gtf:
            genes_not_found += 1
            continue
        
        gene_coords = genes_gtf[gene_id]
        chrom = gene_coords['chrom']
        start = gene_coords['start']
        end = gene_coords['end']
        strand = gene_coords['strand']
        
        # Calculate coverage bins
        cov_5p, cov_middle, cov_3p = calculate_gene_coverage_bins(
            chrom, start, end, strand, depth_arrays
        )
        
        if np.isnan(cov_5p):
            genes_no_coverage += 1
            continue
        
        results.append({
            'gene_id': gene_id,
            'chrom': chrom,
            'start': start,  # Already 1-based from GTF
            'end': end,      # Already 1-based from GTF
            'strand': strand,
            'In_region': in_region,
            'coverage_5p': cov_5p,
            'coverage_middle': cov_middle,
            'coverage_3p': cov_3p,
            'Gene_length': gene_length
        })
    
    if genes_not_found > 0 and verbose:
        print(f"  Warning: {genes_not_found} genes not found in GTF")
    if genes_no_coverage > 0 and verbose:
        print(f"  Warning: {genes_no_coverage} genes had no coverage data")
    
    per_gene_df = pd.DataFrame(results)
    
    if len(per_gene_df) == 0:
        if verbose:
            print("  ERROR: No genes with valid coverage data!")
        return None, None
    
    # Calculate summary statistics
    auto_df = per_gene_df[per_gene_df['In_region'] == 0]
    mt_df = per_gene_df[per_gene_df['In_region'] == 1]
    
    summary_stats = {
        'species': output_prefix,
        'n_autosomal_genes': len(auto_df),
        'n_mt_genes': len(mt_df),
        'auto_mean_5p': auto_df['coverage_5p'].mean() if len(auto_df) > 0 else np.nan,
        'auto_mean_middle': auto_df['coverage_middle'].mean() if len(auto_df) > 0 else np.nan,
        'auto_mean_3p': auto_df['coverage_3p'].mean() if len(auto_df) > 0 else np.nan,
        'mt_mean_5p': mt_df['coverage_5p'].mean() if len(mt_df) > 0 else np.nan,
        'mt_mean_middle': mt_df['coverage_middle'].mean() if len(mt_df) > 0 else np.nan,
        'mt_mean_3p': mt_df['coverage_3p'].mean() if len(mt_df) > 0 else np.nan,
    }
    
    # Statistical tests
    if len(auto_df) > 0 and len(mt_df) > 0:
        for bin_name, auto_col, mt_col in [
            ('5p', 'coverage_5p', 'coverage_5p'),
            ('middle', 'coverage_middle', 'coverage_middle'),
            ('3p', 'coverage_3p', 'coverage_3p')
        ]:
            auto_vals = auto_df[auto_col].dropna()
            mt_vals = mt_df[mt_col].dropna()
            if len(auto_vals) > 1 and len(mt_vals) > 1:
                t_stat, p_val = ttest_ind(auto_vals, mt_vals)
                summary_stats[f'ttest_{bin_name}_stat'] = t_stat
                summary_stats[f'ttest_{bin_name}_pval'] = p_val
    
    return per_gene_df, summary_stats


def plot_coverage_comparison(all_species_data, output_path):
    """
    Create multi-panel figure with subfigures for each species.
    Each subfigure shows coverage in three bins (5' quartile, middle 50%, 3' quartile)
    comparing autosomal vs mating-type genes.
    """
    n_species = len(all_species_data)
    if n_species == 0:
        print("ERROR: No species data to plot")
        return
    
    # Determine grid layout
    n_cols = 2
    n_rows = int(np.ceil(n_species / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    if n_species == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (species_name, per_gene_df) in enumerate(all_species_data):
        ax = axes[idx]
        
        # Prepare data for plotting
        auto_df = per_gene_df[per_gene_df['In_region'] == 0]
        mt_df = per_gene_df[per_gene_df['In_region'] == 1]
        
        # Calculate means and standard errors
        bins = ['5\' quartile', 'Middle 50%', '3\' quartile']
        auto_means = [
            auto_df['coverage_5p'].mean() if len(auto_df) > 0 else 0,
            auto_df['coverage_middle'].mean() if len(auto_df) > 0 else 0,
            auto_df['coverage_3p'].mean() if len(auto_df) > 0 else 0
        ]
        auto_sems = [
            auto_df['coverage_5p'].sem() if len(auto_df) > 1 else 0,
            auto_df['coverage_middle'].sem() if len(auto_df) > 1 else 0,
            auto_df['coverage_3p'].sem() if len(auto_df) > 1 else 0
        ]
        mt_means = [
            mt_df['coverage_5p'].mean() if len(mt_df) > 0 else 0,
            mt_df['coverage_middle'].mean() if len(mt_df) > 0 else 0,
            mt_df['coverage_3p'].mean() if len(mt_df) > 0 else 0
        ]
        mt_sems = [
            mt_df['coverage_5p'].sem() if len(mt_df) > 1 else 0,
            mt_df['coverage_middle'].sem() if len(mt_df) > 1 else 0,
            mt_df['coverage_3p'].sem() if len(mt_df) > 1 else 0
        ]
        
        # Create grouped bar plot
        x = np.arange(len(bins))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, auto_means, width, yerr=auto_sems, 
                       label='Autosomal', color='#4477AA', alpha=0.8, capsize=5)
        bars2 = ax.bar(x + width/2, mt_means, width, yerr=mt_sems,
                       label='Mating-type', color='#FF8C00', alpha=0.8, capsize=5)
        
        ax.set_xlabel('Gene body position', fontsize=10)
        ax.set_ylabel('Mean coverage', fontsize=10)
        ax.set_title(SPECIES_NAMES.get(species_name, species_name), fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(bins, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(n_species, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def find_species_files(data_dir, bam_base_dir):
    """Find all branchpoints.tsv files and corresponding GTF/BAM files."""
    branchpoints_files = glob.glob(os.path.join(data_dir, "*.branchpoints.tsv"))
    species_configs = []
    
    for bp_file in branchpoints_files:
        # Extract species name from filename (e.g., "Mpusilla.features.branchpoints.tsv" -> "Mpusilla")
        basename = os.path.basename(bp_file)
        species_match = re.match(r'([^.]+)\.features\.branchpoints\.tsv', basename)
        if not species_match:
            continue
        species_short = species_match.group(1)
        
        # Find GTF file
        gtf_pattern = os.path.join(bam_base_dir, species_short, "*.gtf")
        gtf_files = glob.glob(gtf_pattern)
        if not gtf_files:
            print(f"Warning: No GTF file found for {species_short}")
            continue
        
        # Find BAM files
        bam_pattern = os.path.join(bam_base_dir, species_short, "*.bam")
        bam_files = glob.glob(bam_pattern)
        if not bam_files:
            print(f"Warning: No BAM files found for {species_short}")
            continue
        
        species_configs.append({
            'species': species_short,
            'branchpoints_file': bp_file,
            'gtf_file': gtf_files[0],  # Use first GTF found
            'bam_files': bam_files
        })
    
    return species_configs


def main():
    parser = argparse.ArgumentParser(
        description="Calculate gene body coverage in bins for Mammiellales species"
    )
    
    # Single species mode
    parser.add_argument('--branchpoints_file', type=str, help='Path to branchpoints.tsv file')
    parser.add_argument('--gtf_file', type=str, help='Path to GTF file')
    parser.add_argument('--bam_files', nargs='+', help='One or more BAM files')
    parser.add_argument('--output_prefix', type=str, help='Output file prefix')
    
    # Multi-species mode
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing *.branchpoints.tsv files (default: data)')
    parser.add_argument('--bam_base_dir', type=str,
                       default='/scratch2/russ/introner/splicing_fails/Mammialles',
                       help='Base directory for BAM files')
    parser.add_argument('--output_dir', type=str, default='analysis',
                       help='Output directory (default: analysis)')
    
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.branchpoints_file and args.gtf_file and args.bam_files:
        # Single species mode
        if not args.output_prefix:
            # Extract species name from branchpoints file
            basename = os.path.basename(args.branchpoints_file)
            match = re.match(r'([^.]+)\.features\.branchpoints\.tsv', basename)
            args.output_prefix = match.group(1) if match else 'species'
        
        per_gene_df, summary_stats = process_species(
            args.branchpoints_file, args.gtf_file, args.bam_files,
            args.output_prefix, verbose=args.verbose
        )
        
        if per_gene_df is not None:
            output_prefix_full = os.path.join(args.output_dir, args.output_prefix)
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Write outputs
            per_gene_df.to_csv(f"{output_prefix_full}_gene_coverage.tsv", sep='\t', index=False)
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_csv(f"{output_prefix_full}_coverage_summary.tsv", sep='\t', index=False)
            
            print(f"\nPer-gene results saved to: {output_prefix_full}_gene_coverage.tsv")
            print(f"Summary statistics saved to: {output_prefix_full}_coverage_summary.tsv")
    
    else:
        # Multi-species mode
        species_configs = find_species_files(args.data_dir, args.bam_base_dir)
        
        if not species_configs:
            print("ERROR: No species configurations found!")
            return
        
        print(f"Found {len(species_configs)} species to process")
        
        all_per_gene_data = []
        all_summary_stats = []
        
        for config in species_configs:
            species = config['species']
            per_gene_df, summary_stats = process_species(
                config['branchpoints_file'],
                config['gtf_file'],
                config['bam_files'],
                species,
                verbose=args.verbose
            )
            
            if per_gene_df is not None:
                output_prefix = os.path.join(args.output_dir, species)
                os.makedirs(args.output_dir, exist_ok=True)
                
                # Write per-species outputs
                per_gene_df.to_csv(f"{output_prefix}_gene_coverage.tsv", sep='\t', index=False)
                summary_df = pd.DataFrame([summary_stats])
                summary_df.to_csv(f"{output_prefix}_coverage_summary.tsv", sep='\t', index=False)
                
                all_per_gene_data.append((species, per_gene_df))
                all_summary_stats.append(summary_stats)
        
        # Create combined plot
        if all_per_gene_data:
            plot_path = os.path.join(args.output_dir, "gene_body_coverage_comparison.pdf")
            plot_coverage_comparison(all_per_gene_data, plot_path)
            
            # Write combined summary
            combined_summary_df = pd.DataFrame(all_summary_stats)
            combined_summary_df.to_csv(
                os.path.join(args.output_dir, "all_species_coverage_summary.tsv"),
                sep='\t', index=False
            )
            print(f"\nCombined summary saved to: {os.path.join(args.output_dir, 'all_species_coverage_summary.tsv')}")


if __name__ == "__main__":
    main()
