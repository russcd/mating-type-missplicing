#!/usr/bin/env python3
"""
Analyze gene density and intron abundance comparing MT vs autosomal regions.
Usage:
    python gene_density_intron_abundance.py [--output-dir analysis]
"""

import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.stats import mannwhitneyu


# Species configuration: short_name, display_name, GTF path, FASTA path, data path
SPECIES_CONFIG = {
    "Mpusilla": {
        "display_name": "M. pusilla",
        "gtf": "/scratch2/russ/introner/splicing_fails/Mammialles/Mpusilla/GCF_000151265.4_Micromonas_pusilla_CCMP1545_v2.0_genomic.gtf",
        "fasta": "/scratch2/russ/introner/splicing_fails/Mammialles/Mpusilla/GCF_000151265.4_Micromonas_pusilla_CCMP1545_v2.0_genomic.fna",
        "data": "data/Mpusilla.features.branchpoints.tsv",
        "mt_chroms": ["NW_003315883.1"],  # Partial scaffold
    },
    "Mcommoda": {
        "display_name": "M. commoda",
        "gtf": "/scratch2/russ/introner/splicing_fails/Mammialles/Mcommoda/GCF_000090985.2_ASM9098v2_genomic.gtf",
        "fasta": "/scratch2/russ/introner/splicing_fails/Mammialles/Mcommoda/GCF_000090985.2_ASM9098v2_genomic.fna",
        "data": "data/Mcommoda.features.branchpoints.tsv",
        "mt_chroms": ["NC_013038.1"],  # Entire chromosome
    },
    "Bprasinos": {
        "display_name": "B. prasinos",
        "gtf": "/scratch2/russ/introner/splicing_fails/Mammialles/Bprasinos/GCF_002220235.1_ASM222023v1_genomic.gtf",
        "fasta": "/scratch2/russ/introner/splicing_fails/Mammialles/Bprasinos/GCF_002220235.1_ASM222023v1_genomic.fna",
        "data": "data/Bprasinos.features.branchpoints.tsv",
        "mt_chroms": ["NC_023995.1"],  # Entire chromosome
    },
    "Otauri": {
        "display_name": "O. tauri",
        "gtf": "/scratch2/russ/introner/splicing_fails/Mammialles/Otauri/GCF_000214015.3_version_140606_genomic.gtf",
        "fasta": "/scratch2/russ/introner/splicing_fails/Mammialles/Otauri/GCF_000214015.3_version_140606_genomic.fna",
        "data": "data/Otauri.features.branchpoints.tsv",
        "mt_chroms": ["NC_014427.2"],  # Entire chromosome
    },
}


def parse_gtf_attributes(attr_string):
    """Parse GTF attribute string into a dictionary."""
    attributes = {}
    parts = re.findall(r'(\w+)\s+"([^"]+)"', attr_string)
    for key, value in parts:
        attributes[key] = value
    return attributes


def get_chromosome_lengths(fasta_file):
    """
    Parse FASTA file to get chromosome/scaffold lengths.

    Returns:
        dict: {chromosome_id: length_in_bp}
    """
    chrom_lengths = {}
    with open(fasta_file, 'r') as f:
        for record in SeqIO.parse(f, "fasta"):
            chrom_lengths[record.id] = len(record.seq)
    return chrom_lengths


def load_intron_data(tsv_file):
    """
    Load intron-level data from .features.branchpoints.tsv file.

    Returns:
        DataFrame with columns including gene_id and In_region
    """
    df = pd.read_csv(tsv_file, sep="\t")
    return df


def get_mt_gene_set(intron_df):
    """
    Extract set of gene IDs that are in MT regions (In_region == 1).

    Args:
        intron_df: DataFrame from load_intron_data()

    Returns:
        set: gene IDs in MT regions
    """
    mt_genes = intron_df[intron_df["In_region"] == 1]["gene_id"].unique()
    return set(mt_genes)


def extract_genes_from_gtf(gtf_file):
    """
    Parse GTF file to extract gene information.

    Returns:
        DataFrame with columns: gene_id, chromosome, start, end, strand
    """
    genes = []
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != "gene":
                continue

            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attrs = parse_gtf_attributes(fields[8])

            if 'gene_id' in attrs:
                genes.append({
                    'gene_id': attrs['gene_id'],
                    'chromosome': chrom,
                    'start': start,
                    'end': end,
                    'strand': strand,
                    'length': end - start + 1,
                })

    return pd.DataFrame(genes)


def calculate_intron_abundance(intron_df, mt_genes):
    """
    Calculate introns per gene for MT and autosomal genes.

    Args:
        intron_df: DataFrame from load_intron_data()
        mt_genes: set of MT gene IDs

    Returns:
        dict with statistics for MT and autosomal genes
    """
    # Count introns per gene
    gene_intron_counts = intron_df.groupby("gene_id").size().reset_index(name='n_introns')

    # Classify genes as MT or autosomal
    gene_intron_counts['region'] = gene_intron_counts['gene_id'].apply(
        lambda x: 'MT' if x in mt_genes else 'Autosomal'
    )

    # Split by region
    mt_counts = gene_intron_counts[gene_intron_counts['region'] == 'MT']['n_introns'].values
    auto_counts = gene_intron_counts[gene_intron_counts['region'] == 'Autosomal']['n_introns'].values

    # Statistical test
    if len(mt_counts) > 0 and len(auto_counts) > 0:
        stat, pval = mannwhitneyu(mt_counts, auto_counts, alternative='two-sided')
    else:
        stat, pval = np.nan, np.nan

    results = {
        'MT': {
            'n_genes': len(mt_counts),
            'mean': np.mean(mt_counts) if len(mt_counts) > 0 else np.nan,
            'median': np.median(mt_counts) if len(mt_counts) > 0 else np.nan,
            'std': np.std(mt_counts) if len(mt_counts) > 0 else np.nan,
            'values': mt_counts,
        },
        'Autosomal': {
            'n_genes': len(auto_counts),
            'mean': np.mean(auto_counts) if len(auto_counts) > 0 else np.nan,
            'median': np.median(auto_counts) if len(auto_counts) > 0 else np.nan,
            'std': np.std(auto_counts) if len(auto_counts) > 0 else np.nan,
            'values': auto_counts,
        },
        'test': {
            'statistic': stat,
            'pvalue': pval,
        }
    }

    return results


def calculate_gene_density(genes_df, chrom_lengths, mt_genes, mt_chroms):
    """
    Calculate gene density (genes per Mb) for MT and autosomal regions.

    Args:
        genes_df: DataFrame from extract_genes_from_gtf()
        chrom_lengths: dict from get_chromosome_lengths()
        mt_genes: set of MT gene IDs
        mt_chroms: list of MT chromosome IDs

    Returns:
        dict with gene density statistics
    """
    # Classify genes as MT or autosomal
    genes_df['region'] = genes_df['gene_id'].apply(
        lambda x: 'MT' if x in mt_genes else 'Autosomal'
    )

    # Count genes by region
    mt_gene_count = len(genes_df[genes_df['region'] == 'MT'])
    auto_gene_count = len(genes_df[genes_df['region'] == 'Autosomal'])

    # Calculate total genomic lengths
    # MT: sum of MT chromosome lengths
    mt_length = sum(chrom_lengths.get(chrom, 0) for chrom in mt_chroms)

    # Autosomal: all chromosomes except MT and organellar genomes
    # Filter out mitochondrial (NC_) and plastid chromosomes that are typically small
    # We'll use a simple heuristic: exclude very small chromosomes (< 50kb) which are likely organellar
    organellar_threshold = 50000
    auto_length = 0
    for chrom, length in chrom_lengths.items():
        if chrom not in mt_chroms and length >= organellar_threshold:
            auto_length += length

    # Calculate density (genes per Mb)
    mt_density = (mt_gene_count / (mt_length / 1e6)) if mt_length > 0 else np.nan
    auto_density = (auto_gene_count / (auto_length / 1e6)) if auto_length > 0 else np.nan

    results = {
        'MT': {
            'n_genes': mt_gene_count,
            'length_bp': mt_length,
            'length_mb': mt_length / 1e6,
            'density': mt_density,
        },
        'Autosomal': {
            'n_genes': auto_gene_count,
            'length_bp': auto_length,
            'length_mb': auto_length / 1e6,
            'density': auto_density,
        }
    }

    return results


def plot_gene_density(all_results, output_path):
    """
    Create a grouped bar chart comparing gene density across species.

    Args:
        all_results: dict of {species_name: density_results}
        output_path: path to save figure
    """
    species_order = ["M. pusilla", "M. commoda", "B. prasinos", "O. tauri"]

    mt_densities = []
    auto_densities = []

    for species in species_order:
        if species in all_results:
            mt_densities.append(all_results[species]['density']['MT']['density'])
            auto_densities.append(all_results[species]['density']['Autosomal']['density'])
        else:
            mt_densities.append(0)
            auto_densities.append(0)

    x = np.arange(len(species_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, mt_densities, width, label='MT region', color='#d62728')
    bars2 = ax.bar(x + width/2, auto_densities, width, label='Autosomal', color='#1f77b4')

    ax.set_ylabel('Genes per Mb', fontsize=12)
    ax.set_xlabel('Species', fontsize=12)
    ax.set_title('Gene Density: MT vs Autosomal Regions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(species_order, fontstyle='italic')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gene density figure saved to {output_path}")


def plot_intron_abundance(all_results, output_path):
    """
    Create 2x2 subplot grid with box plots comparing intron abundance.

    Args:
        all_results: dict of {species_name: intron_abundance_results}
        output_path: path to save figure
    """
    species_order = ["M. pusilla", "M. commoda", "B. prasinos", "O. tauri"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, species in enumerate(species_order):
        ax = axes[i]

        if species not in all_results:
            ax.set_visible(False)
            continue

        results = all_results[species]['intron_abundance']

        mt_vals = results['MT']['values']
        auto_vals = results['Autosomal']['values']

        # Box plot
        bp = ax.boxplot([mt_vals, auto_vals],
                        tick_labels=['MT', 'Autosomal'],
                        patch_artist=True,
                        widths=0.6)

        # Color boxes
        bp['boxes'][0].set_facecolor('#d62728')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#1f77b4')
        bp['boxes'][1].set_alpha(0.7)

        ax.set_ylabel('Introns per gene', fontsize=11)
        ax.set_xlabel('Region', fontsize=11)

        # Title with species name and statistics
        pval = results['test']['pvalue']
        if not np.isnan(pval):
            pval_str = f"p = {pval:.2e}" if pval < 0.01 else f"p = {pval:.4f}"
        else:
            pval_str = "p = N/A"

        title = f"{species}\n{pval_str}"
        ax.set_title(title, fontsize=12, fontweight='bold', fontstyle='italic')

        # Add median values as text
        mt_median = results['MT']['median']
        auto_median = results['Autosomal']['median']

        if not np.isnan(mt_median):
            ax.text(1, mt_median, f'{mt_median:.1f}',
                   ha='right', va='center', fontsize=9, color='darkred')
        if not np.isnan(auto_median):
            ax.text(2, auto_median, f'{auto_median:.1f}',
                   ha='left', va='center', fontsize=9, color='darkblue')

        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Intron abundance figure saved to {output_path}")


def save_results_tables(all_results, output_dir):
    """
    Save results to TSV files.

    Args:
        all_results: dict of {species_name: {density, intron_abundance}}
        output_dir: directory to save tables
    """
    os.makedirs(output_dir, exist_ok=True)

    # Gene density table
    density_rows = []
    for species_name, results in all_results.items():
        density = results['density']
        for region in ['MT', 'Autosomal']:
            density_rows.append({
                'species': species_name,
                'region': region,
                'n_genes': density[region]['n_genes'],
                'length_mb': density[region]['length_mb'],
                'density_genes_per_mb': density[region]['density'],
            })

    density_df = pd.DataFrame(density_rows)
    density_path = os.path.join(output_dir, "gene_density_stats.tsv")
    density_df.to_csv(density_path, sep='\t', index=False, float_format='%.3f')
    print(f"Gene density table saved to {density_path}")

    # Intron abundance table
    abundance_rows = []
    for species_name, results in all_results.items():
        abundance = results['intron_abundance']
        for region in ['MT', 'Autosomal']:
            abundance_rows.append({
                'species': species_name,
                'region': region,
                'n_genes': abundance[region]['n_genes'],
                'mean_introns': abundance[region]['mean'],
                'median_introns': abundance[region]['median'],
                'std_introns': abundance[region]['std'],
            })

    abundance_df = pd.DataFrame(abundance_rows)
    abundance_path = os.path.join(output_dir, "intron_abundance_stats.tsv")
    abundance_df.to_csv(abundance_path, sep='\t', index=False, float_format='%.3f')
    print(f"Intron abundance table saved to {abundance_path}")

    # Combined summary table
    summary_rows = []
    for species_name, results in all_results.items():
        density = results['density']
        abundance = results['intron_abundance']

        for region in ['MT', 'Autosomal']:
            summary_rows.append({
                'species': species_name,
                'region': region,
                'n_genes': density[region]['n_genes'],
                'genomic_length_mb': density[region]['length_mb'],
                'gene_density_per_mb': density[region]['density'],
                'mean_introns_per_gene': abundance[region]['mean'],
                'median_introns_per_gene': abundance[region]['median'],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "genomic_organization_summary.tsv")
    summary_df.to_csv(summary_path, sep='\t', index=False, float_format='%.3f')
    print(f"Combined summary table saved to {summary_path}")


def print_summary(all_results):
    """Print formatted summary to console."""
    print("\n" + "=" * 100)
    print("GENOMIC ORGANIZATION: MT vs AUTOSOMAL REGIONS")
    print("=" * 100)

    for species_name in ["M. pusilla", "M. commoda", "B. prasinos", "O. tauri"]:
        if species_name not in all_results:
            continue

        results = all_results[species_name]
        density = results['density']
        abundance = results['intron_abundance']

        print(f"\n{species_name}")
        print("-" * 100)

        # Gene density
        print("\nGene Density (genes per Mb):")
        print(f"  MT region:   {density['MT']['n_genes']:>6} genes / {density['MT']['length_mb']:>7.2f} Mb = {density['MT']['density']:>6.1f} genes/Mb")
        print(f"  Autosomal:   {density['Autosomal']['n_genes']:>6} genes / {density['Autosomal']['length_mb']:>7.2f} Mb = {density['Autosomal']['density']:>6.1f} genes/Mb")

        # Intron abundance
        print("\nIntron Abundance (introns per gene):")
        print(f"  MT region:   mean = {abundance['MT']['mean']:>5.2f}, median = {abundance['MT']['median']:>5.2f}, n = {abundance['MT']['n_genes']:>5}")
        print(f"  Autosomal:   mean = {abundance['Autosomal']['mean']:>5.2f}, median = {abundance['Autosomal']['median']:>5.2f}, n = {abundance['Autosomal']['n_genes']:>5}")

        pval = abundance['test']['pvalue']
        if not np.isnan(pval):
            print(f"  Mann-Whitney U test: p = {pval:.2e}")
        else:
            print(f"  Mann-Whitney U test: p = N/A")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze gene density and intron abundance: MT vs autosomal regions."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis",
        help="Directory for output files (default: analysis)",
    )
    args = parser.parse_args()

    all_results = {}

    # Process each species
    for short_name, config in SPECIES_CONFIG.items():
        display_name = config['display_name']
        print(f"\nProcessing {display_name}...")

        # Load data
        print(f"  Loading intron data from {config['data']}")
        intron_df = load_intron_data(config['data'])
        mt_genes = get_mt_gene_set(intron_df)
        print(f"  Found {len(mt_genes)} MT genes from intron data")

        print(f"  Parsing GTF file: {config['gtf']}")
        genes_df = extract_genes_from_gtf(config['gtf'])
        print(f"  Extracted {len(genes_df)} genes from GTF")

        print(f"  Parsing FASTA file: {config['fasta']}")
        chrom_lengths = get_chromosome_lengths(config['fasta'])
        print(f"  Found {len(chrom_lengths)} chromosomes/scaffolds")

        # Calculate metrics
        print(f"  Calculating intron abundance...")
        intron_abundance = calculate_intron_abundance(intron_df, mt_genes)

        print(f"  Calculating gene density...")
        gene_density = calculate_gene_density(genes_df, chrom_lengths, mt_genes, config['mt_chroms'])

        # Store results
        all_results[display_name] = {
            'density': gene_density,
            'intron_abundance': intron_abundance,
        }

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_gene_density(all_results, os.path.join(args.output_dir, "gene_density_comparison.pdf"))
    plot_intron_abundance(all_results, os.path.join(args.output_dir, "intron_abundance_comparison.pdf"))

    # Save tables
    print("\nSaving result tables...")
    save_results_tables(all_results, args.output_dir)

    # Print summary
    print_summary(all_results)

    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
