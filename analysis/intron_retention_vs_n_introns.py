#!/usr/bin/env python3
"""
Analyze Spearman correlation between median intron retention rate per gene
and number of introns per gene, for four species. Runs separate analyses and
scatter plots for autosomal (In_region==0) vs mating_type (In_region==1) genes.
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


# Species: (short name for filenames, display name for plots/terminal)
SPECIES = [
    ("Mpusilla", "M. pusilla"),
    ("Mcommoda", "M. commoda"),
    ("Otauri", "O. tauri"),
    ("Bprasinos", "B. prasinos"),
]

# In_region: 0 = autosomal, 1 = mating_type (MT)
REGION_AUTOSOMAL = 0
REGION_MT = 1


def process_species_region(tsv_path, in_region):
    """
    Load branchpoints TSV, filter by In_region (0=autosomal, 1=mating_type),
    and compute per-gene n_introns and median retention.
    Returns a DataFrame with columns: gene_id, n_introns, median_retention.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["In_region"] == in_region].copy()
    df["rMATS_retention_ratio"] = pd.to_numeric(df["rMATS_retention_ratio"], errors="coerce")

    gene = df.groupby("gene_id").agg(
        n_introns=("gene_id", "size"),
        median_retention=("rMATS_retention_ratio", "median"),
    ).reset_index()

    gene = gene.dropna(subset=["median_retention"])
    return gene


def run_region(region_name, in_region, args, results_by_region, data_by_region):
    """Process all species for one region; append to results and data dicts."""
    results = []
    per_species = {}

    for short_name, display_name in SPECIES:
        tsv_path = os.path.join(args.data_dir, f"{short_name}.features.branchpoints.tsv")
        if not os.path.isfile(tsv_path):
            print(f"Warning: {tsv_path} not found, skipping {display_name} ({region_name})")
            continue

        gene_df = process_species_region(tsv_path, in_region)
        n_genes = len(gene_df)
        if n_genes < 2:
            print(f"Warning: {display_name} {region_name} has < 2 genes with valid retention, skipping")
            continue

        rho, pval = spearmanr(
            gene_df["n_introns"],
            gene_df["median_retention"],
            nan_policy="omit",
        )
        results.append(
            {
                "species": display_name,
                "rho": rho,
                "pvalue": pval,
                "n_genes": n_genes,
            }
        )
        per_species[display_name] = gene_df

    results_by_region[region_name] = results
    data_by_region[region_name] = per_species
    return results, per_species


def save_scatter_figure(per_species, results, out_path, region_label):
    """Save a 2x2 scatter figure for one region (autosomal or mating_type)."""
    if not per_species:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    result_by_name = {r["species"]: r for r in results}

    for i, (_short_name, display_name) in enumerate(SPECIES):
        ax = axes[i]
        if display_name not in per_species:
            ax.set_visible(False)
            continue
        gene_df = per_species[display_name]
        r = result_by_name.get(display_name, {})

        ax.scatter(
            gene_df["n_introns"],
            gene_df["median_retention"],
            alpha=0.5,
            s=10,
            rasterized=True,
        )
        ax.set_xlabel("Introns per gene")
        ax.set_ylabel("Median intron retention rate")
        title = f"{display_name} ({region_label})"
        if r:
            title += f"\nρ = {r['rho']:.3f}, p = {r['pvalue']:.2e}"
        ax.set_title(title)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Correlate median intron retention per gene with number of introns per gene (autosomal vs mating_type)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing *.features.branchpoints.tsv files (default: data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/intron_retention_vs_n_introns.pdf",
        help="Base output path for figures; _autosomal.pdf and _mating_type.pdf will be used (default: analysis/intron_retention_vs_n_introns.pdf)",
    )
    args = parser.parse_args()

    # Derive autosomal and MT output paths from base path
    base = args.output.rstrip(".pdf").rstrip("/")
    out_autosomal = f"{base}_autosomal.pdf"
    out_mt = f"{base}_mating_type.pdf"

    results_by_region = {}
    data_by_region = {}

    # Autosomal (In_region == 0)
    run_region("Autosomal", REGION_AUTOSOMAL, args, results_by_region, data_by_region)
    # Mating type (In_region == 1)
    run_region("Mating type (MT)", REGION_MT, args, results_by_region, data_by_region)

    # Terminal output: two tables
    print("Median intron retention rate vs number of introns per gene (Spearman)")
    print()
    for region_name in ("Autosomal", "Mating type (MT)"):
        results = results_by_region.get(region_name, [])
        print(f"--- {region_name} (In_region == {REGION_AUTOSOMAL if region_name == 'Autosomal' else REGION_MT}) ---")
        print("=" * 70)
        print(f"{'Species':<16} {'rho':>10} {'p-value':>12} {'n_genes':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['species']:<16} {r['rho']:>10.4f} {r['pvalue']:>12.2e} {r['n_genes']:>10}")
        print("=" * 70)
        print()

    # Separate scatter plots for autosomal and mating_type
    for region_name, in_region in [("Autosomal", REGION_AUTOSOMAL), ("Mating type (MT)", REGION_MT)]:
        per_species = data_by_region.get(region_name, {})
        results = results_by_region.get(region_name, [])
        out_path = out_autosomal if region_name == "Autosomal" else out_mt
        save_scatter_figure(per_species, results, out_path, region_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
