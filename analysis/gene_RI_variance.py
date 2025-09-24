#!/usr/bin/env python3
import argparse
import pandas as pd
import statsmodels.formula.api as smf

def main():
    parser = argparse.ArgumentParser(
        description="Fit a linear mixed model of intron retention ratios by gene."
    )
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Input TSV file from featureCounts (with intron features)."
    )
    args = parser.parse_args()

    # Load the file
    data = pd.read_csv(args.input, sep="\t")

    # Filter by In_region
    data = data[data["In_region"] == 1]

    # Keep only genes with at least 2 introns
    gene_counts = data["gene_id"].value_counts()
    genes_with_2plus = gene_counts[gene_counts >= 2].index
    data = data[data["gene_id"].isin(genes_with_2plus)]

    print(f"Using {len(data)} introns from {len(genes_with_2plus)} genes")

    # Fit linear mixed model: retention_ratio ~ 1 + (1|gene_id)
    md = smf.mixedlm("rMATS_retention_ratio ~ 1", data, groups=data["gene_id"])
    mdf = md.fit()
    print(mdf.summary())

    # Extract gene-level variance
    var_gene = mdf.cov_re.iloc[0, 0]
    var_residual = mdf.scale
    intra_gene_correlation = var_gene / (var_gene + var_residual)
    print(f"\nApproximate intra-gene correlation of intron retention: {intra_gene_correlation:.3f}")

if __name__ == "__main__":
    main()
