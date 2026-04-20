#!/usr/bin/env python
"""Per-gene Spearman correlations within MT region (RTS-filtered).

For each gene in the MT region, compute:
  - mean perc_A_downstream_TTS across its isoforms
  - fraction of isoforms with intron retention (has_IR)
  - number of isoforms

Then test:
  1. Spearman(mean_perc_A, fraction_IR)
  2. Spearman(mean_perc_A, n_isoforms)

Uses the RTS-filtered per-isoform table from internal_priming_vs_IR.py as input,
plus the SQANTI3 classification for the associated_gene mapping.
"""
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IR_SUBCATEGORIES = {"intron_retention", "mono-exon_by_intron_retention"}


def parse_gtf_attrs(s):
    out = {}
    for kv in s.strip().rstrip(";").split(";"):
        kv = kv.strip()
        if not kv:
            continue
        k, _, v = kv.partition(" ")
        out[k] = v.strip().strip('"')
    return out


def load_isoform_gene_map(gtf_path):
    """Return dict isoform_id -> gene_id from the Mandalorion GTF."""
    gene_of = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            attrs = parse_gtf_attrs(fields[8])
            tid = attrs.get("transcript_id")
            gid = attrs.get("gene_id")
            if tid and gid:
                gene_of[tid] = gid
    return gene_of


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isoform-tsv", required=True,
                   help="RTS-filtered per-isoform TSV from internal_priming_vs_IR.py")
    p.add_argument("--isoform-gtf", required=True,
                   help="Mandalorion isoform GTF (for gene_id mapping)")
    p.add_argument("--out-png", required=True)
    p.add_argument("--out-tsv", required=True, help="Per-gene summary TSV")
    args = p.parse_args()

    iso = pd.read_csv(args.isoform_tsv, sep="\t")
    gene_map = load_isoform_gene_map(args.isoform_gtf)
    iso["gene_id"] = iso["isoform"].map(gene_map)
    n_unmapped = iso["gene_id"].isna().sum()
    if n_unmapped:
        print(f"WARNING: {n_unmapped} isoforms with no gene_id in GTF (dropped)")
        iso = iso.dropna(subset=["gene_id"])

    df = iso

    # Restrict to MT
    mt = df[df["is_MT"] == 1].reset_index(drop=True)
    print(f"MT isoforms (RTS-filtered): {len(mt)}")

    # Per-gene aggregation
    gene = mt.groupby("gene_id").agg(
        mean_perc_A=("perc_A_downstream_TTS", "mean"),
        n_isoforms=("isoform", "count"),
        n_IR=("has_IR", "sum"),
    ).reset_index()
    gene["frac_IR"] = gene["n_IR"] / gene["n_isoforms"]
    # Drop genes with only 1 isoform for the frac_IR correlation (no variance)
    gene_multi = gene[gene["n_isoforms"] > 1]

    print(f"MT genes total: {len(gene)}")
    print(f"MT genes with >1 isoform: {len(gene_multi)}")

    # Spearman correlations
    rho1, p1 = spearmanr(gene_multi["mean_perc_A"], gene_multi["frac_IR"])
    rho2, p2 = spearmanr(gene["mean_perc_A"], gene["n_isoforms"])

    print(f"\n=== Spearman: mean_perc_A vs frac_IR (genes with >1 isoform) ===")
    print(f"  n={len(gene_multi)}  rho={rho1:+.4f}  p={p1:.4e}")
    print(f"\n=== Spearman: mean_perc_A vs n_isoforms (all genes) ===")
    print(f"  n={len(gene)}  rho={rho2:+.4f}  p={p2:.4e}")

    # Write per-gene TSV
    gene.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\nWrote per-gene TSV: {args.out_tsv}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x = gene_multi["mean_perc_A"]
    y = gene_multi["frac_IR"]
    ax.scatter(x, y, s=15, alpha=0.4, color="#d62728", edgecolors="none")
    # Trend line
    z = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, np.polyval(z, xline), color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Mean % A downstream of TTS")
    ax.set_ylabel("Fraction of isoforms with IR")
    ax.set_title(
        f"A. mean perc_A vs fraction IR\n"
        f"(MT genes, >1 isoform, n={len(gene_multi)})\n"
        f"Spearman ρ = {rho1:+.3f}, p = {p1:.2e}"
    )
    ax.set_xlim(0, max(x.max() + 2, 55))
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    x = gene["mean_perc_A"]
    y = gene["n_isoforms"]
    ax.scatter(x, y, s=15, alpha=0.4, color="#d62728", edgecolors="none")
    z = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, np.polyval(z, xline), color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Mean % A downstream of TTS")
    ax.set_ylabel("Number of isoforms")
    ax.set_title(
        f"B. mean perc_A vs isoform count\n"
        f"(MT genes, n={len(gene)})\n"
        f"Spearman ρ = {rho2:+.3f}, p = {p2:.2e}"
    )
    ax.set_xlim(0, max(x.max() + 2, 55))

    fig.suptitle(
        "Per-gene Spearman correlations — MT region, RTS-filtered",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"Wrote plot: {args.out_png}")


if __name__ == "__main__":
    main()
