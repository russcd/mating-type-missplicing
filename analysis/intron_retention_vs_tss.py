#!/usr/bin/env python3
"""
Analyze intron retention rate as a function of distance from the transcription
start site (TSS) in M. pusilla long-read isoforms.  Compares the mating-type
region (scaffold_2:45993-1729827) against the rest of the genome.

Outputs:
  - analysis/intron_retention_vs_tss.tsv  per-intron table
  - analysis/intron_retention_vs_tss.pdf  multi-panel figure
"""

import sys
import os

# Reuse helper functions from rna_extension.py
sys.path.insert(0, "/scratch2/russ/introner/splicing_fails/isoforms/sensitiveIsoforms")
from rna_extension import (
    build_canonical_transcript_introns,
    build_gene_to_transcript_map,
    build_isoform_exons,
    intron_overlaps_exon,
    parse_gff3_features,
)

from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CANONICAL_GFF3 = "/scratch2/russ/introner/splicing_fails/isoforms/MpusillaCCMP1545_228_gene_exons.gff3"
ISOFORM_GTF = "/scratch2/russ/introner/splicing_fails/isoforms/sensitiveIsoforms/09092025_834_Isoforms.filtered.clean.gtf"
ISOFORM_QUANT = "/scratch2/russ/introner/splicing_fails/isoforms/sensitiveIsoforms/09092025_834_Isoforms.filtered.clean.quant"

OUT_TSV = "analysis/intron_retention_vs_tss.tsv"
OUT_PDF = "analysis/intron_retention_vs_tss.pdf"

# Mating-type region
MT_SCAFFOLD = "scaffold_2"
MT_START = 45993
MT_END = 1729827


# ---------------------------------------------------------------------------
# 1. Parse gene-level info from GFF3 (scaffold, start, end, strand)
# ---------------------------------------------------------------------------
def build_gene_info(gff3_path):
    """Return dict gene_id -> {scaffold, start, end, strand}."""
    gene_info = {}
    for f in parse_gff3_features(gff3_path, types=["gene"]):
        gid = f.attrs.get("ID")
        if gid:
            gene_info[gid] = {
                "scaffold": f.seqid,
                "start": f.start,
                "end": f.end,
                "strand": f.strand,
            }
    return gene_info


# ---------------------------------------------------------------------------
# 2. Build isoform -> gene mapping from isoform GTF
# ---------------------------------------------------------------------------
def build_isoform_to_gene(gtf_path):
    """Return dict isoform_id -> gene_id from the isoform GTF."""
    iso2gene = {}
    for f in parse_gff3_features(gtf_path, types=["transcript"]):
        tid = f.attrs.get("transcript_id")
        gid = f.attrs.get("gene_id")
        if tid and gid:
            iso2gene[tid] = gid
    return iso2gene


# ---------------------------------------------------------------------------
# 3. Load quant data: total expression per isoform
# ---------------------------------------------------------------------------
def load_quant(quant_path):
    """Return dict isoform_id -> total_expression (sum across replicates)."""
    df = pd.read_csv(quant_path, sep="\t")
    iso_expr = {}
    rep_cols = [c for c in df.columns if c not in ("Isoform", "Gene", "")]
    # Drop any non-numeric columns that slipped in
    rep_cols = [c for c in rep_cols if c.strip()]
    for _, row in df.iterrows():
        iso = row["Isoform"]
        gene = row["Gene"]
        total = sum(row[c] for c in rep_cols if pd.notna(row[c]))
        iso_expr[iso] = {"gene": gene, "expr": total}
    return iso_expr


# ---------------------------------------------------------------------------
# 4. Per-intron retention classification
# ---------------------------------------------------------------------------
def classify_per_intron(
    gene_info,
    gene_to_transcripts,
    transcript_introns,
    iso_exons,
    iso_expr,
    iso2gene,
):
    """
    For every canonical intron in every gene that has at least one isoform,
    compute weighted and unweighted retention rates across isoforms.

    Returns a list of dicts (one per intron).
    """
    # Invert: gene_id -> list of isoform_ids that map to it
    gene_isoforms = {}
    for iso, gid in iso2gene.items():
        gene_isoforms.setdefault(gid, []).append(iso)

    rows = []
    for gene_id, info in gene_info.items():
        # Get canonical transcript(s) for this gene
        transcripts = gene_to_transcripts.get(gene_id, [])
        if not transcripts:
            continue

        # Collect all introns from canonical transcripts (use longest list)
        best_introns = []
        for tid in transcripts:
            introns = transcript_introns.get(tid, [])
            if len(introns) > len(best_introns):
                best_introns = introns

        if not best_introns:
            continue  # intronless gene

        # Get isoforms for this gene
        isoforms = gene_isoforms.get(gene_id, [])
        if not isoforms:
            continue

        strand = info["strand"]
        gene_start = info["start"]
        gene_end = info["end"]

        # Order introns from 5' end
        if strand == "+":
            ordered_introns = sorted(best_introns, key=lambda x: x[0])
            five_prime_end = gene_start
        else:
            ordered_introns = sorted(best_introns, key=lambda x: -x[0])
            five_prime_end = gene_end

        n_introns = len(ordered_introns)

        for ordinal_pos, (i_start, i_end) in enumerate(ordered_introns, 1):
            # Absolute bp distance from 5' end of gene
            if strand == "+":
                bp_dist = i_start - five_prime_end
            else:
                bp_dist = five_prime_end - i_end

            # Check each isoform for retention of this intron
            n_retaining = 0
            n_total = 0
            weighted_retained = 0.0
            total_weight = 0.0

            for iso in isoforms:
                exons = iso_exons.get(iso)
                if exons is None:
                    continue
                expr = iso_expr.get(iso, {}).get("expr", 0)

                retained = any(
                    intron_overlaps_exon((i_start, i_end), exon)
                    for exon in exons
                )

                n_total += 1
                total_weight += expr
                if retained:
                    n_retaining += 1
                    weighted_retained += expr

            if n_total == 0:
                continue

            unweighted_rate = n_retaining / n_total
            weighted_rate = (
                weighted_retained / total_weight if total_weight > 0 else np.nan
            )

            # Mating-type flag
            in_mt = (
                info["scaffold"] == MT_SCAFFOLD
                and gene_start <= MT_END
                and gene_end >= MT_START
            )

            rows.append(
                {
                    "gene_id": gene_id,
                    "scaffold": info["scaffold"],
                    "strand": strand,
                    "gene_start": gene_start,
                    "gene_end": gene_end,
                    "intron_start": i_start,
                    "intron_end": i_end,
                    "ordinal_position": ordinal_pos,
                    "n_introns_in_gene": n_introns,
                    "bp_distance_from_tss": bp_dist,
                    "n_isoforms_total": n_total,
                    "n_isoforms_retaining": n_retaining,
                    "unweighted_retention_rate": round(unweighted_rate, 4),
                    "weighted_retention_rate": round(weighted_rate, 4)
                    if not np.isnan(weighted_rate)
                    else np.nan,
                    "in_mating_type": in_mt,
                }
            )

    return rows


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------
def compute_spearman(df, metric="unweighted_retention_rate"):
    """Compute Spearman correlation for mating-type and autosomal regions."""
    results = {}
    for label, subset in [
        ("Autosomal", df[~df["in_mating_type"]]),
        ("Mating-type", df[df["in_mating_type"]]),
    ]:
        s = subset.dropna(subset=[metric])
        if len(s) < 3:
            results[label] = (np.nan, np.nan)
            continue
        rho, pval = spearmanr(s["bp_distance_from_tss"], s[metric])
        results[label] = (rho, pval)
    return results


def run_logistic_regression(df):
    """Logistic regression: ever-retained ~ distance/1000 + n_introns_in_gene."""
    print("\n=== Logistic Regression: P(ever retained) ~ distance + n_introns ===")
    for label, subset in [
        ("Autosomal", df[~df["in_mating_type"]]),
        ("Mating-type", df[df["in_mating_type"]]),
    ]:
        s = subset.dropna(subset=["unweighted_retention_rate"]).copy()
        if len(s) < 10:
            print(f"\n{label}: too few observations ({len(s)})")
            continue
        y = (s["n_isoforms_retaining"] > 0).astype(int)
        X = s[["bp_distance_from_tss", "n_introns_in_gene"]].copy()
        X["bp_distance_from_tss"] = X["bp_distance_from_tss"] / 1000.0
        X = sm.add_constant(X)
        try:
            model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            print(f"\n{label} (n={len(s)}):")
            print(f"  {'Predictor':<25s} {'Coef':>8s} {'OR':>8s} {'p-value':>10s}")
            for name in model.params.index:
                coef = model.params[name]
                pval = model.pvalues[name]
                odds_ratio = np.exp(coef)
                print(f"  {name:<25s} {coef:>8.4f} {odds_ratio:>8.4f} {pval:>10.4g}")
        except Exception as e:
            print(f"\n{label}: GLM failed — {e}")


def make_plot(df, out_path):
    """Two-panel figure using unweighted retention rate.

    Panel A: violin plot by ordinal position.
    Panel B: scatter + LOESS smoothing by bp distance from TSS, annotated
             with Spearman rho and p-value.
    """
    metric = "unweighted_retention_rate"
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    mt = df[df["in_mating_type"]]
    auto = df[~df["in_mating_type"]]

    # --- Panel A: violin plot by ordinal position ---
    max_pos = min(int(df["ordinal_position"].quantile(0.95)), 15)
    positions = range(1, max_pos + 1)

    for region, subset, color, offset in [
        ("Autosomal", auto, "#4477AA", -0.15),
        ("Mating-type", mt, "#FF8C00", 0.15),
    ]:
        bp_data = [
            subset.loc[subset["ordinal_position"] == p, metric].dropna()
            for p in positions
        ]
        valid = [(p, d) for p, d in zip(positions, bp_data) if len(d) > 0]
        if not valid:
            continue
        vp = ax_a.violinplot(
            [d.values for _, d in valid],
            positions=[p + offset for p, _ in valid],
            widths=0.25,
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.6)
        vp["cmeans"].set_color(color)
        ax_a.plot([], [], color=color, label=region, linewidth=4, alpha=0.6)

    ax_a.set_xlabel("Ordinal intron position (from 5' end)")
    ax_a.set_ylabel("Unweighted retention rate")
    ax_a.set_title("A. Intron retention by ordinal position")
    ax_a.set_xticks(list(positions))
    ax_a.legend()
    ax_a.set_ylim(-0.05, 1.05)

    # --- Panel B: scatter + LOESS by bp distance ---
    spearman_results = compute_spearman(df, metric)
    y_text = 0.95
    for region, subset, color in [
        ("Autosomal", auto, "#4477AA"),
        ("Mating-type", mt, "#FF8C00"),
    ]:
        s = subset.dropna(subset=[metric])
        ax_b.scatter(
            s["bp_distance_from_tss"],
            s[metric],
            alpha=0.15,
            s=8,
            color=color,
            rasterized=True,
        )
        # LOESS smoothing
        if len(s) > 10:
            s_clip = s[s["bp_distance_from_tss"] <= 3000]
            if len(s_clip) > 10:
                smoothed = lowess(
                    s_clip[metric].values,
                    s_clip["bp_distance_from_tss"].values,
                    frac=0.3,
                )
                ax_b.plot(
                    smoothed[:, 0],
                    smoothed[:, 1],
                    color=color,
                    linewidth=2,
                    label=region,
                )
        # Annotate with Spearman stats
        rho, pval = spearman_results.get(region, (np.nan, np.nan))
        if not np.isnan(rho):
            pstr = f"{pval:.2g}" if pval >= 1e-4 else f"{pval:.2e}"
            ax_b.text(
                0.98, y_text,
                f"{region}: ρ={rho:.3f}, p={pstr}",
                transform=ax_b.transAxes,
                ha="right", va="top",
                fontsize=9,
                color=color,
            )
            y_text -= 0.06

    ax_b.set_xlabel("Distance from TSS (bp)")
    ax_b.set_ylabel("Unweighted retention rate")
    ax_b.set_title("B. Intron retention by distance from TSS")
    ax_b.legend()
    ax_b.set_xlim(0, 3000)
    ax_b.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading canonical GFF3...")
    gene_info = build_gene_info(CANONICAL_GFF3)
    print(f"  {len(gene_info)} genes")

    transcript_introns, transcript_exons = build_canonical_transcript_introns(CANONICAL_GFF3)
    gene_to_transcripts = build_gene_to_transcript_map(CANONICAL_GFF3)
    print(f"  {len(gene_to_transcripts)} genes with transcripts")

    print("Loading isoform GTF...")
    iso_exons = build_isoform_exons(ISOFORM_GTF)
    iso2gene = build_isoform_to_gene(ISOFORM_GTF)
    print(f"  {len(iso_exons)} isoforms with exons")

    print("Loading expression data...")
    iso_expr = load_quant(ISOFORM_QUANT)
    print(f"  {len(iso_expr)} isoforms with expression")

    print("Classifying per-intron retention...")
    rows = classify_per_intron(
        gene_info,
        gene_to_transcripts,
        transcript_introns,
        iso_exons,
        iso_expr,
        iso2gene,
    )
    df = pd.DataFrame(rows)
    print(f"  {len(df)} introns classified")

    # Filter to genes with >=2 introns
    df = df[df["n_introns_in_gene"] >= 2].copy()
    print(f"  {len(df)} introns after filtering to genes with >=2 introns")

    n_mt = df["in_mating_type"].sum()
    n_auto = (~df["in_mating_type"]).sum()
    print(f"  Mating-type introns: {n_mt}, Autosomal introns: {n_auto}")

    # Save TSV
    os.makedirs(os.path.dirname(OUT_TSV) or ".", exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"Table saved to {OUT_TSV}")

    # Generate plot (unweighted only)
    print("\nGenerating plot...")
    make_plot(df, OUT_PDF)

    # --- Statistical tests ---
    print("\n=== Spearman Correlation: distance vs unweighted retention ===")
    spearman = compute_spearman(df)
    for label, (rho, pval) in spearman.items():
        if np.isnan(rho):
            print(f"{label}: insufficient data")
        else:
            print(f"{label}: rho={rho:.4f}, p={pval:.4g}")

    run_logistic_regression(df)

    # Summary stats
    print("\n--- Summary ---")
    for region, subset in [
        ("Autosomal", df[~df["in_mating_type"]]),
        ("Mating-type", df[df["in_mating_type"]]),
    ]:
        if len(subset) == 0:
            print(f"{region}: no introns")
            continue
        ur = subset["unweighted_retention_rate"].dropna()
        print(
            f"{region}: {len(subset)} introns, "
            f"unweighted mean={ur.mean():.4f}, median={ur.median():.4f}"
        )


if __name__ == "__main__":
    main()
