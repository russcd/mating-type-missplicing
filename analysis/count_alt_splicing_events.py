#!/usr/bin/env python
"""Count alternative splicing events from Mandalorion R2C2 isoforms.
Counts for A5SS / A3SS / SE use pairwise-alternative semantics: if an acceptor
site has n donors, that contributes (n-1) A5SS events; and so on.
"""
import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

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


def load_isoforms(gtf_path):
    """Return dict iso_id -> {'chrom','strand','gene_id','exons'}.

    exons is a sorted list of (start, end) tuples (1-based inclusive).
    """
    exons_map = defaultdict(list)
    meta = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            attrs = parse_gtf_attrs(fields[8])
            tid = attrs.get("transcript_id")
            if not tid:
                continue
            if fields[2] == "transcript":
                meta[tid] = {
                    "chrom": fields[0],
                    "strand": fields[6],
                    "gene_id": attrs.get("gene_id", tid),
                }
            elif fields[2] == "exon":
                exons_map[tid].append((int(fields[3]), int(fields[4])))
                if tid not in meta:
                    meta[tid] = {
                        "chrom": fields[0],
                        "strand": fields[6],
                        "gene_id": attrs.get("gene_id", tid),
                    }
    isoforms = {}
    for tid, ex in exons_map.items():
        ex.sort()
        isoforms[tid] = {
            **meta.get(tid, {"chrom": "", "strand": ".", "gene_id": tid}),
            "exons": ex,
        }
    return isoforms


def load_sqanti_flags(sqanti_path):
    """Return (rts_isos, ir_isos): sets of isoform IDs flagged in SQANTI3."""
    df = pd.read_csv(
        sqanti_path,
        sep="\t",
        usecols=["isoform", "RTS_stage", "subcategory"],
        low_memory=False,
    )
    rts = set(df.loc[df["RTS_stage"] == True, "isoform"].astype(str))
    ir = set(
        df.loc[df["subcategory"].isin(IR_SUBCATEGORIES), "isoform"].astype(str)
    )
    return rts, ir


def junctions_from_exons(exons):
    """Return list of (intron_start, intron_end) tuples (1-based inclusive)."""
    return [
        (exons[i][1] + 1, exons[i + 1][0] - 1)
        for i in range(len(exons) - 1)
        if exons[i + 1][0] - 1 >= exons[i][1] + 1
    ]


def count_alt_endpoint(junctions, group_idx):
    """Count pairwise-alternative events given a grouping endpoint.

    group_idx=0 groups by intron_start (counts distinct ends per start).
    group_idx=1 groups by intron_end (counts distinct starts per end).
    Returns sum over groups of max(0, len(group)-1).
    """
    groups = defaultdict(set)
    other = 1 - group_idx
    for jnc in junctions:
        groups[jnc[group_idx]].add(jnc[other])
    return sum(max(0, len(v) - 1) for v in groups.values())


def count_skipped_exons(gene_exons, gene_junctions):
    """Number of distinct exons fully spanned by at least one junction."""
    skipped = set()
    for ex_s, ex_e in gene_exons:
        for j_s, j_e in gene_junctions:
            if j_s < ex_s and j_e > ex_e:
                skipped.add((ex_s, ex_e))
                break
    return len(skipped)


def count_retained_junctions(gene_exons, gene_junctions):
    """Number of distinct junctions fully contained in at least one exon."""
    retained = set()
    for j_s, j_e in gene_junctions:
        for ex_s, ex_e in gene_exons:
            if ex_s <= j_s and ex_e >= j_e:
                retained.add((j_s, j_e))
                break
    return len(retained)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isoform-gtf", required=True)
    p.add_argument("--sqanti", required=True)
    p.add_argument("--mt-chrom", default="scaffold_2")
    p.add_argument("--mt-start", type=int, default=45993)
    p.add_argument("--mt-end", type=int, default=1729827)
    p.add_argument("--out-gene-tsv", required=True)
    p.add_argument("--out-summary-tsv", required=True)
    p.add_argument(
        "--out-fisher-tsv",
        required=True,
        help="Per-event-type Fisher's exact test of MT vs autosomal prevalence",
    )
    p.add_argument(
        "--out-plot-base",
        required=True,
        help="Base path; writes {base}.pdf and {base}.png",
    )
    args = p.parse_args()

    print("Loading isoform GTF...")
    isoforms = load_isoforms(args.isoform_gtf)
    print(f"  {len(isoforms)} isoforms")

    print("Loading SQANTI3 flags...")
    rts, ir_isos = load_sqanti_flags(args.sqanti)
    print(f"  {len(rts)} RTS artifacts to exclude")
    print(f"  {len(ir_isos)} SQANTI3 IR isoforms")

    kept = [tid for tid in isoforms if tid not in rts]
    print(f"  {len(kept)} isoforms remain after RTS filter")

    by_gene = defaultdict(list)
    for tid in kept:
        by_gene[isoforms[tid]["gene_id"]].append(tid)
    print(f"  {len(by_gene)} genes total")

    rows = []
    for gid, members in by_gene.items():
        n_iso = len(members)
        n_ir_iso = sum(1 for t in members if t in ir_isos)
        chroms = {isoforms[t]["chrom"] for t in members}
        strands = {isoforms[t]["strand"] for t in members}
        chrom = next(iter(chroms)) if len(chroms) == 1 else "mixed"
        strand = next(iter(strands)) if len(strands) == 1 else "."
        gstart = min(isoforms[t]["exons"][0][0] for t in members)
        gend = max(isoforms[t]["exons"][-1][1] for t in members)

        if n_iso < 2:
            rows.append({
                "gene_id": gid, "chrom": chrom, "gene_start": gstart,
                "gene_end": gend, "strand": strand, "n_isoforms": n_iso,
                "n_junctions": 0, "n_exons": 0,
                "n_A5SS": 0, "n_A3SS": 0, "n_SE": 0,
                "n_RI_junctions": 0, "n_IR_isoforms": n_ir_iso,
            })
            continue

        gene_junctions = set()
        gene_exons = set()
        for tid in members:
            iso = isoforms[tid]
            for ex in iso["exons"]:
                gene_exons.add(ex)
            for j in junctions_from_exons(iso["exons"]):
                gene_junctions.add(j)

        jlist = sorted(gene_junctions)

        if strand == "+":
            n_A5SS = count_alt_endpoint(jlist, 1)  # group by intron_end
            n_A3SS = count_alt_endpoint(jlist, 0)  # group by intron_start
        elif strand == "-":
            n_A5SS = count_alt_endpoint(jlist, 0)
            n_A3SS = count_alt_endpoint(jlist, 1)
        else:
            n_A5SS = count_alt_endpoint(jlist, 1)
            n_A3SS = count_alt_endpoint(jlist, 0)

        n_SE = count_skipped_exons(gene_exons, gene_junctions)
        n_RI_j = count_retained_junctions(gene_exons, gene_junctions)

        rows.append({
            "gene_id": gid, "chrom": chrom, "gene_start": gstart,
            "gene_end": gend, "strand": strand, "n_isoforms": n_iso,
            "n_junctions": len(gene_junctions), "n_exons": len(gene_exons),
            "n_A5SS": n_A5SS, "n_A3SS": n_A3SS, "n_SE": n_SE,
            "n_RI_junctions": n_RI_j, "n_IR_isoforms": n_ir_iso,
        })

    gene_df = pd.DataFrame(rows)
    gene_df["region"] = np.where(
        (gene_df["chrom"] == args.mt_chrom)
        & (gene_df["gene_start"] >= args.mt_start)
        & (gene_df["gene_end"] <= args.mt_end),
        "MT", "autosomal",
    )
    gene_df.to_csv(args.out_gene_tsv, sep="\t", index=False)
    print(f"\nWrote per-gene TSV: {args.out_gene_tsv}")

    event_cols = ["n_A5SS", "n_A3SS", "n_SE", "n_RI_junctions", "n_IR_isoforms"]

    summary_rows = []
    for region in ["MT", "autosomal"]:
        sub = gene_df[gene_df["region"] == region]
        sub_multi = sub[sub["n_isoforms"] >= 2]
        for col in event_cols:
            vals = sub_multi[col]
            summary_rows.append({
                "region": region,
                "metric": col,
                "total": int(vals.sum()),
                "mean_per_gene": float(vals.mean()) if len(vals) else 0.0,
                "median_per_gene": float(vals.median()) if len(vals) else 0.0,
                "n_genes_any_event": int((vals > 0).sum()),
                "n_multi_iso_genes": int(len(sub_multi)),
                "n_genes_total": int(len(sub)),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_summary_tsv, sep="\t", index=False)
    print(f"Wrote summary TSV: {args.out_summary_tsv}")

    print()
    print("=== Summary: events by type and region ===")
    print(
        f"  {'event':<18s} {'MT total':>10s} {'AU total':>10s} "
        f"{'MT/gene':>10s} {'AU/gene':>10s}"
    )
    for col in event_cols:
        mt = summary_df[(summary_df["region"] == "MT")
                        & (summary_df["metric"] == col)].iloc[0]
        au = summary_df[(summary_df["region"] == "autosomal")
                        & (summary_df["metric"] == col)].iloc[0]
        print(
            f"  {col:<18s} {mt['total']:>10d} {au['total']:>10d} "
            f"{mt['mean_per_gene']:>10.3f} {au['mean_per_gene']:>10.3f}"
        )
    mt_n = int(summary_df[summary_df["region"] == "MT"]
               ["n_multi_iso_genes"].iloc[0])
    au_n = int(summary_df[summary_df["region"] == "autosomal"]
               ["n_multi_iso_genes"].iloc[0])
    print(f"  Multi-isoform genes: MT={mt_n}  autosomal={au_n}")

    # Fisher's exact test of prevalence per event type (MT vs autosomal)
    # Contingency table for each metric:
    #            MT                       autosomal
    # has event  n_MT_with                n_AU_with
    # no event   n_MT_without             n_AU_without
    fisher_rows = []
    n_tests = len(event_cols)
    for col in event_cols:
        mt = summary_df[(summary_df["region"] == "MT")
                        & (summary_df["metric"] == col)].iloc[0]
        au = summary_df[(summary_df["region"] == "autosomal")
                        & (summary_df["metric"] == col)].iloc[0]
        mt_with = int(mt["n_genes_any_event"])
        mt_without = int(mt["n_multi_iso_genes"]) - mt_with
        au_with = int(au["n_genes_any_event"])
        au_without = int(au["n_multi_iso_genes"]) - au_with
        odds_ratio, p_raw = fisher_exact(
            [[mt_with, au_with], [mt_without, au_without]],
            alternative="two-sided",
        )
        p_bonf = min(1.0, p_raw * n_tests)
        fisher_rows.append({
            "metric": col,
            "mt_genes_with_event": mt_with,
            "mt_genes_without_event": mt_without,
            "au_genes_with_event": au_with,
            "au_genes_without_event": au_without,
            "mt_pct_with_event": mt_with / int(mt["n_multi_iso_genes"]),
            "au_pct_with_event": au_with / int(au["n_multi_iso_genes"]),
            "odds_ratio": float(odds_ratio),
            "p_fisher": float(p_raw),
            "p_bonferroni": float(p_bonf),
        })
    fisher_df = pd.DataFrame(fisher_rows)
    fisher_df.to_csv(args.out_fisher_tsv, sep="\t", index=False)
    print(f"\nWrote Fisher's exact TSV: {args.out_fisher_tsv}")

    print()
    print("=== Fisher's exact: MT vs autosomal prevalence per event type ===")
    print(
        f"  {'event':<18s} {'MT %':>8s} {'AU %':>8s} {'OR':>8s} "
        f"{'p_fisher':>11s} {'p_bonf':>11s}"
    )
    for r in fisher_rows:
        print(
            f"  {r['metric']:<18s} "
            f"{r['mt_pct_with_event']*100:>7.2f}% "
            f"{r['au_pct_with_event']*100:>7.2f}% "
            f"{r['odds_ratio']:>8.2f} "
            f"{r['p_fisher']:>11.2e} "
            f"{r['p_bonferroni']:>11.2e}"
        )

    # Plot: grouped bars, event type x region, total counts + per-gene means
    event_labels = {
        "n_A5SS": "A5SS",
        "n_A3SS": "A3SS",
        "n_SE": "Exon\nskipping",
        "n_RI_junctions": "Retained\nintron\n(junction)",
        "n_IR_isoforms": "IR isoforms\n(SQANTI3)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(event_cols))
    width = 0.38

    mt_totals = [
        int(summary_df[(summary_df["region"] == "MT")
                       & (summary_df["metric"] == c)]["total"].iloc[0])
        for c in event_cols
    ]
    au_totals = [
        int(summary_df[(summary_df["region"] == "autosomal")
                       & (summary_df["metric"] == c)]["total"].iloc[0])
        for c in event_cols
    ]
    mt_means = [
        float(summary_df[(summary_df["region"] == "MT")
                         & (summary_df["metric"] == c)]["mean_per_gene"].iloc[0])
        for c in event_cols
    ]
    au_means = [
        float(summary_df[(summary_df["region"] == "autosomal")
                         & (summary_df["metric"] == c)]["mean_per_gene"].iloc[0])
        for c in event_cols
    ]

    ax = axes[0]
    ax.bar(x - width / 2, mt_totals, width, color="#d62728",
           alpha=0.8, label="MT")
    ax.bar(x + width / 2, au_totals, width, color="#1f77b4",
           alpha=0.8, label="Autosomal")
    for xi, v in zip(x - width / 2, mt_totals):
        ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width / 2, au_totals):
        ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([event_labels[c] for c in event_cols])
    ax.set_ylabel("Total count")
    ax.set_title(f"A. Total AS events\n(MT={mt_n} vs autosomal={au_n} multi-iso genes)")
    ax.legend()

    ax = axes[1]
    ax.bar(x - width / 2, mt_means, width, color="#d62728",
           alpha=0.8, label="MT")
    ax.bar(x + width / 2, au_means, width, color="#1f77b4",
           alpha=0.8, label="Autosomal")
    for xi, v in zip(x - width / 2, mt_means):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width / 2, au_means):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([event_labels[c] for c in event_cols])
    ax.set_ylabel("Events per multi-isoform gene")
    ax.set_title("B. Per-gene mean event count")
    ax.legend()

    fig.suptitle(
        "Alternative splicing events from R2C2 isoforms (RTS-filtered)",
        fontsize=12,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{args.out_plot_base}.{ext}"
        fig.savefig(out, dpi=150)
        print(f"Wrote plot: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
