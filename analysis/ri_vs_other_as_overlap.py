#!/usr/bin/env python3
"""
Compare RI (retained intron) events against all other rMATS AS event types
(SE, A3SS, A5SS, MXE) to quantify coordinate overlap and assess whether
overlapping events are redundant with RI calls.

For each species × AS-event-type combination, reports:
  - Total event counts
  - Number/percent of AS events whose coordinates overlap an RI event
  - Number/percent of AS events fully contained within an RI event
  - Mean inclusion ratios for overlapping pairs
  - Pearson correlation between RI and AS inclusion ratios
  - Fraction of high-RI events (ratio > 0.8) with high AS inclusion (> 0.5)

Usage:
    python ri_vs_other_as_overlap.py [--outfile results.tsv]
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np


BASE = "/scratch2/russ/introner/splicing_fails/Mammialles"
SPECIES = ["Bprasinos", "Mcommoda", "Mpusilla", "Otauri"]
EVENT_TYPES = ["SE", "A3SS", "A5SS", "MXE"]

MT_CHROMS = {
    "Bprasinos": "chrNC_023995.1",
    "Mcommoda": "chrNC_013038.1",
    "Mpusilla": "chrNW_003315883.1",
    "Otauri": "chrNC_014427.2",
}


def mean_int(x):
    """Average comma-separated replicate counts (rMATS format)."""
    if not x or x == "NA":
        return 0.0
    parts = x.split(",")
    return np.mean([int(p) for p in parts])


def load_ri_events(species):
    """Load RI events, return dict keyed by GeneID."""
    path = f"{BASE}/{species}/rmats_out/RI.MATS.JC.txt"
    events = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ijc = mean_int(row["IJC_SAMPLE_1"])
            sjc = mean_int(row["SJC_SAMPLE_1"])
            total = ijc + sjc
            events[row["GeneID"]].append({
                "chr": row["chr"],
                "start": int(row["riExonStart_0base"]),
                "end": int(row["riExonEnd"]),
                "ratio": ijc / total if total > 0 else 0,
            })
    return events


def get_as_span(row, event_type):
    """
    Return (span_start, span_end) covering the 'alternative' region
    for a given AS event type.

    SE:   the skipped exon itself (exonStart_0base – exonEnd)
    A3SS: min/max of long and short exon boundaries
    A5SS: min/max of long and short exon boundaries
    MXE:  union of the two mutually-exclusive exons
    """
    if event_type == "SE":
        return int(row["exonStart_0base"]), int(row["exonEnd"])
    elif event_type in ("A3SS", "A5SS"):
        coords = [
            int(row["longExonStart_0base"]), int(row["longExonEnd"]),
            int(row["shortES"]), int(row["shortEE"]),
        ]
        return min(coords), max(coords)
    elif event_type == "MXE":
        coords = [
            int(row["1stExonStart_0base"]), int(row["1stExonEnd"]),
            int(row["2ndExonStart_0base"]), int(row["2ndExonEnd"]),
        ]
        return min(coords), max(coords)
    else:
        raise ValueError(f"Unknown event type: {event_type}")


def load_as_events(species, event_type):
    """Load AS events, return dict keyed by GeneID."""
    path = f"{BASE}/{species}/rmats_out/{event_type}.MATS.JC.txt"
    events = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ijc = mean_int(row["IJC_SAMPLE_1"])
            sjc = mean_int(row["SJC_SAMPLE_1"])
            total = ijc + sjc
            span_start, span_end = get_as_span(row, event_type)
            events[row["GeneID"]].append({
                "chr": row["chr"],
                "span_start": span_start,
                "span_end": span_end,
                "ratio": ijc / total if total > 0 else 0,
            })
    return events


def overlaps(a_start, a_end, b_start, b_end):
    return a_start <= b_end and a_end >= b_start


def contained(inner_start, inner_end, outer_start, outer_end):
    return outer_start <= inner_start and outer_end >= inner_end


def analyse_overlap(ri_events, as_events):
    """
    For each AS event in a shared gene, check overlap/containment with RI
    events. Collect paired ratios for overlapping events.
    """
    shared_genes = set(ri_events) & set(as_events)
    total_as = sum(len(v) for v in as_events.values())
    total_ri = sum(len(v) for v in ri_events.values())

    n_overlap = 0
    n_contained = 0
    ri_ratios = []
    as_ratios = []

    for gene in shared_genes:
        for a in as_events[gene]:
            best_ri = None
            is_overlapping = False
            is_contained = False
            for ri in ri_events[gene]:
                if a["chr"] != ri["chr"]:
                    continue
                if overlaps(a["span_start"], a["span_end"], ri["start"], ri["end"]):
                    is_overlapping = True
                    if contained(a["span_start"], a["span_end"], ri["start"], ri["end"]):
                        is_contained = True
                    best_ri = ri
                    break  # take first overlapping RI
            if is_overlapping:
                n_overlap += 1
                ri_ratios.append(best_ri["ratio"])
                as_ratios.append(a["ratio"])
            if is_contained:
                n_contained += 1

    ri_ratios = np.array(ri_ratios)
    as_ratios = np.array(as_ratios)

    # Correlation
    if len(ri_ratios) >= 2 and np.std(ri_ratios) > 0 and np.std(as_ratios) > 0:
        corr = np.corrcoef(ri_ratios, as_ratios)[0, 1]
    else:
        corr = np.nan

    # High-RI concordance
    ri_high_mask = ri_ratios > 0.8
    n_ri_high = int(ri_high_mask.sum())
    if n_ri_high > 0:
        n_as_high_when_ri_high = int((as_ratios[ri_high_mask] > 0.5).sum())
        pct_concordant = n_as_high_when_ri_high / n_ri_high * 100
    else:
        n_as_high_when_ri_high = 0
        pct_concordant = np.nan

    return {
        "total_ri": total_ri,
        "total_as": total_as,
        "shared_genes": len(shared_genes),
        "n_overlap": n_overlap,
        "pct_overlap": n_overlap / total_as * 100 if total_as > 0 else 0,
        "n_contained": n_contained,
        "pct_contained": n_contained / total_as * 100 if total_as > 0 else 0,
        "mean_ri_ratio": float(np.mean(ri_ratios)) if len(ri_ratios) > 0 else np.nan,
        "mean_as_ratio": float(np.mean(as_ratios)) if len(as_ratios) > 0 else np.nan,
        "pearson_r": corr,
        "n_ri_high": n_ri_high,
        "n_as_high_when_ri_high": n_as_high_when_ri_high,
        "pct_concordant": pct_concordant,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outfile", default="analysis/ri_vs_other_as_overlap.tsv",
                        help="Output TSV path (default: analysis/ri_vs_other_as_overlap.tsv)")
    args = parser.parse_args()

    columns = [
        "species", "event_type",
        "total_ri", "total_as", "shared_genes",
        "n_overlap", "pct_overlap",
        "n_contained", "pct_contained",
        "mean_ri_ratio", "mean_as_ratio",
        "pearson_r",
        "n_ri_high", "n_as_high_when_ri_high", "pct_concordant",
    ]

    rows = []
    for sp in SPECIES:
        ri_events = load_ri_events(sp)
        for et in EVENT_TYPES:
            as_events = load_as_events(sp, et)
            result = analyse_overlap(ri_events, as_events)
            result["species"] = sp
            result["event_type"] = et
            rows.append(result)

    # Write TSV
    with open(args.outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Round floats
            for k in row:
                if isinstance(row[k], float):
                    row[k] = f"{row[k]:.3f}" if not np.isnan(row[k]) else "NA"
            writer.writerow(row)

    # Also print a readable summary to stdout
    print(f"Results written to {args.outfile}\n")
    print(f"{'Species':<12} {'Event':<6} {'Total':>6} {'Overlap':>8} {'%Ovlp':>7} "
          f"{'Contained':>10} {'%Cont':>7} {'RI_ratio':>9} {'AS_ratio':>9} "
          f"{'Pearson_r':>10} {'RI>0.8→AS>0.5':>15}")
    print("-" * 110)
    for r in rows:
        print(f"{r['species']:<12} {r['event_type']:<6} {r['total_as']:>6} "
              f"{r['n_overlap']:>8} {r['pct_overlap']:>7} "
              f"{r['n_contained']:>10} {r['pct_contained']:>7} "
              f"{r['mean_ri_ratio']:>9} {r['mean_as_ratio']:>9} "
              f"{r['pearson_r']:>10} "
              f"{r['n_as_high_when_ri_high']}/{r['n_ri_high']:>14}")


if __name__ == "__main__":
    main()
