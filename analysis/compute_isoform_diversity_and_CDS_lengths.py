#!/usr/bin/env python3
"""
Compute isoform diversity metrics per gene (Shannon, Evenness, Hill numbers)
and annotate genes with coordinates + intron counts parsed from a GTF file.

Adds functionality to classify isoforms into Functional vs NotFunctional
based on CDS length relative to longest CDS isoform in the same gene
(using an additional CDS GFF3 file with GTF-like attributes).
"""

import argparse
import math
from collections import defaultdict


def shannon_diversity(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    H = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            H -= p * math.log(p)
    return H


def hill_numbers(counts):
    total = sum(counts)
    if total == 0:
        return 0, 0.0, 0.0
    q0 = sum(1 for c in counts if c > 0)
    H = shannon_diversity(counts)
    q1 = math.exp(H)
    D = sum((c / total) ** 2 for c in counts if c > 0)
    q2 = 1 / D if D > 0 else 0.0
    return q0, q1, q2


def parse_attributes(attr_str):
    """Parse GTF/GFF3-like attributes into dict."""
    attrs = {}
    for part in attr_str.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        if " " in part:
            key, val = part.split(" ", 1)
            attrs[key] = val.strip('"')
    return attrs


def parse_gtf(gtf_file):
    """Parse GTF file to get gene coordinates and intron counts per gene."""
    genes = {}
    transcripts_exons = defaultdict(list)
    transcript_to_gene = {}
    transcript_chrom = {}

    with open(gtf_file) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            chrom, source, feature, start_s, end_s, score, strand, frame, attrs_str = fields
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            attrs = parse_attributes(attrs_str)

            gene_id = attrs.get("gene_id")
            transcript_id = attrs.get("transcript_id")

            if feature == "gene" and gene_id:
                genes[gene_id] = {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "introns": 0,
                }
            elif feature in ("transcript", "mRNA") and transcript_id and gene_id:
                transcript_to_gene[transcript_id] = gene_id
                if gene_id not in genes:
                    genes[gene_id] = {
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "introns": 0,
                    }
            elif feature == "exon" and transcript_id:
                transcripts_exons[transcript_id].append((start, end))
                transcript_chrom[transcript_id] = chrom
                if gene_id and gene_id not in genes:
                    genes[gene_id] = {
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "strand": attrs.get("strand", "."),
                        "introns": 0,
                    }

    # Compute intron counts
    for tid, exons in transcripts_exons.items():
        if not exons:
            continue
        exons_sorted = sorted(exons, key=lambda x: (x[0], x[1]))
        introns = max(0, len(exons_sorted) - 1)
        gene_id = transcript_to_gene.get(tid)
        if gene_id and gene_id in genes:
            genes[gene_id]["introns"] = max(genes[gene_id]["introns"], introns)
            g = genes[gene_id]
            g["start"] = min(g["start"], exons_sorted[0][0])
            g["end"] = max(g["end"], exons_sorted[-1][1])

    return genes


def parse_cds_gff3(gff_file):
    """Parse CDS features from a GFF3/GTF-like file → transcript_id -> CDS length."""
    cds_lengths = defaultdict(int)

    with open(gff_file) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature != "CDS":
                continue
            start, end = int(fields[3]), int(fields[4])
            attrs = parse_attributes(fields[8])
            tid = attrs.get("transcript_id")
            if tid:
                cds_lengths[tid] += end - start + 1

    return cds_lengths


def main():
    parser = argparse.ArgumentParser(description="Compute isoform diversity per gene using a GTF and CDS GFF3.")
    parser.add_argument("input_file", help="Tab-delimited isoform counts file (header: Isoform Gene <samples...>)")
    parser.add_argument("--gtf", required=True, help="GTF file with transcript/exon annotations")
    parser.add_argument("--cds_gff3", required=True, help="GFF3 file with CDS annotations (per transcript, GTF-like)")
    parser.add_argument("--cds_ratio", type=float, default=0.5, help="Threshold ratio for functional CDS (default=0.5)")
    parser.add_argument("--logbase", type=float, default=math.e, help="Logarithm base (default: e)")
    args = parser.parse_args()

    # Parse inputs
    gtf_genes = parse_gtf(args.gtf)
    cds_lengths = parse_cds_gff3(args.cds_gff3)

    # Collect isoform counts
    gene_isoform_counts = defaultdict(list)
    gene_isoform_data = defaultdict(list)  # (isoform, counts, cds_len)
    all_genes_in_counts = set()

    with open(args.input_file) as f:
        header = f.readline().strip().split()
        for line in f:
            if not line.strip():
                continue
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            isoform, gene = fields[0], str(fields[1])
            if gene.startswith("Locus"):
                gene = gene[len("Locus"):]
            try:
                counts = list(map(int, fields[2:]))
            except ValueError:
                counts = []
            total_isoform_count = sum(counts)
            cds_len = cds_lengths.get(isoform, 0)
            gene_isoform_counts[gene].append(total_isoform_count)
            gene_isoform_data[gene].append((isoform, total_isoform_count, cds_len))
            all_genes_in_counts.add(gene)

    # Prepare output
    header_cols = [
        "Gene", "Chrom", "Start", "End", "Strand", "Introns",
        "Expression",
        "Functional", "NotFunctional",
        "Functional_Transcripts", "NotFunctional_Transcripts",
        "Shannon", "Evenness", "Hill_q0", "Hill_q1", "Hill_q2",
        "Num_Isoforms",
        "MatingTypeRegion"   # <-- new column
    ]
    print("\t".join(header_cols))

    for gene, counts in gene_isoform_counts.items():
        if gene not in gtf_genes:
            continue
        g = gtf_genes[gene]

        total_expression = sum(counts)
        longest_cds = max((cds_len for _, _, cds_len in gene_isoform_data[gene]), default=0)

        functional = 0
        notfunctional = 0
        functional_transcripts = 0
        notfunctional_transcripts = 0

        for iso, cnt, cds_len in gene_isoform_data[gene]:
            if longest_cds > 0 and cds_len / longest_cds >= args.cds_ratio:
                functional += cnt
                functional_transcripts += 1
            else:
                notfunctional += cnt
                notfunctional_transcripts += 1

        H = shannon_diversity(counts)
        if args.logbase != math.e:
            H = H / math.log(args.logbase)
        S = sum(1 for c in counts if c > 0)
        evenness = H / math.log(S, args.logbase) if S > 1 else 0.0
        q0, q1, q2 = hill_numbers(counts)
        num_isoforms = len(counts)

        # Annotate mating-type region
        mating_type_region = int(
            (g["chrom"] == "scaffold_2") and
            (g["start"] >= 45993) and
            (g["end"] <= 1729827)
        )

        row = [
            gene, g["chrom"], str(g["start"]), str(g["end"]), g["strand"],
            str(g.get("introns", 0)), str(total_expression),
            str(functional), str(notfunctional),
            str(functional_transcripts), str(notfunctional_transcripts),
            f"{H:.4f}", f"{evenness:.4f}",
            f"{q0}", f"{q1:.4f}", f"{q2:.4f}",
            str(num_isoforms),
            str(mating_type_region)  # <-- new value
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()
