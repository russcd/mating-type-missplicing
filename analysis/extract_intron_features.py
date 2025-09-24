#!/usr/bin/env python3
import argparse
import re
import subprocess
from collections import defaultdict
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

# -----------------------
# GTF parsing and intron extraction
# -----------------------
def parse_gtf_attributes(attr_string):
    attributes = {}
    parts = re.findall(r'(\w+)\s+"([^"]+)"', attr_string)
    for key, value in parts:
        attributes[key] = value
    return attributes

def extract_introns_from_gtf(gtf_file):
    transcripts = defaultdict(list)
    chroms, strands, gene_ids = {}, {}, {}
    gene_positions = {}

    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            chrom = fields[0]
            start, end = int(fields[3]), int(fields[4])
            strand = fields[6]
            attrs = parse_gtf_attributes(fields[8])
            if 'transcript_id' in attrs and 'gene_id' in attrs:
                tx, gene_id = attrs['transcript_id'], attrs['gene_id']
                transcripts[tx].append((start, end))
                chroms[tx] = chrom
                strands[tx] = strand
                gene_ids[tx] = gene_id
                if gene_id not in gene_positions:
                    gene_positions[gene_id] = {'strand': strand, 'positions': []}
                gene_positions[gene_id]['positions'].extend([start, end])

    gene_starts = {}
    gene_lengths = {}
    for gene_id, data in gene_positions.items():
        positions = data['positions']
        strand = data['strand']
        gene_starts[gene_id] = min(positions) if strand == '+' else max(positions)
        gene_lengths[gene_id] = max(positions) - min(positions) + 1

    transcript_introns = {}
    for tx, cds_list in transcripts.items():
        cds_list = sorted(cds_list, key=lambda x: x[0])
        introns = []
        for i in range(len(cds_list)-1):
            intron_start = cds_list[i][1]+1
            intron_end = cds_list[i+1][0]-1
            if intron_start <= intron_end:
                introns.append({
                    'start': intron_start,
                    'end': intron_end,
                    'upstream_exon': cds_list[i],
                    'downstream_exon': cds_list[i+1]
                })
        transcript_introns[tx] = {
            'introns': introns,
            'chrom': chroms[tx],
            'strand': strands[tx],
            'gene_id': gene_ids[tx],
            'gene_start': gene_starts[gene_ids[tx]],
            'gene_length': gene_lengths[gene_ids[tx]],
            'all_cds': cds_list
        }
    return transcript_introns

# -----------------------
# rMATS parsing
# -----------------------
def parse_rmats_output(rmats_file):
    df = pd.read_csv(rmats_file, sep='\t')
    retention_events = defaultdict(list)
    for _, row in df.iterrows():
        gene_id = str(row['GeneID']).strip('"')
        try:
            ri_start = int(row['riExonStart_0base']) + 1
            ri_end = int(row['riExonEnd'])
        except:
            continue
        def mean_int(x):
            if pd.isna(x):
                return 0
            if isinstance(x, str):
                parts = x.split(',')
                return sum([int(p) for p in parts])/len(parts)
            return int(x)
        ijc = mean_int(row['IJC_SAMPLE_1'])
        sjc = mean_int(row['SJC_SAMPLE_1'])
        total = ijc + sjc
        retention_ratio = ijc / total if total > 0 else 0
        retention_events[gene_id].append({'start': ri_start, 'end': ri_end, 'ratio': retention_ratio})
    return retention_events

def match_intron_with_rmats(intron, gene_id, rmats_events):
    if gene_id not in rmats_events:
        return 0
    for ev in rmats_events[gene_id]:
        if intron['start'] <= ev['end'] and intron['end'] >= ev['start']:
            return ev['ratio']
    return 0

# -----------------------
# Depth computation
# -----------------------
def load_bam_depths(bam_file, verbose=False):
    depth_arrays = {}
    if verbose:
        print(f"[INFO] Processing BAM {bam_file} with samtools depth...", flush=True)
    cmd = ["samtools", "depth", "-aa", bam_file]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    current_chrom = None
    depths = []
    for line in tqdm(proc.stdout, desc="Reading depth", unit="lines"):
        chrom, pos, cov = line.strip().split('\t')
        pos, cov = int(pos)-1, int(cov)
        if current_chrom != chrom:
            if current_chrom is not None:
                depth_arrays[current_chrom] = np.array(depths, dtype=int)
            current_chrom = chrom
            depths = []
        depths.append(cov)
    if current_chrom is not None:
        depth_arrays[current_chrom] = np.array(depths, dtype=int)
    return depth_arrays

def compute_gene_average_depth(chrom, cds_list, depth_array, window=10):
    total = 0
    bases = 0
    chrom_len = len(depth_array)
    for start, end in cds_list:
        u_start = max(0, end - window)
        u_end = min(end, chrom_len)  # clip to chromosome length
        if u_start >= u_end:
            continue
        total += depth_array[u_start:u_end].sum()
        bases += u_end - u_start
    return total / bases if bases > 0 else 0

# -----------------------
# Frame and sequence analysis
# -----------------------
def compute_gc(seq):
    return (seq.count("G") + seq.count("C")) / len(seq) if len(seq) > 0 else 0

def calculate_distance_from_gene_start(intron_start, gene_start, strand):
    return intron_start - gene_start if strand == '+' else gene_start - intron_start

def is_frame_preserving(length):
    return 1 if length % 3 == 0 else 0

# -----------------------
# Intron feature extraction
# -----------------------
def extract_intron_features(fasta_file, transcript_introns, rmats_events, depth_arrays=None, min_depth=10, region=None, window=10, verbose=False):
    chrom_seqs = {r.id: r.seq for r in SeqIO.parse(fasta_file, "fasta")}
    gene_depth = {}
    if depth_arrays:
        if verbose:
            print("[INFO] Computing gene average depths...", flush=True)
        for tx, data in tqdm(transcript_introns.items(), desc="Gene depth"):
            chrom = data['chrom']
            gene_id = data['gene_id']
            if chrom in depth_arrays:
                gene_depth[gene_id] = compute_gene_average_depth(chrom, data['all_cds'], depth_arrays[chrom], window)
            else:
                gene_depth[gene_id] = None

    # Parse region
    if region:
        region_chrom, coords = region.split(":")
        region_start, region_end = map(int, coords.split("-"))

    rows = []
    for tx, data in tqdm(transcript_introns.items(), desc="Processing introns"):
        chrom = data['chrom']
        strand = data['strand']
        gene_id = data['gene_id']
        avg_depth = gene_depth.get(gene_id, None)
        for intron in data['introns']:
            start, end = intron['start'], intron['end']
            seq = chrom_seqs[chrom][start-1:end]
            if strand == "-":
                seq = seq.reverse_complement()
            seq_str = str(seq).upper()
            length = len(seq_str)
            gc = round(compute_gc(seq_str), 3)
            first2 = seq_str[:2] if length >=2 else seq_str
            last2 = seq_str[-2:] if length >=2 else seq_str
            in_region = 0
            if region and chrom == region_chrom and start <= region_end and end >= region_start:
                in_region = 1
            rmats_ratio = match_intron_with_rmats(intron, gene_id, rmats_events)
            frame_preserving = is_frame_preserving(length)
            expressed = 1 if (avg_depth is None or avg_depth >= min_depth) else 0
            rows.append({
                'Header': f"{chrom}:{start}-{end}:{tx}",
                'gene_id': gene_id,
                'Length': length,
                'GC_content': gc,
                'First2': first2,
                'Last2': last2,
                'In_region': in_region,
                'Distance_from_gene_start': calculate_distance_from_gene_start(start, data['gene_start'], strand),
                'Frame_preserving': frame_preserving,
                'Average_gene_depth': avg_depth,
                'Gene_length': data['gene_length'],
                'Expressed': expressed,
                'rMATS_retention_ratio': round(rmats_ratio,3)
            })
    return rows

# -----------------------
# featureCounts parsing and TPM
# -----------------------
def parse_featurecounts_tpm(featurecounts_file):
    """
    Parse featureCounts output and compute TPM per gene.
    """
    df = pd.read_csv(featurecounts_file, sep="\t", comment="#")
    # assume first count column is the expression
    count_col = df.columns[6]
    df = df[['Geneid', 'Length', count_col]].copy()
    df.rename(columns={'Geneid': 'gene_id', 'Length': 'gene_length', count_col: 'counts'}, inplace=True)

    # compute Reads Per Kilobase (RPK)
    df['RPK'] = df['counts'] / (df['gene_length'] / 1000)
    total_rpk = df['RPK'].sum()
    df['TPM'] = (df['RPK'] / total_rpk) * 1e6
    return df[['gene_id', 'TPM']]

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Annotate introns with features, rMATS retention, coverage, and expression (TPM)")
    parser.add_argument("gtf_file", help="GTF file")
    parser.add_argument("fasta_file", help="Reference FASTA of chromosomes")
    parser.add_argument("rmats_file", help="rMATS intron retention output")
    parser.add_argument("output_tsv", help="Output TSV file")
    parser.add_argument("--bam_files", nargs="+", help="One or more BAM files")
    parser.add_argument("--min_depth", type=int, default=10, help="Minimum depth at junctions to consider intron expressed")
    parser.add_argument("--region", type=str, help="Optional region chrom:start-stop")
    parser.add_argument("--window", type=int, default=10, help="Window size in bases for junction coverage")
    parser.add_argument("--featurecounts", help="featureCounts output file (TSV)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    args = parser.parse_args()

    transcript_introns = extract_introns_from_gtf(args.gtf_file)
    if args.verbose:
        print(f"[INFO] Extracted {sum(len(d['introns']) for d in transcript_introns.values())} introns", flush=True)

    rmats_events = parse_rmats_output(args.rmats_file)

    depth_arrays = {}
    if args.bam_files:
        for bam in args.bam_files:
            da = load_bam_depths(bam, verbose=args.verbose)
            for k, v in da.items():
                if k in depth_arrays:
                    depth_arrays[k] += v
                else:
                    depth_arrays[k] = v

    rows = extract_intron_features(
        args.fasta_file, transcript_introns, rmats_events,
        depth_arrays=depth_arrays if depth_arrays else None,
        min_depth=args.min_depth,
        region=args.region,
        window=args.window,
        verbose=args.verbose
    )

    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c != 'rMATS_retention_ratio'] + ['rMATS_retention_ratio']
    df = df[cols]

    # integrate TPM if provided
    if args.featurecounts:
        tpm_df = parse_featurecounts_tpm(args.featurecounts)
        df = df.merge(tpm_df, on="gene_id", how="left")

    df.to_csv(args.output_tsv, sep='\t', index=False)
    if args.verbose:
        print(f"[INFO] Written {len(rows)} introns to {args.output_tsv}", flush=True)

if __name__ == "__main__":
    main()
