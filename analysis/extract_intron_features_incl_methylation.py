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
# Load CG methylation
# -----------------------
def load_cg_methylation(cg_file):
    """
    Expect GFF-like file with CG positions; column7 contains fraction methylation
    """
    meth = defaultdict(dict)
    with open(cg_file) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 7:
                chrom = parts[0]
                try:
                    pos = int(parts[1])
                    val = float(parts[6])
                except:
                    continue
                meth[chrom][pos] = val
    return meth

def mean_meth_in_window(chrom, start, end, meth_data):
    if (not meth_data) or (chrom not in meth_data):
        return np.nan, 0
    vals = [v for pos, v in meth_data[chrom].items() if start <= pos <= end]
    count = len(vals)
    if not vals:
        return np.nan, 0
    return float(np.mean(vals)), count

# -----------------------
# Load nucleosome coverage
# -----------------------
def load_nucleosome_coverage(nuc_file):
    nuc = defaultdict(dict)
    with open(nuc_file) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 7:
                chrom = parts[0]
                try:
                    pos = int(parts[1])
                    cov = float(parts[6])
                except:
                    continue
                nuc[chrom][pos] = cov
    return nuc

def mean_nuc_coverage(chrom, start, end, nuc_dict):
    if not nuc_dict or chrom not in nuc_dict:
        return np.nan
    vals = [v for pos, v in nuc_dict[chrom].items() if start <= pos <= end]
    if not vals:
        return np.nan
    return float(np.mean(vals))

# -----------------------
# Intron feature extraction
# -----------------------
def extract_intron_features(fasta_file, transcript_introns, rmats_events,
                            depth_arrays=None, min_depth=10, region=None,
                            window=10, verbose=False, cg_meth=None, nuc_dict=None, flank_size=50):
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

    if region:
        region_chrom, coords = region.split(":")
        region_start, region_end = map(int, coords.split("-"))

    rows = []
    for tx, data in tqdm(transcript_introns.items(), desc="Processing introns"):
        chrom = data['chrom']
        strand = data['strand']
        gene_id = data['gene_id']
        avg_depth = gene_depth.get(gene_id, None)
        chrom_seq = chrom_seqs.get(chrom, None)
        chrom_len = len(chrom_seq) if chrom_seq is not None else 0

        for intron in data['introns']:
            start, end = intron['start'], intron['end']
            seq = chrom_seq[start-1:end] if chrom_seq is not None else ""
            if strand == "-":
                seq = seq.reverse_complement()
            seq_str = str(seq).upper()
            length = len(seq_str)
            gc = round(compute_gc(seq_str),3) if length>0 else 0
            first2 = seq_str[:2] if length >=2 else seq_str
            last2 = seq_str[-2:] if length >=2 else seq_str

            in_region = 0
            if region and chrom==region_chrom and start<=region_end and end>=region_start:
                in_region =1

            rmats_ratio = match_intron_with_rmats(intron, gene_id, rmats_events)
            frame_preserving = is_frame_preserving(length)
            expressed = 1 if (avg_depth is None or avg_depth>=min_depth) else 0

            # intron CG methylation
            cg_meth_intron, cg_count_intron = mean_meth_in_window(chrom, start, end, cg_meth) if cg_meth else (np.nan,0)

            # Flanking sequences
            flank5_start = max(1, start - flank_size)
            flank5_end = start -1
            if flank5_end<flank5_start:
                flank5_seq_str = ""
            else:
                f5_seq = chrom_seq[flank5_start-1:flank5_end] if chrom_seq else ""
                if strand=="-":
                    f5_seq = f5_seq.reverse_complement()
                flank5_seq_str = str(f5_seq).upper()

            flank3_start = end+1
            flank3_end = min(chrom_len, end+flank_size)
            if flank3_start>flank3_end:
                flank3_seq_str = ""
            else:
                f3_seq = chrom_seq[flank3_start-1:flank3_end] if chrom_seq else ""
                if strand=="-":
                    f3_seq = f3_seq.reverse_complement()
                flank3_seq_str = str(f3_seq).upper()

            # GC content flanks
            gc_5p = round(compute_gc(flank5_seq_str),3) if flank5_seq_str else np.nan
            gc_3p = round(compute_gc(flank3_seq_str),3) if flank3_seq_str else np.nan

            # CG methylation flanks
            cg_5p, cg_count_5p = mean_meth_in_window(chrom, flank5_start, flank5_end, cg_meth) if cg_meth else (np.nan,0)
            cg_3p, cg_count_3p = mean_meth_in_window(chrom, flank3_start, flank3_end, cg_meth) if cg_meth else (np.nan,0)

            # nucleosome coverage
            nuc_intron = mean_nuc_coverage(chrom, start, end, nuc_dict)
            nuc_5p = mean_nuc_coverage(chrom, flank5_start, flank5_end, nuc_dict)
            nuc_3p = mean_nuc_coverage(chrom, flank3_start, flank3_end, nuc_dict)

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
                'CG_methylation': round(cg_meth_intron,3) if not pd.isna(cg_meth_intron) else np.nan,
                'CG_count_intron': cg_count_intron,
                'CG_5p': round(cg_5p,3) if not pd.isna(cg_5p) else np.nan,
                'CG_count_5p': cg_count_5p,
                'CG_3p': round(cg_3p,3) if not pd.isna(cg_3p) else np.nan,
                'CG_count_3p': cg_count_3p,
                'GC_5p': gc_5p,
                'GC_3p': gc_3p,
                'Nucl_intron': round(nuc_intron,3) if not pd.isna(nuc_intron) else np.nan,
                'Nucl_5p': round(nuc_5p,3) if not pd.isna(nuc_5p) else np.nan,
                'Nucl_3p': round(nuc_3p,3) if not pd.isna(nuc_3p) else np.nan,
                'rMATS_retention_ratio': round(rmats_ratio,3)
            })
    return rows

# -----------------------
# featureCounts parsing and TPM
# -----------------------
def parse_featurecounts_tpm(featurecounts_file):
    df = pd.read_csv(featurecounts_file, sep="\t", comment="#")
    count_col = df.columns[6]
    df = df[['Geneid','Length',count_col]].copy()
    df.rename(columns={'Geneid':'gene_id','Length':'gene_length',count_col:'counts'}, inplace=True)
    df['RPK'] = df['counts']/(df['gene_length']/1000)
    total_rpk = df['RPK'].sum()
    df['TPM'] = (df['RPK']/total_rpk)*1e6
    return df[['gene_id','TPM']]

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Annotate introns with features, rMATS retention, coverage, expression (TPM), CG methylation, and nucleosome coverage")
    parser.add_argument("gtf_file", help="GTF file")
    parser.add_argument("fasta_file", help="Reference FASTA of chromosomes")
    parser.add_argument("rmats_file", help="rMATS intron retention output")
    parser.add_argument("output_tsv", help="Output TSV file")
    parser.add_argument("--bam_files", nargs="+", help="One or more BAM files")
    parser.add_argument("--min_depth", type=int, default=10, help="Minimum depth at junctions to consider intron expressed")
    parser.add_argument("--region", type=str, help="Optional region chrom:start-stop")
    parser.add_argument("--window", type=int, default=10, help="Window size in bases for junction coverage")
    parser.add_argument("--featurecounts", help="featureCounts output file (TSV)")
    parser.add_argument("--cg_methylation", help="Tab-delimited CG methylation file (GFF-like col7=frac methylation)")
    parser.add_argument("--nuc_file", help="Nucleosome coverage file (GFF-like col7=coverage)")
    parser.add_argument("--flank_size", type=int, default=50, help="Flank size in bp for methylation, GC, nucleosome")
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
            for k,v in da.items():
                if k in depth_arrays:
                    depth_arrays[k] += v
                else:
                    depth_arrays[k] = v

    cg_meth = load_cg_methylation(args.cg_methylation) if args.cg_methylation else None
    nuc_dict = load_nucleosome_coverage(args.nuc_file) if args.nuc_file else None

    rows = extract_intron_features(
        args.fasta_file, transcript_introns, rmats_events,
        depth_arrays=depth_arrays, min_depth=args.min_depth,
        region=args.region, window=args.window,
        verbose=args.verbose, cg_meth=cg_meth, nuc_dict=nuc_dict,
        flank_size=args.flank_size
    )

    df = pd.DataFrame(rows)
    if args.featurecounts:
        tpm_df = parse_featurecounts_tpm(args.featurecounts)
        df = df.merge(tpm_df, on='gene_id', how='left')

    df.to_csv(args.output_tsv, sep='\t', index=False)
    if args.verbose:
        print(f"[INFO] Written output to {args.output_tsv}", flush=True)

if __name__=="__main__":
    main()
