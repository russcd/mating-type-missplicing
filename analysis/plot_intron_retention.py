#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot genome-wide intron-weighted avg_retention_frequency for each species."
    )
    parser.add_argument(
        "-f", "--files", nargs="+", required=True,
        help="Input files in custom format."
    )
    parser.add_argument(
        "-m", "--min_introns", type=int, default=20,
        help="Minimum number of introns per window (default: 20)."
    )
    parser.add_argument(
        "-r", "--regions", default=None,
        help="Optional file specifying start/stop regions per species (tab/space-delimited)."
    )
    parser.add_argument(
        "-o", "--output", default="genome_retention_plot.pdf",
        help="Output plot file (default: genome_retention_plot.pdf)."
    )
    return parser.parse_args()

def clean_label(path):
    parts = os.path.normpath(path).split(os.sep)
    species = parts[-2] if len(parts) > 1 else os.path.basename(path)
    return species

def load_regions(region_file):
    if region_file is None:
        return {}
    df = pd.read_csv(region_file, sep=r"\s+", header=None, engine='python')
    regions = {}
    for _, row in df.iterrows():
        species = row[0]
        chrom = str(row[2])
        start = int(row[3])
        stop = int(row[4])
        regions.setdefault(species, {}).setdefault(chrom, []).append((start, stop))
    return regions

def load_file(file):
    df = pd.read_csv(file, sep="\t")
    df = df[df["Expressed"] == 1]
    if df.empty:
        return pd.DataFrame()
    def parse_header(header):
        parts = header.split(':')
        if len(parts) >= 3:
            chrom = parts[0]
            coords = parts[1].split('-')
            if len(coords) == 2:
                start = int(coords[0])
                stop = int(coords[1])
                return chrom, start, stop
        return None, None, None
    parsed_coords = df["Header"].apply(parse_header)
    df["chromosome"] = [x[0] for x in parsed_coords]
    df["intron_start"] = [x[1] for x in parsed_coords]
    df["intron_end"] = [x[2] for x in parsed_coords]
    df = df.dropna(subset=["chromosome", "intron_start", "intron_end"])
    df["avg_retention_frequency"] = pd.to_numeric(df["rMATS_retention_ratio"], errors="coerce").fillna(0)
    df["gene_midpoint"] = (df["intron_start"] + df["intron_end"]) / 2
    df["gene_start"] = df["intron_start"]
    df["gene_end"] = df["intron_end"]
    df["num_introns"] = 1
    return df

def make_windows(df, min_introns):
    windows_list = []
    for chrom, chrom_df in df.groupby("chromosome"):
        chrom_df = chrom_df.sort_values("gene_start")
        start = None
        intron_count = 0
        retention_values = []
        intron_weights = []
        for row in chrom_df.itertuples(index=False):
            if start is None:
                start = row.gene_start
            intron_count += row.num_introns
            retention_values.append(row.avg_retention_frequency * row.num_introns)
            intron_weights.append(row.num_introns)
            if intron_count >= min_introns:
                end = row.gene_end
                weighted_avg = np.sum(retention_values) / np.sum(intron_weights)
                windows_list.append({
                    "chromosome": row.chromosome,
                    "start": start,
                    "end": end,
                    "midpoint": (start + end) / 2,
                    "weighted_retention": weighted_avg
                })
                start = None
                intron_count = 0
                retention_values = []
                intron_weights = []
    return pd.DataFrame(windows_list)

def get_per_intron_data(df, regions_dict, species):
    if species not in regions_dict:
        return df["avg_retention_frequency"].tolist(), []
    in_region = pd.Series(False, index=df.index)
    for chrom, intervals in regions_dict[species].items():
        mask_chrom = df["chromosome"] == chrom
        for start, stop in intervals:
            mask = mask_chrom & df["gene_midpoint"].between(start, stop)
            in_region |= mask
    in_vals = df.loc[in_region, "avg_retention_frequency"].tolist()
    out_vals = df.loc[~in_region, "avg_retention_frequency"].tolist()
    return in_vals, out_vals

def main():
    args = parse_args()
    regions_dict = load_regions(args.regions)
    
    all_species_data = []
    all_raw_data = []
    
    for file in tqdm(args.files, desc="Processing species files"):
        species = clean_label(file)
        df = load_file(file)
        windows_df = make_windows(df, args.min_introns)
        if windows_df.empty:
            print(f"Skipping {file}: no valid data after filtering.")
            continue
        windows_df["species"] = species
        all_species_data.append(windows_df)
        all_raw_data.append((species, df))
    
    if not all_species_data:
        raise ValueError("No valid data to plot.")
    
    def get_genome_length(windows_df):
        return windows_df.groupby("chromosome")["end"].max().sum()
    
    species_with_lengths = [(get_genome_length(windows_df), windows_df, raw_df) 
                           for windows_df, (species, raw_df) in zip(all_species_data, all_raw_data)]
    species_with_lengths.sort(key=lambda x: x[0], reverse=True)
    all_species_data = [windows_df for _, windows_df, _ in species_with_lengths]
    all_raw_data = [(windows_df["species"].iloc[0], raw_df) for _, windows_df, raw_df in species_with_lengths]
    
    max_genome_span = 0
    for windows_df in all_species_data:
        chrom_sizes = windows_df.groupby("chromosome")["end"].max().sort_values(ascending=False)
        gap = 1e6 * 0.2
        total_span = chrom_sizes.sum() + (len(chrom_sizes) - 1) * gap
        max_genome_span = max(max_genome_span, total_span)
    
    fig = plt.figure(figsize=(14*0.5, 4*len(all_species_data)*0.5))
    
    for i, (windows_df, (species, raw_df)) in enumerate(
        tqdm(list(zip(all_species_data, all_raw_data)), desc="Plotting species")
    ):
        ax_box = plt.subplot2grid((len(all_species_data), 6), (i, 0), colspan=1)
        ax_genome = plt.subplot2grid((len(all_species_data), 6), (i, 1), colspan=5)
        
        in_region_data, outside_region_data = get_per_intron_data(raw_df, regions_dict, species)
        
        violin_data = []
        violin_colors = []
        violin_labels = []
        if outside_region_data:
            violin_data.append(outside_region_data)
            violin_colors.append('steelblue')
            violin_labels.append('Outside')
        if in_region_data:
            violin_data.append(in_region_data)
            violin_colors.append('orange')
            violin_labels.append('Mating Type\nRegion')
        
        if violin_data:
            vp = ax_box.violinplot(
                violin_data,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            for patch, color in zip(vp['bodies'], violin_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax_box.set_xticks([1 + idx for idx in range(len(violin_data))])
            ax_box.set_xticklabels(violin_labels, rotation=90)
        
        ax_box.set_ylim(-0.05, 1)
        ax_box.set_ylabel("")
        if i < len(all_species_data) - 1:
            ax_box.set_xticklabels([])
        
        species = windows_df["species"].iloc[0]
        chrom_sizes = windows_df.groupby("chromosome")["end"].max().sort_values(ascending=False)
        chrom_order = list(chrom_sizes.index)
        gap = 1e6 * 0.2
        chrom_offsets = {}
        offset = 0
        for chrom in chrom_order:
            chrom_offsets[chrom] = offset
            offset += chrom_sizes[chrom] + gap
        
        # vectorized scatter
        for chrom in tqdm(chrom_order, desc=f"Plotting {species}", leave=False):
            chrom_df = windows_df[windows_df["chromosome"] == chrom].sort_values("midpoint")
            if chrom_df.empty:
                continue
            mids = chrom_df["midpoint"].to_numpy()
            weights = chrom_df["weighted_retention"].to_numpy()
            x = mids + chrom_offsets[chrom]
            colors = np.full(len(mids), "steelblue", dtype=object)
            if species in regions_dict and chrom in regions_dict[species]:
                region_boundaries = regions_dict[species][chrom]
                in_mask = np.zeros(len(mids), dtype=bool)
                for start, stop in region_boundaries:
                    in_mask |= (mids >= start) & (mids <= stop)
                # adjacency: shift masks
                adj_mask = np.roll(in_mask, 1) | np.roll(in_mask, -1)
                colors[in_mask | adj_mask] = "orange"
            ax_genome.scatter(x, weights, c=colors, s=4)
        
        if species in regions_dict:
            for chrom_r, intervals in regions_dict[species].items():
                if chrom_r in chrom_offsets:
                    for start, stop in intervals:
                        ax_genome.axvline(start + chrom_offsets[chrom_r], color='black',
                                          linestyle='--', alpha=0.7, linewidth=1)
                        ax_genome.axvline(stop + chrom_offsets[chrom_r], color='black',
                                          linestyle='--', alpha=0.7, linewidth=1)
        
        tick_positions = []
        tick_labels = []
        for chrom in chrom_order:
            chrom_df = windows_df[windows_df["chromosome"] == chrom]
            if not chrom_df.empty:
                chrom_start = chrom_offsets[chrom]
                chrom_end = chrom_start + chrom_sizes[chrom]
                tick_positions.append((chrom_start + chrom_end) / 2)
                tick_labels.append("")
        ax_genome.set_xticks(tick_positions)
        if i < len(all_species_data) - 1:
            ax_genome.set_xticklabels([])
        else:
            ax_genome.set_xticklabels(tick_labels)
            ax_genome.set_xlabel("Chromosome")
        ax_genome.set_xlim(0, max_genome_span)
        ax_genome.text(0.99, 0.95, species, transform=ax_genome.transAxes, 
                      ha="right", va="top", fontsize=12, fontweight="bold")
        ax_genome.set_ylim(-0.05, 1)
        ax_genome.set_ylabel("")
        ax_genome.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    fig.text(0.02, 0.5, "Intron Retention", va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    fig.subplots_adjust(left=0.12)
    plt.savefig(args.output)
    plt.close()

if __name__ == "__main__":
    main()
