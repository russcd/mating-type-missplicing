# Data files

This directory contains the processed numerical data underlying the manuscript figures.
Each file is mapped to the figure(s) it supports below, with a description of the columns
used for plotting.

## Mating-type region coordinates

`mating_type_regions.tsv` — defines the MT region for each species:
`species  strain  chromosome  start  end  n_genes  ...`. Used by the plotting
scripts to classify each intron / gene / isoform as MT or autosomal.

| Species | Chromosome | Start | End |
|---|---|---|---|
| *Bathycoccus prasinos* RCC1105 | NC_023995.1 | 232,352 | 625,114 |
| *Mantoniella commoda* RCC299 | NC_013038.1 | 269,071 | 1,816,973 |
| *Micromonas pusilla* CCMP1545 | NW_003315883.1 (a.k.a. scaffold_2) | 438,216 / 45,993 | 2,118,999 / 1,729,827 |
| *Ostreococcus tauri* RCC4221 | NC_014427.2 | 1 | 554,995 |

Note: for *M. pusilla*, the assembly used downstream of the rMATS pipeline reports
the MT-region scaffold as `scaffold_2:45,993–1,729,827`; the same biological interval
is referred to as `NW_003315883.1:438,216–2,118,999` in NCBI coordinates.

---

## Per-intron feature tables (Figures 1, 2A, S1, S2, S4)

Files: `Bprasinos.features.branchpoints.tsv`, `Mcommoda.features.branchpoints.tsv`,
`Mpusilla.features.branchpoints.tsv`, `Otauri.features.branchpoints.tsv`.

One row per annotated intron per species. Each intron is identified by its
`Header` field of the form `chromosome:start-end:transcript_id`. The same table
drives several figures depending on which columns are used:

| Column | Used for |
|---|---|
| `Header` | parses chromosome and intron coordinates for plotting position |
| `In_region` | 1 if intron is in the mating-type region, 0 otherwise — used to split MT vs autosomal in every figure below |
| `Expressed` | 1 if gene is expressed; introns with `Expressed=0` are dropped from retention plots |
| `rMATS_retention_ratio` | per-intron retention frequency, IJC/(IJC+SJC) — plotted as IR rate in Figure 1 and Figure S4 |
| `Length`, `GC_content` | plotted as MT-vs-autosomal distributions in Figure S1 |
| `FivePrimeContext`, `ThreePrimeContext` | nucleotide sequences flanking the 5'/3' splice sites — used to build the position weight matrices for Figure 2A (M. pusilla) and Figure S2 (all four species) |
| `Length`, `GC_content`, `First2`, `Last2`, `FifthBase`, `Distance_from_gene_start`, `Frame_preserving`, `Average_gene_depth`, `Gene_length`, `Branchpoint_present`, `Branchpoint_distance_to_3prime`, `TPM` | features fed into the random forest (Figure 2B–E feature importances) |


### Figure-specific mapping

- **Figure 1** — for each species: `Header → chromosome, midpoint` (x-axis),
  `rMATS_retention_ratio` (y-axis), colored by `In_region`. Filtered to
  `Expressed == 1`. Plotted by `analysis/plot_intron_retention.py`.
- **Figure 2A** — *M. pusilla* only: `FivePrimeContext` and `ThreePrimeContext`
  split by `In_region`, converted to PWMs by `analysis/compute_pwd.py`.
- **Figure S1** — for each species: `Length` and `GC_content` distributions split
  by `In_region`.
- **Figure S2** — same as Figure 2A but for all four species.
- **Figure S4** — same as Figure 1 but restricted to the MT-containing chromosome.
  Plotted by `analysis/plot_intron_retention_mt_chromosome.py`.

---

## Random forest outputs (Figures 2B–E)

Located in `rf/` (one level up). Generated from the per-intron feature tables
above by `analysis/fit_RF.py` (see `Snakefile`).

- `rf/{species}_MT_importances.tsv` — feature importances inside the MT region
  (one row per feature: `Feature`, `Importance`).
- `rf/{species}_MT_autosomal_importances.tsv` — same, autosomal regions.
- `rf/{species}_MT_metrics.tsv` and `rf/{species}_MT_autosomal_metrics.tsv` —
  baseline and final ROC AUC, accuracy, precision, recall, F1, and the number of
  features selected.
- `rf/{species}_*_partial_dependence_plots.pdf`, `*_feature_auc_*.pdf`,
  `*_confusion_matrices.pdf` — supporting diagnostic figures.

Figure 2B–E reports the Mpusilla importances; the same columns for the other
species are used in the corresponding panels of the supplement if shown.

---

## Isoform diversity table (Figures 3A and 3B)

`834.isoform.diversity.function.tsv` — one row per gene in the *M. pusilla*
Mandalorion isoform set.

| Column | Description |
|---|---|
| `Gene`, `Chrom`, `Start`, `End`, `Strand` | gene identifier and coordinates |
| `Introns` | number of introns in the gene |
| `Expression` | total isoform-level count |
| `Functional`, `NotFunctional` | counts of isoforms whose CDS is ≥ 80% / < 80% of the longest CDS in the gene (the "Functional" column is the numerator for the Proportion-Functional axis in Figure 3B) |
| `Num_Isoforms` | number of distinct isoforms detected — y-axis of Figure 3A |
| `Shannon`, `Evenness`, `Hill_q0`, `Hill_q1`, `Hill_q2` | diversity metrics (not plotted in main text) |
| `MatingTypeRegion` | 1 if gene is in MT region, 0 otherwise — used to color/split boxes |

- **Figure 3A**: `Num_Isoforms` vs `Introns` (capped at 5+), grouped by `MatingTypeRegion`.
- **Figure 3B**: `Functional / Expression` vs `Introns` (capped at 5+), grouped by `MatingTypeRegion`.
- Plotted by `analysis/plot_isoform_abundance_function.py`.

---

## Methylation and nucleosome features (Figure S3)

`Mpusilla.features.methylation.tsv` — one row per *M. pusilla* intron; same
`Header`/`In_region`/`Expressed` conventions as the branchpoints files.

| Column | Description |
|---|---|
| `Nucl_intron`, `Nucl_5p`, `Nucl_3p` | MNase-based nucleosome coverage in the intron and in the 50 bp 5'/3' flanking exons (top panel of Figure S3) |
| `CG_methylation`, `CG_5p`, `CG_3p` | mean CG methylation rate in the intron and the 50 bp 5'/3' flanking exons (middle panel) |
| `CG_count_intron`, `CG_count_5p`, `CG_count_3p` | number of CG dinucleotides in the intron / flanks (bottom panel) |
| `In_region` | used to split MT vs autosomal distributions; Mann–Whitney U tests are computed on the pooled values per panel |

Source data for the underlying MNase-seq and bisulfite-seq tracks is reference
[22] in the manuscript.

---

## Per-isoform internal-priming table (Figure S5)

`internal_priming_vs_IR_per_isoform.tsv` — one row per *M. pusilla*
Mandalorion isoform.

| Column | Description |
|---|---|
| `isoform` | isoform identifier |
| `chrom`, `start`, `end` | isoform genomic span |
| `gene_id` | parent gene (from the Mandalorion GTF) |
| `is_MT` | 1 if isoform's span lies inside the MT region |
| `has_IR` | True if isoform's SQANTI3 subcategory is `intron_retention` or `mono-exon_by_intron_retention` |
| `perc_A_downstream_TTS` | SQANTI3-computed percentage of A nucleotides in the 20 bp downstream of the isoform's 3' end |

Figure S5 is generated by aggregating this table to the gene level: for each of
the 164 MT genes that contain both intron-retained and non-intron-retained
isoforms, the mean `perc_A_downstream_TTS` is computed separately for IR and
non-IR isoforms; each gene contributes one paired observation. Plotted by
`analysis/internal_priming_vs_IR.py`.

---

## Other files

- `NMD_proteins.txt` — UPF1/UPF2/UPF3 orthologs across the four species
  (supplementary annotation; not a figure data file).
- `*.features.branchpoints.tsv` branchpoint-related columns
  (`Branchpoint_present`, `Branchpoint_distance_to_3prime`) are reported as
  features in the RF analysis (Figures 2B–E) but are not plotted directly.
