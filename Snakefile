"""Run random forest (fit_RF.py) per species for mating_type and autosomal regions."""

SPECIES = glob_wildcards("data/{species}.features.branchpoints.tsv").species
REGION_SUFFIXES = ["MT", "MT_autosomal"]  # MT = mating_type, MT_autosomal = autosomal

rule all:
    input:
        expand(
            "rf/{species}_{suffix}_importances.tsv",
            species=SPECIES,
            suffix=REGION_SUFFIXES,
        ),
        expand(
            "rf/{species}_{suffix}_metrics.tsv",
            species=SPECIES,
            suffix=REGION_SUFFIXES,
        )

wildcard_constraints:
    suffix="MT|MT_autosomal",

rule run_rf:
    input:
        "data/{species}.features.branchpoints.tsv"
    output:
        imp="rf/{species}_{suffix}_importances.tsv",
        metrics="rf/{species}_{suffix}_metrics.tsv",
        elbow="rf/{species}_{suffix}_feature_auc_elbow_curve.pdf",
        scatter="rf/{species}_{suffix}_feature_auc_scatter.pdf",
        pdp="rf/{species}_{suffix}_partial_dependence_plots.pdf",
    params:
        region_type=lambda w: "autosomal" if w.suffix == "MT_autosomal" else "mating_type",
    log:
        "logs/rf_{species}_{suffix}.log",
    shell:
        """
        mkdir -p rf logs && python analysis/fit_RF.py \
        --input {input} --output rf/{wildcards.species}_MT --region-type {params.region_type} \
        2>&1 | tee {log}
        """
