from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig
from .util import atomic_write_json, sha256_file, utc_now


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "not estimable"
    return f"{float(value):.{digits}f}"


def _ci(row: pd.Series | dict, digits: int = 3) -> str:
    return f"{_fmt(row['estimate'], digits)} (95% CI, {_fmt(row['ci_low'], digits)} to {_fmt(row['ci_high'], digits)})"


def _clean_text(value: object) -> str:
    return str(value).replace("�", "-").replace("\\-", "-").strip().rstrip(".")


def _reference_list(reference_path: Path) -> str:
    references = pd.read_csv(reference_path).sort_values("id")
    lines = []
    for number, row in enumerate(references.itertuples(index=False), 1):
        authors = _clean_text(row.authors).replace(";", ",")
        title = _clean_text(row.title)
        journal = _clean_text(row.journal)
        doi = f" https://doi.org/{row.doi}." if isinstance(row.doi, str) and row.doi else ""
        lines.append(f"{number}. {authors}. {title}. *{journal}*. {int(row.year)}.{doi}")
    return "\n".join(lines)


def _word_count(text: str) -> int:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[#*_`|]", " ", text)
    return len(re.findall(r"\b[\w'-]+\b", text))


def _load_inputs(output_root: Path, manuscript_dir: Path):
    artifacts = output_root / "artifacts"
    required = [
        artifacts / "metrics.json",
        artifacts / "primary_endpoints.csv",
        artifacts / "seed_level_metrics.csv",
        artifacts / "predictions.parquet",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"locked evaluation outputs are missing: {missing}")
    project_root = manuscript_dir.parent
    dataset_summary = json.loads(
        (project_root / "data" / "processed" / "production" / "dataset_summary.json").read_text(
            encoding="utf-8"
        )
    )
    primary = pd.read_csv(artifacts / "primary_endpoints.csv")
    seeds = pd.read_csv(artifacts / "seed_level_metrics.csv")
    predictions = pd.read_parquet(artifacts / "predictions.parquet")
    return artifacts, project_root, dataset_summary, primary, seeds, predictions


def build_paper(
    output_root: str | Path,
    manuscript_dir: str | Path,
    config: StudyConfig,
) -> dict:
    output_root = Path(output_root)
    manuscript_dir = Path(manuscript_dir)
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    artifacts, project_root, dataset, primary, seed_metrics, predictions = _load_inputs(
        output_root, manuscript_dir
    )
    endpoint1 = primary.loc[
        primary["endpoint"] == "paired_macro_f1_random_minus_strain_grouped"
    ].iloc[0]
    endpoint2 = primary.loc[
        primary["endpoint"] == "near_ood_false_accept_rate_at_95pct_known_target"
    ].iloc[0]

    model_summary = (
        seed_metrics.groupby(["design", "model"], as_index=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            top3_accuracy=("top3_accuracy", "mean"),
            log_loss=("log_loss", "mean"),
            brier=("brier", "mean"),
            known_acceptance=("heldout_known_acceptance_rate", "mean"),
            ood_auroc=("ood_auroc", "mean"),
            ood_auprc=("ood_auprc", "mean"),
            near_ood_far=("near_ood_false_accept_rate", "mean"),
            far_ood_far=("far_ood_false_accept_rate", "mean"),
        )
    )
    pivot = model_summary.pivot(index="model", columns="design", values="macro_f1")
    model_delta = (
        pivot.assign(delta_random_minus_grouped=lambda frame: frame.get("spectrum_random") - frame.get("strain_grouped"))
        .reset_index()[["model", "spectrum_random", "strain_grouped", "delta_random_minus_grouped"]]
    )

    primary_grouped = model_summary.loc[
        (model_summary["model"] == config.primary_model)
        & (model_summary["design"] == "strain_grouped")
    ].iloc[0]
    raw_manifest = pd.read_parquet(project_root / "data" / "processed" / "production" / "manifest.parquet")
    known = raw_manifest.loc[raw_manifest["primary_known"]]
    ood = raw_manifest.loc[raw_manifest["primary_ood"]]
    near = ood.loc[ood["ood_distance"] == "near"]
    far = ood.loc[ood["ood_distance"] == "far"]

    table1 = pd.DataFrame(
        [
            ["All retained spectra", len(raw_manifest), raw_manifest["strain_id"].nunique(), raw_manifest["species"].nunique()],
            ["Primary known set (>=5 strains/species)", len(known), known["strain_id"].nunique(), known["species"].nunique()],
            ["Primary OOD set (<5 strains/species)", len(ood), ood["strain_id"].nunique(), ood["species"].nunique()],
            ["Near-OOD", len(near), near["strain_id"].nunique(), near["species"].nunique()],
            ["Far-OOD", len(far), far["strain_id"].nunique(), far["species"].nunique()],
            ["Singleton-OOD", int(raw_manifest["ood_singleton"].sum()), raw_manifest.loc[raw_manifest["ood_singleton"], "strain_id"].nunique(), raw_manifest.loc[raw_manifest["ood_singleton"], "species"].nunique()],
        ],
        columns=["Analysis population", "Spectra", "Strains", "Species"],
    )
    table2 = model_delta.rename(
        columns={
            "model": "Model",
            "spectrum_random": "Spectrum-random macro-F1",
            "strain_grouped": "Strain-grouped macro-F1",
            "delta_random_minus_grouped": "Difference",
        }
    )
    table3 = model_summary.loc[model_summary["design"] == "strain_grouped", [
        "model", "known_acceptance", "ood_auroc", "ood_auprc", "near_ood_far", "far_ood_far"
    ]].rename(
        columns={
            "model": "Model",
            "known_acceptance": "Known acceptance",
            "ood_auroc": "OOD AUROC",
            "ood_auprc": "OOD AUPRC",
            "near_ood_far": "Near-OOD false acceptance",
            "far_ood_far": "Far-OOD false acceptance",
        }
    )
    for frame in (table2, table3):
        numeric = frame.select_dtypes(include=["number"]).columns
        frame[numeric] = frame[numeric].round(4)
    table1.to_csv(artifacts / "table1_dataset_composition.csv", index=False)
    table2.to_csv(artifacts / "table2_split_performance.csv", index=False)
    table3.to_csv(artifacts / "table3_open_set_performance.csv", index=False)

    e1_supported = float(endpoint1["holm_adjusted_p"]) < 0.05 and float(endpoint1["estimate"]) > 0
    e2_supported = float(endpoint2["holm_adjusted_p"]) < 0.05 and float(endpoint2["estimate"]) > 0.05
    e1_sentence = (
        "Spectrum-random evaluation produced higher macro-F1 than strain-grouped evaluation"
        if e1_supported
        else "The prespecified analysis did not establish higher macro-F1 under spectrum-random evaluation"
    )
    e2_sentence = (
        "Near-OOD false acceptance exceeded the prespecified 5% null rate"
        if e2_supported
        else "Near-OOD false acceptance was not shown to exceed the prespecified 5% null rate"
    )
    abstract = f"""**Abstract**

Matrix-assisted laser desorption ionization-time of flight mass spectrometry classifiers are often evaluated on technical spectra rather than independent strains, and forced-choice classifiers cannot represent an organism absent from training. We analyzed the openly licensed RKI v4.2 database using identical preprocessing and six prespecified classifiers. The primary known set contained {known['species'].nunique()} species, {known['strain_id'].nunique():,} strains, and {len(known):,} spectra. Spectrum-random and strain-grouped fivefold evaluation were repeated with five seeds. The remaining {ood['species'].nunique()} species served as previously unseen classes. Probabilities were temperature scaled using training-only grouped predictions; independent calibration strains set a 95% known-acceptance threshold and conformal species sets. For ExtraTrees, the random-minus-grouped macro-F1 difference was {_ci(endpoint1)}. Near-OOD false acceptance was {_ci(endpoint2)}, with held-out known acceptance of {_fmt(endpoint2['heldout_known_acceptance'])}. {e1_sentence}. {e2_sentence}. These findings quantify evaluation and rejection behavior within one reference-spectrum collection; they do not estimate patient-level diagnostic accuracy or prospective clinical utility."""

    importance = f"""**Importance**

Technical replicate spectra from one bacterial strain are not independent biological samples. When replicates are divided between training and testing, a classifier may be rewarded for recognizing a previously observed strain rather than a new strain of the same species. A second problem arises when software must choose among known species even though the measured organism was absent from training. Using {len(raw_manifest):,} public spectra from {raw_manifest['strain_id'].nunique():,} strains, this study measures both effects under a fixed pipeline and evaluates an explicit option to withhold a species-level answer. The results define what this public benchmark can and cannot demonstrate and provide reusable strain-level splits, predictions, and code for future MALDI-TOF classifier evaluations."""

    methods_text = f"""## Materials and Methods

### Study design and data source

This retrospective computational benchmark used version 4.2 of the Robert Koch Institute MALDI-TOF database, distributed through Zenodo under CC BY 4.0 (1, 2). The release contains microbial reference spectra and non-personal acquisition metadata. No patients, identifiable information, animals, new cultures, or pathogen handling were involved. The protocol and statistical analysis plan were fixed before aggregate test predictions were inspected.

Every Zenodo object was checked against its deposited byte count and MD5 value and was then assigned a local SHA-256 digest. The archive expanded to {len(raw_manifest):,} readable Bruker `fid`/`acqu` spectrum units. Folder-level strain directories were the conservative biological grouping unit because 33 directories contained more than one raw CMT identity. Original folder and CMT labels were retained. Twenty documented aliases mapped 270 raw taxon labels to {raw_manifest['species'].nunique()} canonical species, reproducing the release workbook counts. Exact raw duplicates, unreadable spectra, zero-signal spectra, and unresolved labels were to be excluded with explicit reason codes; none of the retained analysis records lacked a taxon identifier.

### Spectral processing

Bruker flex files were read with `readBrukerFlexData` and `MALDIquantForeign` (24). Each spectrum was processed independently to prevent cross-spectrum leakage. The pipeline retained 2,000 to <13,000 Da, applied a square-root transform, Savitzky-Golay smoothing with half-window 10 (25, 26), SNIP baseline removal with 100 iterations (27), truncation of negative residuals, total-ion-current normalization, and summation into 11,000 half-open 1-Da bins. Each float32 feature row was required to be finite, nonnegative, and sum to one within 1e-6. The deposited 30- and 45-peak MSP files were used only for parser quality control, not model fitting.

### Analysis populations and splitting

Species with at least five retained strains formed the primary known set: {known['species'].nunique()} species, {known['strain_id'].nunique():,} strains, and {len(known):,} spectra. Species represented by fewer than five strains formed the primary OOD set: {ood['species'].nunique()} species and {ood['strain_id'].nunique():,} strains. OOD species were designated near when their genus was represented in the fitted known set and far otherwise. Species with at least three strains formed a prespecified sensitivity set.

Fivefold out-of-fold evaluation used seeds {', '.join(str(value) for value in config.seeds)}. Spectrum-random folds stratified spectra by species and deliberately permitted technical replicates from one strain to occur in training and testing. Strain-grouped folds used the strain directory as an indivisible group. Within each grouped outer-training partition, 20% of strains per species were reserved for rejection calibration; fit, calibration, and test strain sets were disjoint. Hierarchical data require group-aware validation because ordinary cross-validation can underestimate error when related observations cross partitions (28-34).

### Models, calibration, and selective reporting

The prespecified classifiers were cosine nearest centroid, linear and radial-basis-function support vector machines, ExtraTrees, XGBoost, and a three-block one-dimensional convolutional neural network. Existing MALDI-TOF machine-learning studies have used these model families but vary in preprocessing, split unit, external validation, and uncertainty handling (3-17). ExtraTrees was the confirmatory model because it was the strongest transferred classifier in a recent selected-subset analysis of the RKI resource (3); that result was not treated as evidence of superiority in the full release.

Classifier settings were fixed before test fitting; grouped threefold out-of-fold predictions within fit strains were used only for temperature estimation. Within each species, every strain received equal total training weight and that weight was divided among its spectra. One scalar temperature was fitted from those training-only grouped scores (39, 40). Calibration strains were then used to set the lower empirical fifth percentile of strain-median maximum probabilities, targeting 95% known acceptance.

Open-set recognition distinguishes examples belonging to known classes from those outside the training label space (35-38, 41, 42). Species prediction sets used nonconformity `1-p(true class)` and finite-sample quantiles (43-47). A class-specific threshold was used only with at least 20 calibration spectra; otherwise a pooled fallback was labelled. Because spectra within a calibration strain are correlated, no distribution-free strain-level 95% coverage claim was made. A species was reported only when its set was a singleton and maximum probability exceeded the confidence threshold. Rejected species calls fell back to a genus only when the aggregated genus set was a singleton; all other results were `unidentified`.

### Outcomes and statistics

The co-primary endpoints were the mean across seeds of the paired ExtraTrees macro-F1 difference between spectrum-random and strain-grouped out-of-fold evaluation, and the species-macro near-OOD false-acceptance rate at the threshold targeting 95% known acceptance. Secondary metrics were balanced accuracy, top-three accuracy, log loss, multiclass Brier score (48), 15-bin expected calibration error, OOD area under the receiver-operating-characteristic and precision-recall curves, false-positive rate at 95% OOD sensitivity, selective risk-coverage, genus fallback, and near-, far-, and singleton-OOD results.

For the leakage endpoint, 2,000 bootstrap samples resampled strains within species and carried all paired predictions and seeds with each strain. Near-OOD intervals resampled species and then strains. One-sided tests compared the leakage difference with zero and near-OOD false acceptance with 0.05; Holm adjustment controlled the two-test family-wise error rate (49, 50). Technical spectra were never counted as independent biological replicates in uncertainty estimates.

### Data and code availability

The source spectra and license are available from Zenodo at https://doi.org/10.5281/zenodo.14562231. The release package accompanying this study contains the frozen manifest, aliases, split assignments, source code, environment locks, tests, predictions, metric numerators and denominators, figure data, protocol, analysis plan, and deviation log. A permanent repository DOI must replace the release-package placeholder before submission."""

    results_text = f"""## Results

### Data integrity and analysis populations

All {len(raw_manifest):,} exported spectra were readable and yielded finite 1-Da feature vectors. They represented {raw_manifest['strain_id'].nunique():,} conservative strain-directory groups, {raw_manifest['species'].nunique()} canonical species, and {raw_manifest['genus'].nunique()} genera. The primary known set contained {len(known):,} spectra from {known['strain_id'].nunique():,} strains and {known['species'].nunique()} species. The primary OOD set contained {len(ood):,} spectra from {ood['strain_id'].nunique():,} strains and {ood['species'].nunique()} species; {near['species'].nunique()} were near-OOD and {far['species'].nunique()} were far-OOD. Thirty-three strain directories, comprising {int(raw_manifest['cmt_identity_conflict'].sum()):,} spectra, contained heterogeneous raw CMT identity strings and therefore remained grouped by directory rather than split by those strings (Table 1; Fig. 1).

### Technical-replicate leakage

For ExtraTrees, spectrum-random out-of-fold macro-F1 was {_fmt(model_delta.loc[model_delta['model'] == config.primary_model, 'spectrum_random'].iloc[0], 4)} and strain-grouped macro-F1 was {_fmt(model_delta.loc[model_delta['model'] == config.primary_model, 'strain_grouped'].iloc[0], 4)}. The prespecified random-minus-grouped difference was {_ci(endpoint1, 4)}; the one-sided Holm-adjusted P value was {_fmt(endpoint1['holm_adjusted_p'], 4)}. {e1_sentence} under the confirmatory criterion (Table 2; Fig. 2). Model-specific contrasts are retained in Table 2 and are not pooled as independent observations.

### Calibration and unseen-species behavior

Under strain-grouped ExtraTrees evaluation, the mean known-set acceptance was {_fmt(endpoint2['heldout_known_acceptance'], 4)}. The near-OOD species-macro false-acceptance rate was {_ci(endpoint2, 4)}, with Holm-adjusted P={_fmt(endpoint2['holm_adjusted_p'], 4)} against the prespecified 0.05 null. {e2_sentence}. The mean strain-grouped OOD AUROC was {_fmt(primary_grouped['ood_auroc'], 4)}, and the OOD AUPRC was {_fmt(primary_grouped['ood_auprc'], 4)}. Near- and far-OOD false-acceptance estimates for every classifier are shown in Table 3. Confidence distributions and selective reporting outcomes are shown in Figures 3 and 4.

### Error structure and robustness

The most frequent strain-disjoint species confusions are displayed in Figure 5. Per-species estimates, calibration support, prediction-set sizes, genus fallback, sensitivity bin widths, the >=3-strain cohort, and the replicate-injection dose-response are reported in the supplement. These analyses were interpreted as secondary or sensitivity results and did not replace the co-primary endpoints."""

    discussion = f"""## Discussion

This benchmark separated two questions that are easily conflated when technical MALDI-TOF spectra are treated as independent cases. First, it measured how the evaluation unit changed apparent species-classification performance while holding the spectrum collection and analysis pipeline fixed. Second, it tested whether a classifier trained only on common species could withhold an unsupported species call for rarer, unseen taxa. The ExtraTrees leakage contrast was {_ci(endpoint1, 4)}, and near-OOD false acceptance was {_ci(endpoint2, 4)} at a held-out known acceptance of {_fmt(endpoint2['heldout_known_acceptance'], 4)}. The results therefore describe evaluation bias and selective behavior in RKI v4.2 rather than clinical sensitivity or specificity.

The distinction between spectra and strains is consequential because technical replicates share organism, culture, preparation, acquisition, and often run-level characteristics. General machine-learning work has shown that leakage between related observations can yield reproducible but optimistic results (28-34). MALDI-specific studies have already evaluated novel replicates, strains, species, uncertainty, and out-of-distribution behavior (6, 8). The contribution here is narrower: paired quantification of random-versus-grouped evaluation on the complete RKI v4.2 resource, combined with a prespecified replicate-contamination experiment and the same-fold assessment of near- and far-OOD species. We do not describe strain-disjoint or open-set MALDI analysis as unprecedented.

Forced argmax classification answers which known class has the largest score, not whether that class is supported. This matters most for near-OOD organisms because related species may share peaks and can receive confident but wrong known-class assignments. Reject options have a long statistical history (35-46), and MALDI-TOF studies have begun to use uncertainty and conformal methods for resistance prediction (8, 47). Our hierarchical rule makes the reporting action explicit: a species call, a genus-only fallback, or no identification. Its value should be judged jointly by unknown-species rejection and the proportion of known strains that retain a usable report.

The study has limitations. The RKI collection is a reference database enriched for highly pathogenic and related organisms, not a consecutive clinical sample. Acquisition spans laboratories, years, instruments, preparation variants, and uneven numbers of strains and spectra. These features create a demanding benchmark but do not reproduce clinical prevalence. Taxonomic labels were inherited from the release and normalized with a frozen alias map; 33 folders contained heterogeneous CMT strings, so the directory was used as the conservative grouping unit. Class-conditional calibration used correlated technical spectra from independent calibration strains and therefore does not provide a formal strain-level coverage guarantee. Models were evaluated on one database and a fixed preprocessing family. No result supports replacement of a commercial identification system, reporting without laboratory review, or performance claims for routine specimens.

The practical implication is methodological. MALDI-TOF classifier studies should state whether folds separate spectra, preparations, strains, sites, and time; report performance on truly unseen biological groups; and include a defined action for organisms outside the training label space. Public split files and record-level predictions make those choices auditable. External evaluation on prospectively collected routine spectra would be required before any clinical deployment claim."""

    references = _reference_list(project_root / "references" / "references.csv")
    manuscript = f"""# Strain-disjoint evaluation of technical-replicate leakage and open-set errors in MALDI-TOF bacterial identification

**Article type:** Research Article  
**Authors:** [AUTHOR NAMES TO BE CONFIRMED]  
**Affiliations:** [AFFILIATIONS TO BE CONFIRMED]  
**Corresponding author:** [NAME AND EMAIL TO BE CONFIRMED]

{abstract}

{importance}

## Introduction

Matrix-assisted laser desorption ionization-time of flight mass spectrometry has shortened bacterial identification workflows and is now embedded in clinical microbiology practice (18-23). Identification depends on comparison with a reference library. Organisms that are absent, sparsely represented, closely related, or affected by acquisition differences remain difficult, making database composition and evaluation design part of the evidence supporting a classifier.

Machine learning has been applied to species identification, subspecies separation, and antimicrobial-resistance prediction from MALDI-TOF spectra (3-17). Two systematic reviews found substantial heterogeneity in preprocessing, validation, external testing, code availability, and uncertainty assessment (4, 5). A large benchmark by Mortier and colleagues explicitly separated novel replicates, strains, and species (6), while PIKE incorporated calibrated uncertainty and out-of-distribution rejection for phenotype prediction (8). These studies establish that both biological grouping and unsupported-class behavior require direct evaluation.

Technical spectra from one strain share more than a species label. A spectrum-level random fold can expose the model to another preparation or acquisition of the test strain, thereby changing the prediction problem from generalization to a new strain toward recognition of a partly observed strain. This is a form of hierarchical data leakage (28-34). High closed-set accuracy also does not determine how a classifier behaves when the true organism is absent from its class list. Open-set and reject-option methods address that separate problem by allowing a model to abstain (35-47).

RKI version 4.2 provides {len(raw_manifest):,} public spectra from {raw_manifest['strain_id'].nunique():,} strains and {raw_manifest['species'].nunique()} species, including common, rare, highly pathogenic, and closely related bacteria (1, 2). We used the full release to quantify the effect of spectrum-random versus strain-disjoint validation, evaluate forced classification of unseen species, and test a calibrated hierarchical reporting rule. The prespecified primary model was ExtraTrees. The primary hypotheses were that spectrum-random validation would increase macro-F1 and that near-OOD false acceptance would exceed 5% at a threshold targeting 95% known acceptance.

{methods_text}

{results_text}

{discussion}

## Acknowledgments

The authors acknowledge the Robert Koch Institute and contributing laboratories for releasing the MALDI-TOF reference spectra under CC BY 4.0. Funding and additional acknowledgments must be confirmed by the authors.

## Author contributions

CRediT roles must be completed from actual human contributions before submission. Artificial-intelligence tools are not authors.

## Conflicts of interest

The authors must confirm all financial and nonfinancial interests before submission.

## Ethics statement

This computational study used publicly available microbial reference spectra and associated non-personal metadata. It did not involve human participants, identifiable private information, animals, or new microbial culture; therefore, human-subject and animal-use approval and consent were not applicable.

## References

{references}
"""
    manuscript_path = manuscript_dir / "manuscript_draft.md"
    manuscript_path.write_text(manuscript, encoding="utf-8", newline="\n")
    results_path = manuscript_dir / "results_autogenerated.md"
    results_path.write_text(results_text + "\n", encoding="utf-8", newline="\n")

    legends = """# Figure legends

**Figure 1. Study cohort and evaluation design.** Counts are recomputed from the frozen manifest. Primary-known species had at least five retained strains; all other species were excluded from model fitting and served as unknown classes.

**Figure 2. Spectrum-random and strain-grouped species-classification performance.** Points summarize seed-level fivefold out-of-fold macro-F1. Error bars show the between-seed standard deviation and do not treat seeds as independent biological samples.

**Figure 3. Calibrated confidence for known strains and unseen taxa.** Maximum class probability is shown for held-out known strains, near-OOD species whose genus occurred in training, and far-OOD species whose genus was absent.

**Figure 4. Species-level selective acceptance.** Acceptance required a singleton conformal species set and maximum probability at or above the threshold targeting 95% calibration-strain acceptance. Bars are descriptive model summaries.

**Figure 5. Most frequent strain-disjoint species confusions.** Counts pool five prespecified seeds for the ExtraTrees classifier. Technical spectra are displayed to show the observed error pattern; inferential intervals use strain clusters.
"""
    (manuscript_dir / "figure_legends.md").write_text(legends, encoding="utf-8", newline="\n")

    abstract_words = _word_count(re.sub(r"^\*\*Abstract\*\*", "", abstract))
    importance_words = _word_count(re.sub(r"^\*\*Importance\*\*", "", importance))
    report = {
        "created_at": utc_now(),
        "title": "Strain-disjoint evaluation of technical-replicate leakage and open-set errors in MALDI-TOF bacterial identification",
        "abstract_words": abstract_words,
        "importance_words": importance_words,
        "manuscript_words_total": _word_count(manuscript),
        "reference_count": int(len(pd.read_csv(project_root / "references" / "references.csv"))),
        "abstract_within_jcm_limit": abstract_words <= 250,
        "importance_within_jcm_limit": importance_words <= 150,
        "manuscript_sha256": sha256_file(manuscript_path),
        "results_sha256": sha256_file(results_path),
        "tables": [
            "table1_dataset_composition.csv",
            "table2_split_performance.csv",
            "table3_open_set_performance.csv",
        ],
    }
    atomic_write_json(manuscript_dir / "manuscript_build_report.json", report)
    if not report["abstract_within_jcm_limit"] or not report["importance_within_jcm_limit"]:
        raise ValueError("JCM abstract or Importance word limit exceeded")
    return report
