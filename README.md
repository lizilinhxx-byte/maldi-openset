# MALDI OpenSet analysis code

Source code for **Within-strain replicate leakage and unseen-species abstention in a public MALDI-TOF bacterial reference collection**.

This initial repository contains the original analysis modules, version 2 correction and sensitivity modules, analysis configurations, preprocessing script, and self-contained unit tests. It is a **source-only deposit**, not the complete frozen reproduction archive. A synthetic demo tests execution; it does not reproduce the manuscript results.

## Installation and smoke checks

Use Python 3.11 or newer. Install from the repository root in a virtual environment:

```sh
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -e ".[dev]"
python -m pytest -q
python -m maldi_openset --help
python -m maldi_openset demo --output-dir data/processed/demo
```

The recorded production dependency versions are in `requirements-production-lock.txt` and `R-requirements-lock.txt`; their presence does not guarantee cross-platform installation. CNN training additionally needs the optional PyTorch dependency. Bruker preprocessing uses R and `scripts/export_bruker_peaks.R`; supply `--rscript` when R is not at the Windows default path.

## Structure

- `src/maldi_openset/`: original data ingestion, preprocessing, splitting, models, calibration and evaluation.
- `src/maldi_openset_v2/`: corrected reporting action, strain-level sensitivity, contamination controls and reference/source analyses.
- `scripts/run_v2_*.py`: version 2 command-line runners.
- `config/`: analysis parameters, without private author metadata.
- `tests/`: unit tests using fixtures or synthetic data. Full-production manuscript/release integration tests are not included because their frozen inputs are not included.
- `sources/`: taxonomy aliases and copies of the upstream license texts for reference, not a grant of new permissions.

## Reproducing the historical analyses

The study used RKI MALDI-TOF version 4.2: https://doi.org/10.5281/zenodo.14562231 . This is the **upstream dataset DOI**, not a DOI for this software. The original frozen analysis commit was `2293ce52a07e24e507e48814dcd3303bf512c0f3`; this source export does not import that Git history.

The version 2 runners require the corresponding historical predictions, probabilities, split assignments, feature matrix, and frozen input manifest in the expected directory layout. Those inputs are not in this initial repository. Do not represent a fresh run or the synthetic demo as an exact reproduction of the frozen paper results. The reviewed companion reproducibility deposit and its permanent DOI remain pending.

Example commands, **only after the required frozen inputs have been restored**:

```sh
python scripts/run_v2_open_set.py --help
python scripts/run_v2_reference_source.py --help
python scripts/run_v2_contamination.py --help
python scripts/run_v2_cmt_sensitivity.py --help
python scripts/validate_v2_analysis.py --help
```

## Analysis provenance

The original protocol specified singleton prediction sets and a confidence threshold. Version 2 additionally enforces agreement between the singleton-set label and argmax after historical results were inspected. Original and corrected decisions must remain distinguishable. Version 2 is estimation-focused and adds no confirmatory P-value claim.

Human authors are responsible for verifying the analyses and approving publication statements. This source deposit is not a journal-policy approval or clinical validation.

## License and excluded material

Project-authored code is under the MIT License in `LICENSE`. That license does not relicense third-party spectra. The retained source record gives CC BY 4.0 at record level and CC BY-NC 4.0 for the spectrum files; depositor clarification remains pending. Obtain upstream data directly and respect its applicable terms.

Raw spectra, extracted feature matrices, fitted models, record-level predictions, manuscript/submission documents, private author records, API credentials, and workstation logs are excluded from this source-only upload. Outputs produced locally are ignored by `.gitignore` and should be reviewed before any separate release.
