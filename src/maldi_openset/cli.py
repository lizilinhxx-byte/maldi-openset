from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from .config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "production.json"


def _csv_values(value: str | None, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maldi-openset",
        description="Reproducible strain-disjoint and open-set MALDI-ToF benchmark",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download and verify the locked Zenodo record")
    fetch.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw" / "zenodo_14562231"))
    fetch.add_argument("--source-dir", default=str(PROJECT_ROOT / "sources"))
    fetch.add_argument("--include", default="taxonomy,metadata,pkf,btmsp,raw")
    fetch.add_argument("--extract", action="store_true")

    taxonomy = sub.add_parser("taxonomy", help="build the taxonomy-only locked manifest")
    taxonomy.add_argument("--workbook")
    taxonomy.add_argument(
        "--output", default=str(PROJECT_ROOT / "data" / "interim" / "taxonomy_manifest.parquet")
    )

    demo = sub.add_parser("demo", help="create a deterministic synthetic smoke-test bundle")
    demo.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed" / "demo"))
    demo.add_argument("--seed", type=int, default=20260907)

    prepare = sub.add_parser("prepare", help="preprocess raw Bruker spectra and build features")
    prepare.add_argument("--raw-extracted", default=str(PROJECT_ROOT / "data" / "raw" / "rki_v4_2"))
    prepare.add_argument("--workbook")
    prepare.add_argument("--peak-table", default=str(PROJECT_ROOT / "data" / "interim" / "peaks.csv"))
    prepare.add_argument(
        "--profile-matrix", default=str(PROJECT_ROOT / "data" / "interim" / "profiles.float32")
    )
    prepare.add_argument(
        "--exporter-manifest", default=str(PROJECT_ROOT / "data" / "interim" / "r_export_manifest.json")
    )
    prepare.add_argument("--run-r", action="store_true")
    prepare.add_argument("--bin-width", type=float)
    prepare.add_argument(
        "--acquisition-metadata",
        default=str(PROJECT_ROOT / "data" / "interim" / "rki_acqu_metadata.jsonl"),
    )
    prepare.add_argument(
        "--species-aliases", default=str(PROJECT_ROOT / "sources" / "species_aliases.json")
    )
    prepare.add_argument("--rscript", default=str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "R" / "R-4.6.1" / "bin" / "Rscript.exe"))
    prepare.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed" / "production"))

    split = sub.add_parser("split", help="create deterministic spectrum and strain split plans")
    split.add_argument("--feature-dir", default=str(PROJECT_ROOT / "data" / "processed" / "production"))
    split.add_argument("--output", default=str(PROJECT_ROOT / "data" / "processed" / "splits.json"))

    train = sub.add_parser("train", help="train one or more resumable benchmark cells")
    train.add_argument("--feature-dir", default=str(PROJECT_ROOT / "data" / "processed" / "production"))
    train.add_argument("--splits", default=str(PROJECT_ROOT / "data" / "processed" / "splits.json"))
    train.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "production"))
    train.add_argument("--designs")
    train.add_argument("--models")
    train.add_argument("--seeds")
    train.add_argument("--folds")
    train.add_argument("--n-jobs", type=int, default=-1)
    train.add_argument("--cell-workers", type=int, default=1)
    train.add_argument("--no-resume", action="store_true")

    dose = sub.add_parser("dose-response", help="run the prespecified technical-replicate injection analysis")
    dose.add_argument("--feature-dir", default=str(PROJECT_ROOT / "data" / "processed" / "production"))
    dose.add_argument("--splits", default=str(PROJECT_ROOT / "data" / "processed" / "splits.json"))
    dose.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "production"))
    dose.add_argument("--seeds")
    dose.add_argument("--folds")

    evaluate = sub.add_parser("evaluate", help="aggregate runs, bootstrap endpoints and draw figures")
    evaluate.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "production"))

    negative = sub.add_parser("negative-control", help="run strain-level label permutation controls")
    negative.add_argument("--feature-dir", default=str(PROJECT_ROOT / "data" / "processed" / "production"))
    negative.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "production"))
    negative.add_argument("--permutations", type=int, default=100)
    negative.add_argument("--seed", type=int, default=20260907)

    rebin = sub.add_parser("rebin", help="derive coarser fixed-grid sensitivity feature bundles")
    rebin.add_argument("--source-dir", default=str(PROJECT_ROOT / "data" / "processed" / "production"))
    rebin.add_argument("--output-dir", required=True)
    rebin.add_argument("--source-width", type=float, default=1.0)
    rebin.add_argument("--target-width", type=float, required=True)

    paper = sub.add_parser("build-paper", help="build manuscript files from locked outputs")
    paper.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "production"))
    paper.add_argument("--manuscript-dir", default=str(PROJECT_ROOT / "manuscript"))

    sub.add_parser("status", help="report availability of inputs and completed cells")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "fetch":
        from .fetch import download_record

        rows = download_record(
            config.zenodo_record_id,
            args.raw_dir,
            args.source_dir,
            _csv_values(args.include),
            extract_raw=args.extract,
        )
        print(json.dumps({"downloaded": len(rows), "files": rows}, indent=2))
        return 0
    if args.command == "taxonomy":
        from .prepare import locate_single, prepare_taxonomy_only

        workbook = Path(args.workbook) if args.workbook else locate_single(PROJECT_ROOT / "data" / "raw", "*Taxonomy*.xlsx")
        print(json.dumps(prepare_taxonomy_only(workbook, args.output, config), indent=2))
        return 0
    if args.command == "demo":
        from .demo import make_demo_bundle

        print(json.dumps(make_demo_bundle(args.output_dir, args.seed), indent=2))
        return 0
    if args.command == "prepare":
        from dataclasses import asdict, replace
        from .prepare import locate_single, prepare_feature_bundle, run_r_exporter
        from .util import atomic_write_json

        effective_config_path = Path(args.config)
        if args.bin_width is not None:
            config = replace(config, bin_width=float(args.bin_width))
            token = str(args.bin_width).replace(".", "p")
            effective_config_path = PROJECT_ROOT / "data" / "interim" / f"effective_config_bin-{token}.json"
            atomic_write_json(effective_config_path, asdict(config))
        workbook = Path(args.workbook) if args.workbook else locate_single(PROJECT_ROOT / "data" / "raw", "*Taxonomy*.xlsx")
        if args.run_r:
            run_r_exporter(
                args.raw_extracted,
                args.peak_table,
                args.exporter_manifest,
                effective_config_path,
                PROJECT_ROOT / "scripts" / "export_bruker_peaks.R",
                args.rscript,
                args.profile_matrix,
            )
        summary = prepare_feature_bundle(
            args.peak_table,
            args.exporter_manifest,
            workbook,
            args.output_dir,
            config,
            extracted_root=args.raw_extracted,
            acquisition_metadata_path=args.acquisition_metadata,
            species_aliases_path=args.species_aliases,
            profile_matrix_path=args.profile_matrix,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "split":
        from .features import load_feature_bundle
        from .splits import build_splits, write_splits

        _, manifest, _ = load_feature_bundle(args.feature_dir)
        assignments = build_splits(manifest, config.seeds, config.outer_folds)
        print(json.dumps(write_splits(assignments, args.output), indent=2))
        return 0
    if args.command == "train":
        from .training import train_grid

        runs = train_grid(
            args.feature_dir,
            args.splits,
            args.output_root,
            config,
            designs=_csv_values(args.designs),
            models=_csv_values(args.models),
            seeds=_csv_values(args.seeds, int),
            folds=_csv_values(args.folds, int),
            n_jobs=args.n_jobs,
            resume=not args.no_resume,
            cell_workers=args.cell_workers,
        )
        print(json.dumps({"completed": len(runs), "runs": [str(path) for path in runs]}, indent=2))
        return 0
    if args.command == "evaluate":
        from .evaluation import evaluate_all

        print(json.dumps(evaluate_all(args.output_root, config), indent=2))
        return 0
    if args.command == "dose-response":
        from .dose_response import run_dose_response

        print(
            json.dumps(
                run_dose_response(
                    args.feature_dir,
                    args.splits,
                    args.output_root,
                    config,
                    seeds=_csv_values(args.seeds, int),
                    folds=_csv_values(args.folds, int),
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "negative-control":
        from .negative_control import run_label_permutation_control

        print(
            json.dumps(
                run_label_permutation_control(
                    args.feature_dir,
                    args.output_root,
                    config,
                    permutations=args.permutations,
                    seed=args.seed,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "rebin":
        from .sensitivity import rebin_feature_bundle

        print(
            json.dumps(
                rebin_feature_bundle(
                    args.source_dir,
                    args.output_dir,
                    args.source_width,
                    args.target_width,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "build-paper":
        from .paper import build_paper

        print(json.dumps(build_paper(args.output_root, args.manuscript_dir, config), indent=2))
        return 0
    if args.command == "status":
        raw = PROJECT_ROOT / "data" / "raw" / "zenodo_14562231"
        runs = list((PROJECT_ROOT / "output" / "production" / "runs").glob("*/run_manifest.json"))
        status = {
            "project_root": str(PROJECT_ROOT),
            "raw_files": len(list(raw.glob("*"))) if raw.exists() else 0,
            "raw_zip_bytes": next((p.stat().st_size for p in raw.glob("*.zip")), 0),
            "production_features": (PROJECT_ROOT / "data" / "processed" / "production" / "features.npz").exists(),
            "completed_runs": sum(
                json.loads(path.read_text(encoding="utf-8")).get("status") == "complete" for path in runs
            ),
        }
        print(json.dumps(status, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
