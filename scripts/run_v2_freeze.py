#!/usr/bin/env python3
"""Freeze v1 artifacts as immutable inputs for the separated v2 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "2293ce52a07e24e507e48814dcd3303bf512c0f3"
EXPECTED_FEATURE = "7a57e01a7ed0adf6b76bf1403192f7988802e230f94a13424f35ce78bfbb4f45"
DESIGNS = ("spectrum_random", "strain_grouped")
COMPLETE_MODELS = ("cnn1d", "cosine_centroid", "extra_trees", "rbf_svm")
SEEDS = (20260907, 20260919, 20261001, 20261013, 20261025)
FOLDS = (0, 1, 2, 3, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    subprocess.run(
        ["git", "cat-file", "-e", f"{EXPECTED_COMMIT}^{{commit}}"],
        cwd=root,
        check=True,
    )
    source_diff = subprocess.check_output(
        ["git", "diff", EXPECTED_COMMIT, "--", "src/maldi_openset"], cwd=root, text=True
    )
    if source_diff.strip():
        raise RuntimeError("legacy src/maldi_openset differs from the archived analysis commit")

    metadata = json.loads(
        (root / "data/processed/production/feature_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("features_sha256") != EXPECTED_FEATURE:
        raise RuntimeError("feature metadata hash identity differs")
    feature_path = root / "data/processed/production/features.npy"
    if sha256(feature_path) != EXPECTED_FEATURE:
        raise RuntimeError("feature matrix bytes differ from the locked hash")

    fixed = [
        "config/production.json",
        "config/v2/analysis_spec.json",
        "data/processed/production/manifest.parquet",
        "data/processed/production/feature_metadata.json",
        "data/processed/splits.json",
        "data/processed/splits.parquet",
        "output/production/artifacts/predictions.parquet",
        "output/production/artifacts/seed_level_metrics.csv",
        "output/production/artifacts/primary_endpoints.csv",
        "protocol/SAP.md",
        "protocol/v2/amendment.md",
        "protocol/v2/SAP.md",
    ]
    files = [
        {
            "path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": sha256(root / relative),
        }
        for relative in fixed
    ]

    expected_keys = {
        (design, model, seed, fold)
        for design in DESIGNS
        for model in COMPLETE_MODELS
        for seed in SEEDS
        for fold in FOLDS
    }
    expected_keys.update(
        (design, "xgboost", 20260907, fold)
        for design in DESIGNS
        for fold in FOLDS
    )
    actual_manifest_hash = sha256(root / "data/processed/production/manifest.parquet")
    actual_split_hash = sha256(root / "data/processed/splits.parquet")
    runs = []
    observed_keys = set()
    for manifest_path in sorted((root / "output/production/runs").glob("*/run_manifest.json")):
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        if record.get("git_commit") != EXPECTED_COMMIT or record.get("feature_sha256") != EXPECTED_FEATURE:
            continue
        key = (
            str(record.get("design")),
            str(record.get("model")),
            int(record.get("seed")),
            int(record.get("fold")),
        )
        if key not in expected_keys:
            raise RuntimeError(f"unexpected legacy run key: {key}")
        if key in observed_keys:
            raise RuntimeError(f"duplicate legacy run key: {key}")
        if record.get("status") != "complete":
            raise RuntimeError(f"legacy run is not complete: {key}")
        if record.get("feature_manifest_sha256") != actual_manifest_hash:
            raise RuntimeError(f"legacy run manifest identity mismatch: {key}")
        if record.get("split_sha256") != actual_split_hash:
            raise RuntimeError(f"legacy run split identity mismatch: {key}")
        observed_keys.add(key)
        run_dir = manifest_path.parent
        for field, name in (
            ("predictions_sha256", "predictions.parquet"),
            ("probabilities_sha256", "probabilities.npy"),
            ("classes_sha256", "classes.json"),
        ):
            if sha256(run_dir / name) != record.get(field):
                raise RuntimeError(f"v1 run hash mismatch: {run_dir.name}/{name}")
        runs.append(
            {
                "run_id": record["run_id"],
                "model": record["model"],
                "design": record["design"],
                "seed": record["seed"],
                "fold": record["fold"],
                "manifest_path": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": sha256(manifest_path),
                "predictions_sha256": record["predictions_sha256"],
                "probabilities_sha256": record["probabilities_sha256"],
                "classes_sha256": record["classes_sha256"],
                "model_sha256": record.get("model_sha256"),
                "source_tree_sha256": record.get("source_tree_sha256"),
                "config_sha256": record.get("config_sha256"),
                "split_sha256": record.get("split_sha256"),
            }
        )
    if observed_keys != expected_keys or len(runs) != len(expected_keys):
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise RuntimeError(f"legacy run grid mismatch: missing={missing}, extra={extra}")

    output = root / "output/v2/input_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2.0.0",
        "legacy_analysis_commit": EXPECTED_COMMIT,
        "current_head_at_freeze": head,
        "feature_sha256": EXPECTED_FEATURE,
        "feature_matrix_path": "data/processed/production/features.npy",
        "feature_matrix_bytes": feature_path.stat().st_size,
        "files": files,
        "runs": runs,
        "run_count": len(runs),
        "expected_run_keys": [
            {"design": design, "model": model, "seed": seed, "fold": fold}
            for design, model, seed, fold in sorted(expected_keys)
        ],
        "v1_write_protected_by_contract": True,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "run_count": len(runs), "sha256": sha256(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
