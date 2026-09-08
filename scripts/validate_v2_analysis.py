#!/usr/bin/env python3
"""Fail-closed validation of all MALDI-OpenSet v2 analysis branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


LEGACY_COMMIT = "2293ce52a07e24e507e48814dcd3303bf512c0f3"
FEATURE_SHA256 = "7a57e01a7ed0adf6b76bf1403192f7988802e230f94a13424f35ce78bfbb4f45"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def verify_record(root: Path, record: dict, *, base: Path | None = None) -> None:
    declared = Path(str(record["path"]))
    path = declared if declared.is_absolute() else (root / declared)
    if not path.is_file() and base is not None:
        path = base / declared
    if not path.is_file():
        raise FileNotFoundError(record["path"])
    if int(record["bytes"]) != path.stat().st_size or record["sha256"] != sha256(path):
        raise ValueError(f"artifact hash mismatch: {path}")


def verify_record_collection(root: Path, records, *, base: Path | None = None) -> int:
    values = records.values() if isinstance(records, dict) else records
    count = 0
    for record in values:
        verify_record(root, record, base=base)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project.resolve()
    errors: list[str] = []
    checks: dict[str, object] = {}

    try:
        legacy_diff = subprocess.check_output(
            ["git", "diff", LEGACY_COMMIT, "--", "src/maldi_openset"],
            cwd=root,
            text=True,
        )
        if legacy_diff.strip():
            raise ValueError("legacy src/maldi_openset differs from archived commit")

        frozen_path = root / "output/v2/input_manifest.json"
        frozen_sha = sha256(frozen_path)
        frozen = load(frozen_path)
        if frozen.get("legacy_analysis_commit") != LEGACY_COMMIT:
            raise ValueError("legacy commit identity mismatch")
        if frozen.get("feature_sha256") != FEATURE_SHA256:
            raise ValueError("feature identity mismatch")
        if frozen.get("run_count") != 210 or len(frozen.get("expected_run_keys", [])) != 210:
            raise ValueError("frozen run grid is incomplete")
        expected_keys = {
            (row["design"], row["model"], int(row["seed"]), int(row["fold"]))
            for row in frozen["expected_run_keys"]
        }
        actual_keys = {
            (row["design"], row["model"], int(row["seed"]), int(row["fold"]))
            for row in frozen["runs"]
        }
        if actual_keys != expected_keys or len(frozen["runs"]) != 210:
            raise ValueError("frozen expected and observed run grids differ")
        for record in frozen["files"]:
            verify_record(root, record)
        feature_actual = sha256(root / frozen["feature_matrix_path"])
        if feature_actual != FEATURE_SHA256:
            raise ValueError("feature matrix bytes differ")
        checks["frozen_input_manifest_sha256"] = frozen_sha
        checks["legacy_runs"] = 210
    except Exception as exc:
        errors.append(f"frozen-input validation: {exc}")
        frozen_sha = None

    manifests = {
        "open_set": root / "output/v2/open_set/run_manifest.json",
        "reference_source": root / "output/v2/reference_source/manifests/reference_source_manifest.json",
        "contamination": root / "output/v2/contamination/run_manifest.json",
        "cmt_sensitivity": root / "output/v2/cmt_sensitivity/run_manifest.json",
    }
    try:
        open_manifest = load(manifests["open_set"])
        if open_manifest.get("frozen_input_manifest_sha256") != frozen_sha:
            raise ValueError("open-set frozen binding differs")
        if open_manifest.get("n_locked_runs") != 200 or open_manifest.get("bootstrap_replicates") != 2000:
            raise ValueError("open-set execution grid incomplete")
        open_outputs = verify_record_collection(root, open_manifest["outputs"])
        checks["open_set"] = {"outputs": open_outputs, "manifest_sha256": sha256(manifests["open_set"])}
    except Exception as exc:
        errors.append(f"open-set validation: {exc}")

    try:
        reference = load(manifests["reference_source"])
        if reference.get("status") != "complete" or reference.get("frozen_input_manifest_sha256") != frozen_sha:
            raise ValueError("reference/source status or frozen binding differs")
        if reference.get("run_count") != 70 or reference.get("unexpected_run_count") != 0:
            raise ValueError("reference/source run universe differs")
        if not reference.get("all_cells_hash_and_fingerprint_verified"):
            raise ValueError("reference/source cell verification missing")
        reference_outputs = verify_record_collection(root, reference["artifacts"])
        checks["reference_source"] = {
            "runs": 70,
            "outputs": reference_outputs,
            "manifest_sha256": sha256(manifests["reference_source"]),
        }
    except Exception as exc:
        errors.append(f"reference/source validation: {exc}")

    try:
        contamination_path = manifests["contamination"]
        contamination = load(contamination_path)
        binding = contamination["inputs"]["frozen_input_manifest"]["input_manifest_sha256"]
        audit = contamination["checkpoint_universe_audit"]
        if contamination.get("status") != "complete" or binding != frozen_sha:
            raise ValueError("contamination status or frozen binding differs")
        if contamination.get("completed_fits") != 400 or contamination.get("expected_fits") != 400:
            raise ValueError("contamination fit grid incomplete")
        if not audit.get("observed_equals_expected"):
            raise ValueError("contamination checkpoint universe differs")
        contamination_outputs = verify_record_collection(
            root,
            contamination["output_files"],
            base=contamination_path.parent,
        )
        checks["contamination"] = {
            "fits": 400,
            "outputs": contamination_outputs,
            "manifest_sha256": sha256(contamination_path),
        }
    except Exception as exc:
        errors.append(f"contamination validation: {exc}")

    try:
        cmt = load(manifests["cmt_sensitivity"])
        binding = cmt["provenance"]["common_freeze"]["sha256"]
        if cmt.get("status") != "complete" or binding != frozen_sha:
            raise ValueError("CMT status or frozen binding differs")
        if not cmt["provenance"].get("all_inputs_verified_before_analysis"):
            raise ValueError("CMT transitive input verification missing")
        cmt_outputs = verify_record_collection(root, cmt["outputs"])
        checks["cmt_sensitivity"] = {
            "excluded_strain_directories": cmt["component_counts"]["excluded_strain_directories"],
            "outputs": cmt_outputs,
            "manifest_sha256": sha256(manifests["cmt_sensitivity"]),
        }
    except Exception as exc:
        errors.append(f"CMT validation: {exc}")

    author = load(root / "config/v2/author_information.json")
    if len(author.get("authors", [])) != 2 or not any(row.get("corresponding_author") for row in author["authors"]):
        errors.append("author-information validation: two authors/corresponding author not resolved")
    checks["author_information"] = {
        "authors": len(author.get("authors", [])),
        "funding_records": len(author.get("funding", [])),
        "orcid_missing": sum(row.get("orcid") is None for row in author.get("authors", [])),
        "final_approval_required": author.get("author_final_approval_required"),
    }

    report = {
        "status": "pass" if not errors else "fail",
        "legacy_analysis_commit": LEGACY_COMMIT,
        "feature_sha256": FEATURE_SHA256,
        "checks": checks,
        "errors": errors,
    }
    destination = root / "output/v2/validation_report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
