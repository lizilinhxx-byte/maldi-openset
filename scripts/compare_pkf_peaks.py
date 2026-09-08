#!/usr/bin/env python3
"""Cross-check locally detected raw-spectrum peaks against published RKI MSP peaks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("|", " ")
    return re.sub(r"\s+", " ", text).strip().casefold()


def pkf_rows(path: Path):
    rows = np.atleast_1d(loadmat(path, struct_as_record=False, squeeze_me=True)["C"])
    for row in rows:
        yield {
            "identity": (norm(row.gen), norm(row.spe), norm(row.str)),
            "masses": np.asarray(row.pik, dtype=float)[0],
            "pkf_name": str(row.nam),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=2.0)
    args = parser.parse_args()
    project = args.project.resolve()
    manifest = pd.read_parquet(project / "data/processed/production/manifest.parquet")
    manifest["identity"] = list(
        zip(
            manifest["raw_genus_label"].map(norm),
            manifest["raw_species_epithet"].map(norm),
            manifest["raw_strain_label"].map(norm),
        )
    )
    identity_to_strains = manifest.groupby("identity")["strain_id"].agg(lambda values: sorted(set(values)))
    peaks = pd.read_csv(project / "data/interim/peaks.csv", usecols=["spectrum_id", "mass"])
    peaks = peaks.merge(manifest[["spectrum_id", "strain_id"]], on="spectrum_id", how="inner")
    raw_dir = project / "data/raw/zenodo_14562231"
    outputs = []
    exclusions = []
    for filename in ("230306_ZENODO_30Peaks_0.75.pkf", "230306_ZENODO_45Peaks_0.75.pkf"):
        candidates = []
        for row in pkf_rows(raw_dir / filename):
            strains = identity_to_strains.get(row["identity"], [])
            if len(strains) != 1:
                exclusions.append(
                    {"pkf_file": filename, "identity": "|".join(row["identity"]), "matched_strains": len(strains)}
                )
                continue
            row["strain_id"] = strains[0]
            row["hash"] = hashlib.sha256(strains[0].encode()).hexdigest()
            candidates.append(row)
        selected = sorted(candidates, key=lambda row: row["hash"])[: args.sample]
        selected_ids = {row["strain_id"] for row in selected}
        local = {
            strain: np.sort(part["mass"].to_numpy(dtype=float))
            for strain, part in peaks.loc[peaks["strain_id"].isin(selected_ids)].groupby("strain_id")
        }
        for row in selected:
            local_mass = local.get(row["strain_id"], np.array([], dtype=float))
            distances = []
            for mass in row["masses"]:
                if len(local_mass):
                    index = np.searchsorted(local_mass, mass)
                    neighbours = local_mass[max(0, index - 1) : min(len(local_mass), index + 1)]
                    distances.append(float(np.min(np.abs(neighbours - mass))))
                else:
                    distances.append(float("nan"))
            finite = np.asarray(distances)[np.isfinite(distances)]
            outputs.append(
                {
                    "pkf_file": filename,
                    "strain_id": row["strain_id"],
                    "pkf_name": row["pkf_name"],
                    "published_peak_count": len(row["masses"]),
                    "matched_within_tolerance": int((finite <= args.tolerance).sum()),
                    "match_proportion": float((finite <= args.tolerance).mean()),
                    "median_absolute_mass_error_da": float(np.median(finite)),
                    "max_absolute_mass_error_da": float(np.max(finite)),
                }
            )
    destination = project / "output/production/qc"
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(outputs)
    frame.to_csv(destination / "pkf_peak_crosscheck.csv", index=False)
    summary = {
        "sample_per_pkf": args.sample,
        "tolerance_da": args.tolerance,
        "rows": len(frame),
        "median_match_proportion": float(frame["match_proportion"].median()),
        "minimum_match_proportion": float(frame["match_proportion"].min()),
        "median_absolute_mass_error_da": float(frame["median_absolute_mass_error_da"].median()),
        "ambiguous_or_unmatched_pkf_rows": len(exclusions),
        "interpretation": "Parser QC only; the PKF MSP files are not model inputs.",
    }
    (destination / "pkf_peak_crosscheck_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (destination / "pkf_peak_crosscheck_exclusions.json").write_text(
        json.dumps(exclusions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

