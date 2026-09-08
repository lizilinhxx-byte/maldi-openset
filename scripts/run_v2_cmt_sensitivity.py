#!/usr/bin/env python3
"""Run the dedicated MALDI-OpenSet v2 CMT-identity sensitivity."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maldi_openset_v2.cmt_sensitivity import run_analysis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exclude conservative CMT-identity components and recompute frozen "
            "ExtraTrees leakage/open-set sensitivities."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the dedicated output/v2/cmt_sensitivity directory",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = (
        args.output_dir or root / "output" / "v2" / "cmt_sensitivity"
    ).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            parser.error(f"output directory is nonempty; use --force: {output}")
        expected_parent = (root / "output" / "v2").resolve()
        if expected_parent not in output.parents or output.name != "cmt_sensitivity":
            parser.error("refusing to remove anything except output/v2/cmt_sensitivity")
        shutil.rmtree(output)

    result = run_analysis(root, output)
    counts = result["component_counts"]
    print(f"CMT sensitivity complete: {output}")
    print(
        "excluded directory groups: "
        f"{counts['excluded_strain_directories']} "
        f"({counts['excluded_primary_known_strains']} primary-known)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

