#!/usr/bin/env python3
"""Run the paired MALDI-OpenSet v2 contamination experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from maldi_openset_v2.contamination import plan_contamination, run_contamination  # noqa: E402


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Separate technical-replicate contamination prevalence from intensity, "
            "with paired same-species different-strain donor controls."
        )
    )
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--experiments",
        type=_csv_strings,
        default=("prevalence", "intensity"),
        help="comma-separated: prevalence,intensity",
    )
    parser.add_argument("--seeds", type=_csv_ints, help="optional comma-separated subset")
    parser.add_argument("--rotations", type=_csv_ints, default=(0, 1, 2, 3, 4))
    parser.add_argument(
        "--max-cells",
        type=int,
        help="process at most this many incomplete experiment/seed/rotation cells",
    )
    parser.add_argument("--parallel-cells", type=int, default=2, choices=(1, 2))
    parser.add_argument("--n-jobs", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate cohort construction and print the execution plan without writing",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one prevalence cell with 8 trees under output/v2/contamination/_smoke",
    )
    args = parser.parse_args()
    root = args.project.resolve()
    if args.dry_run:
        payload = plan_contamination(
            root,
            experiments=args.experiments,
            seeds=args.seeds,
            rotations=args.rotations,
        )
    elif args.smoke:
        payload = run_contamination(
            root,
            output_root=root / "output/v2/contamination/_smoke",
            experiments=("prevalence",),
            seeds=(20260907,),
            rotations=(0,),
            max_cells=1,
            parallel_cells=1,
            n_jobs=min(args.n_jobs, 2),
            n_estimators=8,
            bootstrap_replicates=20,
            force=args.force,
        )
    else:
        payload = run_contamination(
            root,
            output_root=args.output,
            experiments=args.experiments,
            seeds=args.seeds,
            rotations=args.rotations,
            max_cells=args.max_cells,
            parallel_cells=args.parallel_cells,
            n_jobs=args.n_jobs,
            n_estimators=args.n_estimators,
            bootstrap_replicates=args.bootstrap_replicates,
            force=args.force,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

