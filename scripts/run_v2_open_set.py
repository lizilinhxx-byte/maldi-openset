from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maldi_openset_v2.open_set import run_analysis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the locked MALDI-OpenSet v2 reporting-rule correction and sensitivities."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--production-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--frozen-input-manifest", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="replace only the dedicated output/v2/open_set directory")
    args = parser.parse_args()

    root = args.project_root.resolve()
    production = (args.production_root or root / "output" / "production").resolve()
    output = (args.output_dir or root / "output" / "v2" / "open_set").resolve()
    spec = (args.spec or root / "config" / "v2" / "analysis_spec.json").resolve()
    frozen_input_manifest = (
        args.frozen_input_manifest or root / "output" / "v2" / "input_manifest.json"
    ).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            parser.error(f"output directory is nonempty; use --force to replace it: {output}")
        expected_parent = (root / "output" / "v2").resolve()
        if expected_parent not in output.parents or output.name != "open_set":
            parser.error("refusing to remove anything except output/v2/open_set")
        shutil.rmtree(output)
    manifest = run_analysis(
        root,
        production,
        output,
        spec,
        args.bootstrap_replicates,
        frozen_input_manifest,
    )
    print(f"v2 open-set analysis complete: {output}")
    print(f"locked runs: {manifest['n_locked_runs']}")
    print(f"bootstrap replicates: {manifest['bootstrap_replicates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
