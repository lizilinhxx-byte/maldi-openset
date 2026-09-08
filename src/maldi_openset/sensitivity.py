from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from .features import load_feature_bundle, write_feature_bundle


def rebin_feature_bundle(
    source_dir: str | Path,
    output_dir: str | Path,
    source_bin_width: float,
    target_bin_width: float,
) -> dict:
    if target_bin_width < source_bin_width:
        raise ValueError("finer bins require reprocessing the raw profiles")
    ratio = target_bin_width / source_bin_width
    factor = int(round(ratio))
    if not np.isclose(ratio, factor) or factor < 1:
        raise ValueError("target bin width must be an integer multiple of source width")
    matrix, manifest, metadata = load_feature_bundle(source_dir)
    dense = matrix.toarray().astype(np.float32) if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float32)
    usable = dense.shape[1] - dense.shape[1] % factor
    rebinned = dense[:, :usable].reshape(len(dense), usable // factor, factor).sum(axis=2)
    row_sums = rebinned.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("rebinned TIC sums are invalid")
    parameters = dict(metadata.get("parameters", {}))
    parameters.update(
        {
            "derived_from_features_sha256": metadata.get("features_sha256"),
            "source_bin_width": source_bin_width,
            "bin_width": target_bin_width,
            "rebin_factor": factor,
        }
    )
    return write_feature_bundle(rebinned.astype(np.float32), manifest.copy(), output_dir, parameters)
