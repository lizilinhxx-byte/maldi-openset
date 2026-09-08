from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .features import write_feature_bundle
from .taxonomy import assign_analysis_sets
from .util import stable_id


def make_demo_bundle(output_dir: str | Path, seed: int = 20260907) -> dict:
    """Create a small synthetic dataset that contains controlled replicate leakage."""
    rng = np.random.default_rng(seed)
    n_features = 512
    rows = []
    vectors = []
    species_names = [f"Genus{i // 2 + 1} species{i + 1}" for i in range(6)]
    ood_names = ["Genus1 novelA", "Genus3 novelB", "NovelGenus speciesX"]
    spectrum_counter = 0
    for species_index, species in enumerate(species_names):
        genus = species.split()[0]
        species_peaks = rng.choice(n_features, size=18, replace=False)
        for strain_index in range(5):
            strain = f"{species} strain-{strain_index + 1}"
            strain_peaks = rng.choice(n_features, size=12, replace=False)
            for replicate in range(3):
                spectrum_counter += 1
                vector = rng.gamma(1.0, 0.001, size=n_features).astype(np.float32)
                vector[species_peaks] += rng.uniform(0.8, 1.2, size=len(species_peaks))
                vector[strain_peaks] += rng.uniform(0.6, 1.0, size=len(strain_peaks))
                vector += rng.normal(0, 0.01, size=n_features).astype(np.float32)
                vector = np.clip(vector, 0, None)
                vector /= vector.sum()
                vectors.append(vector)
                rows.append(
                    {
                        "spectrum_id": f"demo_{spectrum_counter:05d}",
                        "strain_name": strain,
                        "strain_id": stable_id(strain, prefix="strain_"),
                        "species": species,
                        "genus": genus,
                        "replicate_ordinal": replicate + 1,
                        "included": True,
                        "ood_singleton": False,
                        "qc_flags": "",
                    }
                )
    for species_index, species in enumerate(ood_names):
        genus = species.split()[0]
        species_peaks = rng.choice(n_features, size=18, replace=False)
        n_strains = 1 if species_index == 2 else 2
        for strain_index in range(n_strains):
            strain = f"{species} strain-{strain_index + 1}"
            strain_peaks = rng.choice(n_features, size=12, replace=False)
            for replicate in range(3):
                spectrum_counter += 1
                vector = rng.gamma(1.0, 0.001, size=n_features).astype(np.float32)
                vector[species_peaks] += rng.uniform(0.8, 1.2, size=len(species_peaks))
                vector[strain_peaks] += rng.uniform(0.6, 1.0, size=len(strain_peaks))
                vector = np.clip(vector, 0, None)
                vector /= vector.sum()
                vectors.append(vector)
                rows.append(
                    {
                        "spectrum_id": f"demo_{spectrum_counter:05d}",
                        "strain_name": strain,
                        "strain_id": stable_id(strain, prefix="strain_"),
                        "species": species,
                        "genus": genus,
                        "replicate_ordinal": replicate + 1,
                        "included": True,
                        "ood_singleton": n_strains == 1,
                        "qc_flags": "",
                    }
                )
    manifest = assign_analysis_sets(pd.DataFrame(rows), known_min_strains=5, sensitivity_min_strains=3)
    matrix = sparse.csr_matrix(np.vstack(vectors))
    return write_feature_bundle(
        matrix,
        manifest,
        output_dir,
        {"synthetic_demo": True, "seed": seed, "n_features": n_features},
    )

