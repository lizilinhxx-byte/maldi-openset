from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StudyConfig:
    project_name: str
    zenodo_record_id: int
    zenodo_doi: str
    data_license: str
    mass_min: float
    mass_max: float
    bin_width: float
    known_min_strains: int
    sensitivity_min_strains: int
    outer_folds: int
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    primary_model: str
    known_acceptance_target: float
    bootstrap_replicates: int
    preprocessing: dict[str, Any] = field(default_factory=dict)
    sensitivity_bin_widths: tuple[float, ...] = ()

    @property
    def n_bins(self) -> int:
        return int(round((self.mass_max - self.mass_min) / self.bin_width))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StudyConfig":
        cooked = dict(raw)
        for key in ("seeds", "models", "sensitivity_bin_widths"):
            cooked[key] = tuple(cooked.get(key, ()))
        cfg = cls(**cooked)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mass_min >= self.mass_max:
            raise ValueError("mass_min must be below mass_max")
        if self.bin_width <= 0:
            raise ValueError("bin_width must be positive")
        if self.outer_folds < 2:
            raise ValueError("outer_folds must be at least 2")
        if self.known_min_strains < self.outer_folds:
            raise ValueError("known_min_strains must be >= outer_folds")
        if not 0 < self.known_acceptance_target < 1:
            raise ValueError("known_acceptance_target must lie in (0, 1)")
        if self.primary_model not in self.models:
            raise ValueError("primary_model must be listed in models")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")


def load_config(path: str | Path) -> StudyConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return StudyConfig.from_dict(json.load(handle))

