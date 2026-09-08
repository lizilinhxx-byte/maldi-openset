from pathlib import Path

from maldi_openset.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_production_config_is_valid():
    config = load_config(ROOT / "config" / "production.json")
    assert config.n_bins == 11000
    assert config.known_min_strains == config.outer_folds == 5
    assert config.primary_model == "extra_trees"
    assert len(config.seeds) == 5

