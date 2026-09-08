from pathlib import Path

import pandas as pd

from maldi_openset.taxonomy import assign_analysis_sets, load_taxonomy_workbook, taxonomy_summary


ROOT = Path(__file__).resolve().parents[1]


def _workbook():
    matches = list((ROOT / "data" / "raw").rglob("*Taxonomy*.xlsx"))
    return matches[0] if matches else None


def test_locked_rki_taxonomy_counts_when_source_is_present():
    workbook = _workbook()
    if workbook is None:
        return
    spectra, species = load_taxonomy_workbook(workbook)
    manifest = assign_analysis_sets(spectra, 5, 3)
    summary = taxonomy_summary(manifest)
    assert summary["spectra"] == 11055
    assert summary["strains"] == 1601
    assert summary["species"] == 264
    assert summary["genera"] == 73
    assert summary["primary_known_species"] == 56
    assert summary["primary_known_strains"] == 1224
    assert summary["sensitivity_known_species"] == 106
    assert summary["ood_species"] == 208
    assert summary["singleton_ood_species"] == 103
    assert species["declared_strain_count"].sum() == 1601


def test_primary_ood_and_sensitivity_known_can_overlap():
    rows = []
    for species, strains in [("A a", 5), ("B b", 3), ("C c", 1)]:
        for index in range(strains):
            rows.append(
                {
                    "spectrum_id": f"{species}-{index}",
                    "strain_id": f"{species}-strain-{index}",
                    "species": species,
                    "genus": species.split()[0],
                }
            )
    result = assign_analysis_sets(pd.DataFrame(rows), 5, 3)
    b = result[result["species"] == "B b"]
    assert b["primary_ood"].all()
    assert b["sensitivity_known"].all()

