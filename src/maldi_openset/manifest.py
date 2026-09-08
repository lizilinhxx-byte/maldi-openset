from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .taxonomy import assign_analysis_sets, load_taxonomy_workbook, normalize_label
from .util import sha256_file, stable_id


def _read_acqu(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="latin-1", errors="replace")

    def field(name: str) -> str | None:
        match = re.search(rf"^##\${re.escape(name)}=\s*(?:<([^>]*)>|([^\r\n]*))", text, re.MULTILINE)
        if not match:
            return None
        return (match.group(1) if match.group(1) is not None else match.group(2)).strip()

    def tags(value: str | None) -> dict[str, str]:
        if not value:
            return {}
        return {key: raw.strip() for key, raw in re.findall(r"#([A-Za-z0-9_]+)=([^#]*)", value)}

    cmt1 = tags(field("CMT1"))
    cmt3 = tags(field("CMT3"))
    taxid_text = cmt1.get("UIE") or cmt1.get("UID") or ""
    taxid = int(taxid_text) if taxid_text.isdigit() else None
    genus = cmt1.get("GEN") or None
    epithet = cmt1.get("SPE") or None
    return {
        "raw_genus_label": genus,
        "raw_species_epithet": epithet,
        "raw_species_label": f"{genus} {epithet}" if genus and epithet else None,
        "raw_strain_label": cmt1.get("STR") or None,
        "raw_type_label": cmt1.get("TYP") or None,
        "taxid": taxid,
        "growth_time": cmt1.get("GTI") or None,
        "growth_temperature": cmt1.get("TEM") or None,
        "atmosphere": cmt1.get("AIR") or None,
        "growth_medium": cmt1.get("MED") or None,
        "spore_status": cmt1.get("SPO") or None,
        "acquisition_date": field("AQ_DATE"),
        "measurement_label": field("CMT2"),
        "prep_method": cmt3.get("TRT") or None,
        "operator_label": cmt3.get("EXT") or None,
        "calibration_material": cmt3.get("CAL") or None,
        "source_lab": cmt3.get("CUS") or None,
    }


def _spectrum_digest(fid_path: Path) -> str | None:
    if not fid_path.is_file():
        return None
    digest = hashlib.sha256()
    for name in ("fid", "acqu", "acqus"):
        path = fid_path.parent / name
        if not path.is_file():
            continue
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _match_strain(source_path: str, strain_names: list[str]) -> str | None:
    normalized_path = normalize_label(source_path).casefold()
    candidates = [name for name in strain_names if name.casefold() in normalized_path]
    return max(candidates, key=len) if candidates else None


def manifest_from_export(
    exporter_manifest: str | Path,
    taxonomy_workbook: str | Path,
    extracted_root: str | Path | None = None,
    acquisition_metadata_path: str | Path | None = None,
    species_aliases_path: str | Path | None = None,
    known_min_strains: int = 5,
    sensitivity_min_strains: int = 3,
) -> pd.DataFrame:
    exporter_manifest = Path(exporter_manifest)
    raw = json.loads(exporter_manifest.read_text(encoding="utf-8"))
    records = pd.DataFrame(raw.get("records", []))
    if records.empty:
        raise ValueError("R exporter manifest contains no spectra")
    records["profile_row"] = range(len(records))
    taxonomy_spectra, species_table = load_taxonomy_workbook(taxonomy_workbook)
    canonical_species = set(species_table["species"].astype(str))
    strain_map = (
        taxonomy_spectra[["strain_name", "strain_id", "species", "genus"]]
        .drop_duplicates("strain_name")
        .set_index("strain_name")
    )
    strain_names = sorted(strain_map.index.astype(str).tolist(), key=len, reverse=True)
    metadata_path = Path(acquisition_metadata_path) if acquisition_metadata_path else None
    if metadata_path is not None and metadata_path.is_file():
        metadata = pd.read_json(metadata_path, lines=True, dtype=False)
        metadata["source_key"] = metadata["archive_spectrum_dir"].astype(str).str.replace("\\", "/") + "/fid"
        metadata = metadata.drop(columns=["spectrum_id"], errors="ignore")
        records["source_key"] = records["source_path"].astype(str).str.replace("\\", "/")
        records = records.merge(metadata, on="source_key", how="left", validate="one_to_one")
        if records["archive_spectrum_dir"].isna().any():
            raise ValueError("acquisition metadata does not cover every exported spectrum")
        aliases = {}
        if species_aliases_path and Path(species_aliases_path).is_file():
            aliases = json.loads(Path(species_aliases_path).read_text(encoding="utf-8")).get("alias_map", {})
        records["directory_genus_label"] = records["folder_genus"].map(normalize_label)
        records["directory_taxon_label"] = records["folder_species"].map(normalize_label)
        records["directory_strain_label"] = records["folder_strain"].map(normalize_label)
        records["species"] = records["directory_taxon_label"].map(lambda value: aliases.get(value, value))
        records["genus"] = records["species"].map(
            lambda value: value.split()[0] if isinstance(value, str) and value else None
        )
        records["strain_name"] = records["directory_strain_label"]

        def group_path(row) -> str:
            parts = str(row["archive_spectrum_dir"]).replace("\\", "/").split("/")
            measurement = str(row.get("folder_measurement") or "")
            if measurement in parts:
                parts = parts[: parts.index(measurement)]
            elif len(parts) > 4:
                parts = parts[:-4]
            return "/".join(parts)

        records["strain_group_path"] = records.apply(group_path, axis=1)
        records["strain_id"] = records["strain_group_path"].map(
            lambda value: stable_id(value, prefix="strain_")
        )
        records["raw_genus_label"] = records["gen"].map(normalize_label)
        records["raw_species_epithet"] = records["spe"].map(normalize_label)
        records["raw_species_label"] = records.apply(
            lambda row: normalize_label(f"{row.get('gen', '')} {row.get('spe', '')}"), axis=1
        )
        records["raw_strain_label"] = records["str"].map(normalize_label)
        records["raw_type_label"] = records["typ"].map(normalize_label)
        records["taxid"] = pd.to_numeric(records["uie"].where(records["uie"].ne(""), records["uid"]), errors="coerce").astype("Int64")
        records["growth_time"] = records["gti"]
        records["growth_temperature"] = records["tem"]
        records["atmosphere"] = records["air"]
        records["growth_medium"] = records["med"]
        records["spore_status"] = records["spo"]
        records["acquisition_date"] = records["aq_date"]
        records["prep_method"] = records["trt"]
        records["operator_label"] = records["ext"]
        records["calibration_material"] = records["cal"]
        records["source_lab"] = records["cus"]
        records["taxonomy_status"] = records["species"].isin(canonical_species).map(
            {True: "matched", False: "unresolved"}
        )
        records["spectrum_ordinal"] = range(1, len(records) + 1)
    else:
        acquisition: dict[str, dict[str, object]] = {}
        if extracted_root is not None:
            root = Path(extracted_root)
            for source_path in records["source_path"].astype(str).unique():
                acquisition[source_path] = _read_acqu((root / source_path).parent / "acqu")
        raw_metadata = pd.DataFrame.from_records(
            [acquisition.get(path, {}) for path in records["source_path"].astype(str)], index=records.index
        )
        for column in raw_metadata.columns:
            records[column] = raw_metadata[column]
        path_parts = records["source_path"].astype(str).map(lambda value: Path(value).parts)
        records["directory_genus_label"] = path_parts.map(lambda parts: parts[0] if len(parts) > 0 else None)
        records["directory_taxon_label"] = path_parts.map(lambda parts: parts[1] if len(parts) > 1 else None)
        records["directory_strain_label"] = path_parts.map(lambda parts: parts[2] if len(parts) > 2 else None)

        def resolve_strain(row) -> str | None:
            genus = normalize_label(row.get("raw_genus_label"))
            epithet = normalize_label(row.get("raw_species_epithet"))
            strain = normalize_label(str(row.get("raw_strain_label") or "").replace("|", " "))
            directory_taxon = normalize_label(row.get("directory_taxon_label"))
            directory_constructed = normalize_label(
                " ".join(value for value in (directory_taxon, strain) if value)
            )
            directory_exact = [
                name
                for name in strain_names
                if normalize_label(name).casefold() == directory_constructed.casefold()
            ]
            if directory_exact:
                return directory_exact[0]
            constructed = normalize_label(" ".join(value for value in (genus, epithet, strain) if value))
            exact = [name for name in strain_names if normalize_label(name).casefold() == constructed.casefold()]
            if exact:
                return exact[0]
            return _match_strain(str(row["source_path"]), strain_names)

        records["strain_name"] = records.apply(resolve_strain, axis=1)
        records = records.join(strain_map, on="strain_name")
        records["taxonomy_status"] = records["species"].notna().map({True: "matched", False: "unresolved"})

        taxonomy_ordinals = defaultdict(list)
        for _, row in taxonomy_spectra.sort_values("replicate_ordinal").iterrows():
            taxonomy_ordinals[row["strain_name"]].append(int(row["spectrum_ordinal"]))
        seen = defaultdict(int)
        spectrum_ordinals = []
        for strain in records["strain_name"]:
            index = seen[strain]
            choices = taxonomy_ordinals.get(strain, [])
            spectrum_ordinals.append(choices[index] if index < len(choices) else None)
            seen[strain] += 1
        records["spectrum_ordinal"] = spectrum_ordinals
    records["cmt_identity"] = records.apply(
        lambda row: "|".join(
            normalize_label(row.get(column))
            for column in ("raw_genus_label", "raw_species_epithet", "raw_strain_label")
        ),
        axis=1,
    )
    conflict_strains = set(
        records.dropna(subset=["strain_id"])
        .groupby("strain_id")["cmt_identity"]
        .nunique()
        .loc[lambda values: values > 1]
        .index.astype(str)
    )
    records["cmt_identity_conflict"] = records["strain_id"].astype(str).isin(conflict_strains)
    records["raw_sha256"] = None
    if extracted_root is not None:
        root = Path(extracted_root)
        hashes: dict[str, str | None] = {}
        for source_path in records["source_path"].astype(str).unique():
            path = root / source_path
            hashes[source_path] = _spectrum_digest(path)
        records["raw_sha256"] = records["source_path"].map(hashes)

    records["source"] = "RKI MALDI-ToF v4.2"
    for column in ("taxid", "acquisition_date", "source_lab", "prep_method"):
        if column not in records:
            records[column] = pd.NA
    records["qc_flags"] = records.apply(
        lambda row: ";".join(
            flag
            for flag, active in (
                ("zero-peaks", int(row.get("peak_count", 0)) == 0),
                ("cmt-identity-conflict", bool(row.get("cmt_identity_conflict", False))),
            )
            if active
        ),
        axis=1,
    )
    raw_present = (
        records["raw_sha256"].notna()
        if extracted_root is not None
        else pd.Series(True, index=records.index)
    )
    records["included"] = (
        records["taxonomy_status"].eq("matched")
        & records["peak_count"].fillna(0).astype(int).gt(0)
        & raw_present
    )
    records["exclusion_reason"] = ""
    records.loc[records["taxonomy_status"].eq("unresolved"), "exclusion_reason"] = "unresolved-taxonomy"
    records.loc[records["peak_count"].fillna(0).astype(int).eq(0), "exclusion_reason"] = "zero-peaks"
    records["spectrum_id"] = records["spectrum_id"].astype(str)
    duplicate = records.sort_values("spectrum_id").duplicated("raw_sha256", keep="first") & records[
        "raw_sha256"
    ].notna()
    records.loc[duplicate, "included"] = False
    records.loc[duplicate, "exclusion_reason"] = "exact_raw_duplicate"
    included = assign_analysis_sets(
        records.loc[records["included"]].copy(),
        known_min_strains=known_min_strains,
        sensitivity_min_strains=sensitivity_min_strains,
    )
    excluded = records.loc[~records["included"]].copy()
    excluded["species_strain_count"] = 0
    excluded["analysis_set"] = "excluded"
    excluded["primary_known"] = False
    excluded["primary_ood"] = False
    excluded["sensitivity_known"] = False
    excluded["ood_distance"] = None
    excluded["ood_singleton"] = False
    output = pd.concat([included, excluded], ignore_index=True, sort=False)
    return output.sort_values(["included", "spectrum_id"], ascending=[False, True]).reset_index(drop=True)
