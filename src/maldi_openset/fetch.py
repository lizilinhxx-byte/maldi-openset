from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from .util import atomic_write_json, md5_file, sha256_file, utc_now

ZENODO_API = "https://zenodo.org/api/records/{record_id}"


def _cern_address() -> str:
    addresses = {
        item[4][0]
        for item in socket.getaddrinfo("paas-apps-shard-3.cern.ch", 443, type=socket.SOCK_STREAM)
        if ":" not in item[4][0]
    }
    if not addresses:
        raise OSError("could not resolve the CERN Zenodo application endpoint")
    return sorted(addresses)[0]


def _curl_download(url: str, destination: Path, resume: bool = False) -> None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise OSError("curl is required for the Zenodo DNS fallback")
    command = [
        curl,
        "-L",
        "--fail",
        "--retry",
        "4",
        "--retry-delay",
        "2",
        "--resolve",
        f"zenodo.org:443:{_cern_address()}",
    ]
    if resume:
        command.extend(["-C", "-"])
    command.extend(["-o", str(destination), url])
    subprocess.run(command, check=True)


def fetch_record(record_id: int) -> dict:
    request = urllib.request.Request(
        ZENODO_API.format(record_id=record_id),
        headers={"User-Agent": "maldi-openset/0.1 (reproducible research)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError):
        with tempfile.TemporaryDirectory(prefix="maldi-zenodo-") as temp:
            path = Path(temp) / "record.json"
            _curl_download(ZENODO_API.format(record_id=record_id), path)
            return json.loads(path.read_text(encoding="utf-8"))


def classify_file(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".xlsx") and "taxonomy" in lower:
        return "taxonomy"
    if lower.endswith("metadata.pdf"):
        return "metadata"
    if lower.endswith(".pkf"):
        return "pkf"
    if lower.endswith(".zip"):
        return "raw"
    if lower.endswith(".btmsp"):
        return "btmsp"
    return "other"


def _download(url: str, destination: Path, expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "maldi-openset/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if offset else "wb"
    try:
        with urllib.request.urlopen(request, timeout=180) as response, partial.open(mode) as out:
            if offset and getattr(response, "status", None) == 200:
                out.close()
                partial.unlink(missing_ok=True)
                return _download(url, destination, expected_size)
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and expected_size and partial.stat().st_size == expected_size:
            pass
        else:
            raise
    except (urllib.error.URLError, TimeoutError, OSError):
        _curl_download(url, partial, resume=bool(offset))
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise IOError(
            f"incomplete download for {destination.name}: "
            f"{partial.stat().st_size} != {expected_size}"
        )
    os.replace(partial, destination)


def safe_extract_zip(archive: str | Path, destination: str | Path) -> list[Path]:
    archive = Path(archive)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe zip member: {member.filename}")
        zf.extractall(destination)
        extracted = [(destination / member.filename) for member in zf.infolist()]
    return extracted


def download_record(
    record_id: int,
    raw_dir: str | Path,
    source_dir: str | Path,
    include: Iterable[str] = ("taxonomy", "metadata", "pkf", "raw"),
    extract_raw: bool = False,
) -> list[dict]:
    include_set = set(include)
    raw_dir = Path(raw_dir)
    source_dir = Path(source_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    record = fetch_record(record_id)
    atomic_write_json(source_dir / "zenodo_record.json", record)
    rows: list[dict] = []
    for item in record.get("files", []):
        category = classify_file(item["key"])
        if category not in include_set:
            continue
        target = raw_dir / item["key"]
        url = item.get("links", {}).get("self") or item.get("links", {}).get("download")
        if not url:
            raise KeyError(f"no download URL for {item['key']}")
        expected_size = int(item["size"])
        if not target.exists() or target.stat().st_size != expected_size:
            _download(url, target, expected_size)
        expected_md5 = str(item.get("checksum", "")).removeprefix("md5:")
        actual_md5 = md5_file(target)
        if expected_md5 and actual_md5.lower() != expected_md5.lower():
            raise IOError(f"MD5 mismatch for {target.name}")
        row = {
            "filename": target.name,
            "category": category,
            "bytes": target.stat().st_size,
            "zenodo_md5": expected_md5,
            "computed_md5": actual_md5,
            "sha256": sha256_file(target),
            "source_url": url,
            "downloaded_at": utc_now(),
        }
        rows.append(row)
        if extract_raw and category == "raw":
            safe_extract_zip(target, raw_dir / "extracted")
    csv_path = source_dir / "source_files.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return rows
