"""Download raw ΔΔG binding-affinity datasets used by SIMBA.

Fetches:
  - SKEMPI2 (general protein-protein ΔΔG)
  - AB-Bind (antibody-antigen point mutants)

AbDesign cannot be auto-downloaded: it requires manual registration at
https://naturalantibody.com/ab-design/. After registering, place the
downloaded file at data/raw/abdesign/abdesign_raw.csv (or .xlsx) and re-run
the cleaning pipeline; scripts here will pick it up if present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCES = {
    "skempi2": {
        "url": "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv",
        "dest": RAW_DIR / "skempi2" / "skempi_v2.csv",
    },
    "skempi2_pdbs": {
        "url": "https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz",
        "dest": RAW_DIR / "skempi2" / "SKEMPI2_PDBs.tgz",
    },
    "abbind": {
        "url": "https://raw.githubusercontent.com/sarahsirin/AB-Bind-Database/master/AB-Bind_experimental_data.csv",
        "dest": RAW_DIR / "abbind" / "AB-Bind_experimental_data.csv",
    },
}

ABBIND_REPO_API = "https://api.github.com/repos/sarahsirin/AB-Bind-Database/contents/"
ABBIND_PDB_RAW = "https://raw.githubusercontent.com/sarahsirin/AB-Bind-Database/master/{name}"


def download_abbind_pdbs(timeout: int = 60) -> None:
    dest_dir = RAW_DIR / "abbind" / "pdbs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(ABBIND_REPO_API, timeout=timeout)
    resp.raise_for_status()
    entries = resp.json()
    pdb_names = [e["name"] for e in entries if e["name"].endswith(".pdb")]
    print(f"[fetch] abbind_pdbs: {len(pdb_names)} structure files")
    for name in pdb_names:
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        r = requests.get(ABBIND_PDB_RAW.format(name=name), timeout=timeout)
        r.raise_for_status()
        dest.write_bytes(r.content)
    print(f"[ok]    abbind_pdbs: {len(list(dest_dir.glob('*.pdb')))} files in {dest_dir}")


def download(name: str, url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {name}: already present at {dest} ({dest.stat().st_size} bytes)")
        return
    print(f"[fetch] {name}: {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"[ok]    {name}: wrote {dest} ({len(resp.content)} bytes)")


def main() -> int:
    for name, spec in SOURCES.items():
        try:
            download(name, spec["url"], spec["dest"])
        except requests.RequestException as exc:
            print(f"[error] {name}: {exc}", file=sys.stderr)
            return 1

    try:
        download_abbind_pdbs()
    except requests.RequestException as exc:
        print(f"[error] abbind_pdbs: {exc}", file=sys.stderr)
        return 1

    abdesign_dir = RAW_DIR / "abdesign"
    abdesign_dir.mkdir(parents=True, exist_ok=True)
    note = abdesign_dir / "MANUAL_DOWNLOAD_REQUIRED.txt"
    if not any(abdesign_dir.glob("abdesign_raw.*")):
        note.write_text(
            "AbDesign requires manual registration at https://naturalantibody.com/ab-design/\n"
            "Download the dataset there and save it as one of:\n"
            "  data/raw/abdesign/abdesign_raw.csv\n"
            "  data/raw/abdesign/abdesign_raw.xlsx\n"
        )
        print(f"[manual] abdesign: see {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
