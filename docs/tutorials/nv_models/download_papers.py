#!/usr/bin/env python3
"""Download sample papers for PaperQA tutorials.

Papers are saved into the papers/ directory next to this script.

Usage:
  python download_papers.py               # download all default papers
  python download_papers.py --list        # show what will be downloaded
  python download_papers.py --force       # re-download even if files exist
  python download_papers.py --dest /tmp   # custom output directory
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

PAPERS: list[dict[str, str]] = [
    {
        "name": "vaswani2017_attention_is_all_you_need.pdf",
        "url": "https://arxiv.org/pdf/1706.03762",
        "description": "Vaswani et al. - Attention Is All You Need (2017)",
    },
    {
        "name": "palandri2026_momelotinib_os_myelofibrosis.pdf",
        "url": "pmc:PMC12924849",
        "description": "Palandri et al. - Overall survival with momelotinib vs. BAT in ruxolitinib-experienced myelofibrosis (2026)",
    },
]

PMC_OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def resolve_pmc_pdf_url(pmc_id: str, timeout: int = 30) -> str:
    """Use the NCBI PMC Open Access API to resolve a PMC ID to a PDF URL."""
    clean_id = pmc_id.replace("pmc:", "").replace("PMC", "")
    resp = requests.get(
        PMC_OA_API, params={"id": f"PMC{clean_id}"}, timeout=timeout
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    error = root.find(".//error")
    if error is not None:
        raise RuntimeError(f"PMC OA API error for PMC{clean_id}: {error.text}")

    for link in root.findall(".//link"):
        fmt = link.get("format", "")
        href = link.get("href", "")
        if fmt == "pdf" and href:
            return href.replace("ftp://", "https://")

    raise RuntimeError(
        f"No PDF link found via PMC OA API for PMC{clean_id}. "
        "Article may not be in the Open Access subset."
    )


def download(url: str, dest: Path, timeout: int = 120) -> None:
    if url.startswith("pmc:"):
        url = resolve_pmc_pdf_url(url, timeout=timeout)
        print(f"      resolved → {url}")

    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    resp = session.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    if len(resp.content) < 1024:
        raise RuntimeError(f"Response too small ({len(resp.content)} bytes), may not be a valid PDF")
    dest.write_bytes(resp.content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent / "papers",
                        help="Output directory (default: papers/ next to this script).")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    parser.add_argument("--list", action="store_true", help="Only list papers; do not download.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    args = parser.parse_args()

    if args.list:
        for p in PAPERS:
            print(f"  {p['name']}")
            print(f"    {p['description']}")
            print(f"    {p['url']}")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for entry in PAPERS:
        dest_path = args.dest / entry["name"]
        if dest_path.exists() and not args.force:
            print(f"SKIP  {entry['name']} (already exists, use --force to re-download)")
            ok += 1
            continue

        print(f"GET   {entry['name']}")
        print(f"      {entry['url']}")
        try:
            download(entry["url"], dest_path, timeout=args.timeout)
            size_kb = dest_path.stat().st_size / 1024
            print(f"  OK  {size_kb:.0f} KB -> {dest_path}")
            ok += 1
        except Exception as exc:
            print(f"  FAIL {exc}")
            fail += 1

        time.sleep(1)

    print(f"\nDone: {ok} succeeded, {fail} failed.  Output dir: {args.dest}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
