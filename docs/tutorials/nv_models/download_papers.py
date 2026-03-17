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
import sys
import time
from pathlib import Path

import requests

PAPERS: list[dict[str, str]] = [
    {
        "name": "vaswani2017_attention_is_all_you_need.pdf",
        "url": "https://arxiv.org/pdf/1706.03762",
        "description": "Vaswani et al. - Attention Is All You Need (2017)",
    },
    {
        "name": "abosabie2026_thrombo_inflammatory_mpn.pdf",
        "url": "https://www.biorxiv.org/content/10.64898/2026.02.16.706250v1.full.pdf",
        "description": "Abosabie et al. - Thrombo-inflammatory endothelial signatures in JAK2-mutated MPN (2026)",
    },
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def download(url: str, dest: Path, timeout: int = 120) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
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
