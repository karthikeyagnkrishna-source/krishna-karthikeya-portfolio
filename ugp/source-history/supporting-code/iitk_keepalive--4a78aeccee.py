#!/usr/bin/env python3
"""Refresh an IITK captive-portal keep-alive URL without storing a password."""

from __future__ import annotations

import argparse
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url_file", type=Path)
    parser.add_argument("--interval", type=int, default=5400)
    args = parser.parse_args()

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]

    while True:
        try:
            url = args.url_file.read_text(encoding="utf-8").strip()
            if not url.startswith("https://gateway.iitk.ac.in:1003/"):
                raise RuntimeError("Unexpected IITK keep-alive URL")
            with opener.open(url, timeout=30) as response:
                response.read(1024)
            print(f"{stamp()} keep-alive refreshed", flush=True)
            time.sleep(args.interval)
        except Exception as exc:
            print(f"{stamp()} keep-alive failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
