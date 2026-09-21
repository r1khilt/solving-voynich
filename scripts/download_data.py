#!/usr/bin/env python3
"""Retrieve the one pinned public source, using curl's system TLS support."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/zl3b_source.json"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/ZL3b-n.txt"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.output.exists():
        actual = hashlib.sha256(args.output.read_bytes()).hexdigest()
        if actual != manifest["sha256"]:
            raise SystemExit("Existing file checksum mismatch; preserve and inspect it before replacing")
        print(f"Verified cached source: {args.output} ({args.output.stat().st_size} bytes)")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    subprocess.run(["curl", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
                    "--max-time", "60", manifest["url"], "-o", str(temporary)], check=True)
    actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if actual != manifest["sha256"]:
        raise SystemExit(f"Downloaded checksum differs; retained {temporary} for review, no source installed")
    temporary.replace(args.output)
    print(f"Downloaded and verified: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
