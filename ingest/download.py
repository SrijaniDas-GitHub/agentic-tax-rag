"""Download every source in the manifest into data/raw/.

Not needed to run the app: data/chunks.jsonl is committed. This rebuilds the
corpus from the original sources.

    uv run python -m ingest.download
    uv run python -m ingest.download --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from ingest.manifest import RAW_DIR, Doc, load_manifest, save_manifest

# gov.uk and irs.gov both reject the default urllib user agent.
UA = "tax-rag/0.1 (+https://github.com/; educational RAG project)"
TIMEOUT = 120


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, dest.open("wb") as out:
        while block := resp.read(1 << 16):
            out.write(block)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download manifest sources into data/raw/")
    ap.add_argument("--force", action="store_true", help="re-download even if the file is present")
    args = ap.parse_args(argv)

    docs = load_manifest()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Two UK entries share one gov.uk page, so fetch by filename.
    by_file: dict[str, Doc] = {}
    for d in docs:
        by_file.setdefault(d.filename, d)

    digests: dict[str, str] = {}
    failed = False
    for filename, doc in by_file.items():
        dest = doc.path
        if dest.exists() and not args.force:
            print(f"  present   {filename}  ({dest.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  fetching  {filename}  <- {doc.download_url}")
            try:
                fetch(doc.download_url, dest)
            except Exception as exc:                      # noqa: BLE001 - report and continue
                print(f"  FAILED    {filename}: {exc}")
                failed = True
                continue
            print(f"            saved ({dest.stat().st_size / 1e6:.1f} MB)")
        digests[filename] = sha256(dest)

    if failed:
        print("\nat least one source failed; not touching manifest sha256 values")
        return 1

    print()
    changed = []
    for d in docs:
        got = digests.get(d.filename)
        if got is None:
            continue
        if d.sha256 is None:
            print(f"  {d.doc_id:16s} sha256 recorded  {got[:16]}…")
            changed.append(Doc(**{**d.__dict__, "sha256": got}))
        elif d.sha256 != got:
            # Not fatal: sources get updated upstream. Re-run ingest and review the diff.
            print(f"  {d.doc_id:16s} sha256 CHANGED since manifest - source was updated")
            print(f"  {'':16s}   manifest {d.sha256[:16]}…  now {got[:16]}…")
        else:
            print(f"  {d.doc_id:16s} sha256 ok        {got[:16]}…")

    if changed:
        save_manifest(changed)
        print(f"\n  wrote {len(changed)} sha256 value(s) into data/manifest.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
