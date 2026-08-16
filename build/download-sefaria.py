"""
download_sefaria.py

Downloads all cltk-full files and schemas from Sefaria's public export,
skipping any (book, version) pairs listed in an exclusions CSV
(e.g. versions whose detected language doesn't match their stored
language — see removable_versions.csv).

Output structure:
    sefaria_export/
        cltk-full/{categories}/{title}/{language}/{versionTitle}.json
        schemas/{title}.json

No API key or authentication required.

Usage:
    python download_sefaria.py                        # download everything, minus exclusions
    python download_sefaria.py --skip-cltk             # schemas only
    python download_sefaria.py --skip-schemas          # cltk-full only
    python download_sefaria.py --workers 8              # more parallel downloads (default: 4)
    python download_sefaria.py --out ./my_dir           # custom output folder
    python download_sefaria.py --exclude-csv other.csv  # use a different exclusions file
    python download_sefaria.py --no-exclude             # ignore removable_versions.csv entirely
"""

import argparse
import csv
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlparse

BOOKS_JSON_URL = (
    "https://raw.githubusercontent.com/Sefaria/Sefaria-Export/master/books.json"
)
GCS_BASE = "https://storage.googleapis.com/sefaria-export"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
DEFAULT_EXCLUDE_CSV = "removable_versions.csv"


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        import json

        return json.loads(resp.read())


def load_exclusions(csv_path: Path) -> set[tuple[str, str]]:
    """
    Load a set of (book_id, version_title) pairs to skip.

    Matches removable_versions.csv's columns: version_id, book_id,
    version_title, stored_language, detected_lang, confidence, reason.
    Only book_id + version_title are used — that's what actually maps
    to a file on disk, regardless of why the row was flagged.
    """
    exclusions: set[tuple[str, str]] = set()
    if not csv_path.exists():
        print(f"[!] Exclusions file not found at {csv_path} — downloading everything.")
        return exclusions

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "book_id" not in (reader.fieldnames or []) or "version_title" not in (
            reader.fieldnames or []
        ):
            print(
                f"[!] {csv_path} is missing 'book_id'/'version_title' columns — "
                f"found {reader.fieldnames}. Skipping exclusions."
            )
            return exclusions
        for row in reader:
            book_id = (row.get("book_id") or "").strip()
            version_title = (row.get("version_title") or "").strip()
            if book_id and version_title:
                exclusions.add((book_id, version_title))

    print(f"[*] Loaded {len(exclusions):,} version exclusions from {csv_path}")
    return exclusions


def download_file(
    url: str, dest: Path, retries: int = MAX_RETRIES
) -> tuple[str, bool, str]:
    """Download url to dest. Returns (url, success, message)."""
    if dest.exists():
        return url, True, "skipped (exists)"

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            # The raw URL may have spaces, semicolons, and literal ? chars
            # (from book/version titles) that confuse urllib's parser.
            # Split only on "://" to get scheme+host vs. raw path, then
            # percent-encode the path ourselves before handing it to urllib.
            scheme_host, _, raw_path = url.partition("googleapis.com")
            encoded_path = quote(raw_path, safe="/")
            encoded_path = encoded_path.replace(";", "%3B").replace("?", "%3F")
            safe_url = scheme_host + "googleapis.com" + encoded_path

            with urllib.request.urlopen(safe_url, timeout=60) as resp:
                data = resp.read()

            dest.write_bytes(data)
            return url, True, "ok"

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return url, False, f"404 not found"
            if attempt < retries:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return url, False, f"HTTP {e.code}"

        except Exception as e:
            if attempt < retries:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return url, False, str(e)

    return url, False, "max retries exceeded"


def build_tasks(
    books: list[dict],
    out_dir: Path,
    skip_cltk: bool,
    skip_schemas: bool,
    exclusions: set[tuple[str, str]],
) -> tuple[list[tuple[str, Path]], int]:
    tasks = []
    excluded_count = 0

    if not skip_cltk:
        for book in books:
            url = book.get("cltk_full_url")
            if not url:
                continue

            title = book.get("title", "")

            # Reconstruct local path from URL: strip GCS_BASE prefix
            rel = url.replace(GCS_BASE + "/", "")
            dest = out_dir / rel

            # Derive the version title the same way it ends up on disk
            # (filename stem, underscores standing in for spaces) rather
            # than trusting a possibly-differently-named field on `book`.
            # This guarantees the exclusion check matches what actually
            # gets written, regardless of the exact books.json schema.
            version_title_from_path = dest.stem.replace("_", " ")

            candidate_keys = {version_title_from_path}
            for field in ("versionTitle", "version_title"):
                val = book.get(field)
                if val:
                    candidate_keys.add(str(val))

            if any((title, vt) in exclusions for vt in candidate_keys):
                excluded_count += 1
                continue

            tasks.append((url, dest))

    if not skip_schemas:
        titles = sorted(set(b["title"] for b in books))
        for title in titles:
            # GCS schema object names use underscores instead of spaces.
            filename = title.replace(" ", "_") + ".json"
            url = f"{GCS_BASE}/schemas/{filename}"
            dest = out_dir / "schemas" / filename
            tasks.append((url, dest))

    return tasks, excluded_count


def run(args):
    out_dir = Path(args.out)
    print(f"Fetching book list from GitHub...")
    data = fetch_json(BOOKS_JSON_URL)
    books = data["books"]
    print(
        f"Found {len(books):,} book entries ({len(set(b['title'] for b in books)):,} unique titles)"
    )

    exclusions: set[tuple[str, str]] = set()
    if not args.no_exclude:
        exclusions = load_exclusions(Path(args.exclude_csv))

    tasks, excluded_count = build_tasks(
        books, out_dir, args.skip_cltk, args.skip_schemas, exclusions
    )
    total = len(tasks)

    already_done = sum(1 for _, dest in tasks if dest.exists())
    to_download = total - already_done
    print(
        f"\nFiles to download : {to_download:,}  (skipping {already_done:,} already present, "
        f"{excluded_count:,} excluded by CSV)"
    )
    print(f"Output directory  : {out_dir.resolve()}")
    print(f"Workers           : {args.workers}\n")

    if to_download == 0:
        print("Nothing to do — all files already downloaded.")
        return

    success = skipped = failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_file, url, dest): url for url, dest in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            url, ok, msg = future.result()
            if msg.startswith("skipped"):
                skipped += 1
            elif ok:
                success += 1
            else:
                failed += 1
                print(f"  [FAIL] {msg}  {url}", file=sys.stderr)

            if i % 500 == 0 or i == total:
                elapsed = time.time() - start
                rate = (success + failed) / elapsed if elapsed else 0
                eta = (to_download - success - failed) / rate if rate else 0
                print(
                    f"  [{i:>6}/{total}]  "
                    f"ok={success}  skipped={skipped}  failed={failed}  "
                    f"{rate:.1f} files/s  ETA {eta / 60:.1f}m"
                )

    elapsed = time.time() - start
    print(f"\nFinished in {elapsed / 60:.1f} min")
    print(f"  Downloaded : {success:,}")
    print(f"  Skipped    : {skipped:,}  (already existed)")
    print(f"  Excluded   : {excluded_count:,}  (matched exclusions CSV)")
    print(f"  Failed     : {failed:,}")
    print(f"  Output     : {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Sefaria cltk-full files and schemas."
    )
    parser.add_argument(
        "--out",
        default="sefaria_export",
        help="Output directory (default: sefaria_export)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel download workers (default: 4)"
    )
    parser.add_argument("--skip-cltk", action="store_true", help="Skip cltk-full files")
    parser.add_argument("--skip-schemas", action="store_true", help="Skip schema files")
    parser.add_argument(
        "--exclude-csv",
        default=DEFAULT_EXCLUDE_CSV,
        help=f"CSV of (book_id, version_title) pairs to skip (default: {DEFAULT_EXCLUDE_CSV})",
    )
    parser.add_argument(
        "--no-exclude",
        action="store_true",
        help="Ignore the exclusions CSV and download every version",
    )
    args = parser.parse_args()

    if args.skip_cltk and args.skip_schemas:
        print(
            "Error: --skip-cltk and --skip-schemas both set — nothing to download.",
            file=sys.stderr,
        )
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
