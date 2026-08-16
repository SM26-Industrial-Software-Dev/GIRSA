#!/usr/bin/env python3
"""
add_authors.py

Parses Sefaria schema files to extract rich author metadata objects,
populates an 'authors' table with bilingual mapping names, and links records
to the 'books' table via a many-to-many join table.
"""

import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = "sefaria.db"
EXPORT_DIR = Path("./sefaria_export")  # Must contain your 'schemas' folder


def main():
    schemas_dir = EXPORT_DIR / "schemas"

    if not Path(DB_PATH).exists():
        print(f"[-] Target database '{DB_PATH}' not found.")
        sys.exit(1)
    if not schemas_dir.exists():
        print(f"[-] Schemas directory not found at: {schemas_dir}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Initialize DB structures with Foreign Keys enabled
    cur.execute("PRAGMA foreign_keys = ON;")

    print("[*] Creating authors and author_book_map schemas...")
    cur.executescript("""
        BEGIN TRANSACTION;

        -- Master table for distinct author identities
        CREATE TABLE IF NOT EXISTS authors (
            id          TEXT PRIMARY KEY,  -- The Sefaria unique 'slug' identifier
            en_name     TEXT,              -- English display name
            he_name     TEXT               -- Hebrew display name
        );

        -- Many-to-Many Join Table
        CREATE TABLE IF NOT EXISTS author_book_map (
            author_id   TEXT NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
            book_id     TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
            PRIMARY KEY (author_id, book_id)
        );

        CREATE INDEX IF NOT EXISTS idx_auth_book_book ON author_book_map(book_id);
        
        COMMIT;
    """)

    # 2. Map existing books inside the database to keep data linked accurately
    cur.execute("SELECT id FROM books;")
    valid_book_ids = {row[0] for row in cur.fetchall()}
    print(f"[+] Loaded {len(valid_book_ids)} valid book rows from database context.")

    # 3. Read and process schema targets
    schema_files = list(schemas_dir.glob("*.json"))
    print(f"[*] Extracting bilingual authors across {len(schema_files)} files...")

    cur.execute("BEGIN TRANSACTION;")
    authors_inserted = 0
    links_inserted = 0

    for fpath in schema_files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        book_title = data.get("title", "").strip()
        if not book_title or book_title not in valid_book_ids:
            continue

        # Target the explicit dictionary structure array safely
        raw_authors = data.get("authors", [])
        if not isinstance(raw_authors, list):
            continue

        for author_obj in raw_authors:
            if not isinstance(author_obj, dict):
                continue

            # Extract names using the schema definition keys
            slug = author_obj.get("slug", "").strip()
            en_name = author_obj.get("en", "").strip()
            he_name = author_obj.get("he", "").strip()

            if not slug:
                continue

            # Populate author profile with rich naming attributes
            cur.execute(
                """
                INSERT INTO authors (id, en_name, he_name)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    en_name = COALESCE(excluded.en_name, en_name),
                    he_name = COALESCE(excluded.he_name, he_name);
            """,
                (slug, en_name or slug, he_name or None),
            )
            authors_inserted += cur.rowcount

            # Bind relationship to the active book identifier
            cur.execute(
                """
                INSERT OR IGNORE INTO author_book_map (author_id, book_id)
                VALUES (?, ?);
            """,
                (slug, book_title),
            )
            links_inserted += cur.rowcount

    cur.execute("COMMIT;")

    # 4. Diagnostics Output
    cur.execute("SELECT COUNT(*) FROM authors;")
    total_authors = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM author_book_map;")
    total_links = cur.fetchone()[0]

    print(f"[+] Complete!")
    print(f"    -> Unique Authors parsed and tracked: {total_authors}")
    print(f"    -> Active relationships linked to books: {total_links}")

    conn.close()


if __name__ == "__main__":
    main()
