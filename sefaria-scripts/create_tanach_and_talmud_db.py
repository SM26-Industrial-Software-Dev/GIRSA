#!/usr/bi#!/usr/bin/env python3
"""
build_db.py

Ingests ONLY the pure base texts of Sefaria Tanakh and Talmud CLTK text files.
Excludes all commentaries, super-commentaries, and translations of commentaries.
"""

import argparse
import json
import logging
import re
import sqlite3
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS books (
    id              TEXT PRIMARY KEY,
    he_title        TEXT,
    en_title        TEXT,
    categories      TEXT,
    depth           INTEGER,
    address_types   TEXT,
    section_names   TEXT,
    he_section_names TEXT,
    is_commentary   INTEGER DEFAULT 0,
    base_text_title TEXT,
    collective_title TEXT,
    era             TEXT,
    comp_date_start INTEGER,
    comp_date_end   INTEGER,
    comp_place      TEXT
);

CREATE TABLE IF NOT EXISTS book_aliases (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    alias   TEXT NOT NULL,
    lang    TEXT NOT NULL CHECK(lang IN ('en','he','other')),
    alias_type TEXT DEFAULT 'title'
);
CREATE INDEX IF NOT EXISTS idx_alias_text ON book_aliases(alias);
CREATE INDEX IF NOT EXISTS idx_alias_book ON book_aliases(book_id);

CREATE TABLE IF NOT EXISTS versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id         TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    version_title   TEXT NOT NULL,
    language        TEXT NOT NULL,
    source_path     TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_version_unique ON versions(book_id, version_title, language);

CREATE TABLE IF NOT EXISTS passages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id     TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    addr1       TEXT,
    addr2       TEXT,
    addr3       TEXT,
    addr4       TEXT,
    sefaria_ref TEXT UNIQUE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_passages_book  ON passages(book_id);
CREATE INDEX IF NOT EXISTS idx_passages_ref   ON passages(sefaria_ref);

CREATE TABLE IF NOT EXISTS passage_texts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    passage_id  INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    version_id  INTEGER NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ptexts_passage  ON passage_texts(passage_id);
CREATE INDEX IF NOT EXISTS idx_ptexts_version  ON passage_texts(version_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ptexts_unique ON passage_texts(passage_id, version_id);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    passage_id      INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    version_id      INTEGER NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    text            TEXT NOT NULL,
    token_count     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_passage ON chunks(passage_id);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id    INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    vector      BLOB NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Strict Filter Helpers
# ---------------------------------------------------------------------------


def is_target_base_text(categories: list, dependence: str, book_id: str) -> bool:
    """
    Returns True ONLY for the exact base text of the Tanakh and Talmud.
    Blocks commentaries, translations (Targumim), super-commentaries, and modern additions.
    """
    if not categories:
        return False

    root = str(categories[0]).strip().lower()
    if root not in ("tanakh", "talmud"):
        return False

    # 1. Block explicit dependency flags (Commentaries, Translations/Targumim)
    if dependence:
        dep_lower = str(dependence).strip().lower()
        if dep_lower in ("commentary", "translation", "supercommentary"):
            return False

    # 2. Block structural category paths
    # This catches things tucked into sub-folders like "Tanakh/Commentary/" or "Talmud/Bavli/Commentary/"
    category_set = {str(c).strip().lower() for c in categories}
    blocked_categories = {"commentary", "targum", "translation", "modern", "ein yaakov"}
    if not category_set.isdisjoint(blocked_categories):
        return False

    # 3. Explicit Title Blacklist (The Ultimate Defense)
    # This catches rogue deep commentaries like Tziyyun LaNefesh, Rif, Rosh, Maharsha, etc.
    # that might have weird or incomplete category metadata.
    id_lower = str(book_id).strip().lower()
    blocked_keywords = [
        "targum",
        "onkelos",
        "jonathan",
        "pseudo-jonathan",
        "yerushalmi",
        "tziyyun",
        "lanefesh",
        "tzlach",
        "rashi",
        "tosafot",
        "maharsha",
        "maharam",
        "maharyf",
        "rif",
        "rosh",
        "ran",
        "rashba",
        "ritva",
        "pene mosh",
        "korban ha'edah",
        "chiddushei",
        "commentary",
        "biur",
    ]
    if any(kw in id_lower for kw in blocked_keywords):
        return False

    return True


# ---------------------------------------------------------------------------
# Address helpers
# ---------------------------------------------------------------------------


def talmud_index_to_daf(n: int) -> str:
    daf = (n // 2) + 1
    amud = "b" if n % 2 == 1 else "a"
    return f"{daf}{amud}"


def coord_to_str(address_types: list[str], depth_index: int, raw_index: int) -> str:
    if depth_index >= len(address_types):
        return str(raw_index + 1)
    atype = address_types[depth_index]
    if atype == "Talmud":
        return talmud_index_to_daf(raw_index)
    else:
        return str(raw_index + 1)


def build_sefaria_ref(
    book_title: str,
    address_types: list[str],
    coords: list[int],
    max_depth: int = None,
    section_type: str = "default",
) -> str:
    if section_type == "Introduction" and len(coords) >= 1:
        line_num = coords[-1] + 1
        return f"{book_title}, Introduction {line_num}"

    if max_depth is not None and len(coords) > max_depth:
        coords = coords[:max_depth]

    parts = [coord_to_str(address_types, i, c) for i, c in enumerate(coords)]
    is_talmud = any("Talmud" in str(t) for t in address_types)

    if is_talmud and len(parts) >= 1:
        daf_string = parts[0]
        if len(parts) > 1:
            return f"{book_title} {daf_string}:{parts[1]}"
        return f"{book_title} {daf_string}"

    if len(parts) > 1:
        return f"{book_title} {parts[0]}:{':'.join(parts[1:])}"
    elif len(parts) == 1:
        return f"{book_title} {parts[0]}"

    return book_title


# ---------------------------------------------------------------------------
# CLTK text flattening
# ---------------------------------------------------------------------------


def flatten_cltk(text_obj: dict, address_types: list[str], book_title: str):
    def extract_index(key_str: str) -> int:
        if key_str.isdigit():
            return int(key_str)
        match = re.search(r"\d+", key_str)
        if match:
            return int(match.group())
        return 0

    has_intro = any(
        "intro" in str(k).lower() or "הקדמה" in str(k) for k in text_obj.keys()
    )

    def _recurse(node, depth, coords, section_type):
        if isinstance(node, str):
            yield coords, node, section_type
            return
        if isinstance(node, list):
            yield coords, " ".join(str(s) for s in node if s), section_type
            return
        if isinstance(node, dict):
            sorted_keys = sorted(node.keys(), key=extract_index)
            for key in sorted_keys:
                child = node[key]
                raw_idx = extract_index(key)

                current_type = section_type
                if depth == 0:
                    if "intro" in key.lower() or "הקדמה" in key:
                        current_type = "Introduction"
                        raw_idx = 0
                    else:
                        current_type = "default"
                        if has_intro:
                            raw_idx = max(0, raw_idx - 1)

                yield from _recurse(child, depth + 1, coords + [raw_idx], current_type)

    yield from _recurse(text_obj, 0, [], "default")


def token_count(text: str) -> int:
    return len(text.split())


CHUNK_TARGET = 200
CHUNK_OVERLAP = 50


def make_chunks(text: str) -> list[str]:
    words = text.split()
    n = len(words)
    if n <= CHUNK_TARGET:
        return [text]
    chunks = []
    start = 0
    while start < n:
        end = min(start + CHUNK_TARGET, n)
        chunks.append(" ".join(words[start:end]))
        if end == n:
            break
        start += CHUNK_TARGET - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# Schema ingestion
# ---------------------------------------------------------------------------


def ingest_schemas(conn: sqlite3.Connection, schemas_dir: Path):
    log.info("Ingesting STRICT base texts for Tanakh & Talmud from %s", schemas_dir)
    files = list(schemas_dir.glob("*.json"))

    inserted = skipped = 0
    cur = conn.cursor()

    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning("Could not read %s: %s", fpath.name, e)
            continue

        title = data.get("title", "").strip()
        if not title:
            continue

        # --- THE STRICT FILTER PASS ---
        book_categories = data.get("categories", [])
        dependence = data.get("dependence", "")

        if not is_target_base_text(book_categories, dependence, title):
            skipped += 1
            continue

        schema = data.get("schema", {})
        categories_str = "|".join(book_categories)

        def get_deepest_node_meta(node):
            best_addr = node.get("addressTypes", [])
            best_sec = node.get("sectionNames", [])
            best_he = node.get("heSectionNames", [])
            best_depth = node.get("depth", 0)

            if "nodes" in node:
                for child in node["nodes"]:
                    c_addr, c_sec, c_he, c_depth = get_deepest_node_meta(child)
                    if len(c_addr) > len(best_addr):
                        best_addr, best_sec, best_he, best_depth = (
                            c_addr,
                            c_sec,
                            c_he,
                            c_depth,
                        )
            return best_addr, best_sec, best_he, best_depth

        he_title = data.get("heTitle", "") or schema.get("heTitle", "")
        address_types = data.get("addressTypes", [])
        section_names = data.get("sectionNames", [])
        he_section_names = data.get("heSectionNames", [])
        depth = data.get("depth", 1)

        if schema:
            s_addr, s_sec, s_he, s_depth = get_deepest_node_meta(schema)
            if len(s_addr) > len(address_types):
                address_types = s_addr
                section_names = s_sec
                he_section_names = s_he
                depth = s_depth or len(s_addr)

        is_commentary = 0  # Forced to 0 because we're explicitly blocking commentaries
        base_text_title = ""
        collective_title = ""

        comp_date = data.get("compDate", [])
        comp_date_start = comp_date[0] if len(comp_date) >= 1 else None
        comp_date_end = comp_date[1] if len(comp_date) >= 2 else comp_date_start

        try:
            cur.execute(
                """
                INSERT OR REPLACE INTO books (
                    id, he_title, en_title, categories, depth,
                    address_types, section_names, he_section_names,
                    is_commentary, base_text_title, collective_title,
                    era, comp_date_start, comp_date_end, comp_place
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
                (
                    title,
                    he_title,
                    title,
                    categories_str,
                    depth,
                    json.dumps(address_types, ensure_ascii=False),
                    json.dumps(section_names, ensure_ascii=False),
                    json.dumps(he_section_names, ensure_ascii=False),
                    is_commentary,
                    base_text_title,
                    collective_title,
                    data.get("era", ""),
                    comp_date_start,
                    comp_date_end,
                    data.get("compPlace", ""),
                ),
            )
            inserted += 1
        except sqlite3.Error as e:
            log.warning("DB error inserting book %s: %s", title, e)
            continue

        aliases: list[tuple[str, str, str]] = []
        for variant in data.get("titleVariants", []):
            aliases.append((variant, "en", "title"))
        for variant in data.get("heTitleVariants", []):
            aliases.append((variant, "he", "title"))

        for t in schema.get("titles", []):
            text_val = t.get("text", "").strip()
            lang = t.get("lang", "en")
            if text_val and lang in ("en", "he"):
                aliases.append((text_val, lang, "title"))

        for mt in schema.get("match_templates", []):
            slug_str = " ".join(mt.get("term_slugs", []))
            if slug_str:
                aliases.append((slug_str, "en", "match_template"))

        seen_aliases = set()
        for alias_text, lang, atype in aliases:
            key = (alias_text.lower(), lang)
            if key in seen_aliases or not alias_text.strip():
                continue
            seen_aliases.add(key)
            cur.execute(
                """
                INSERT OR IGNORE INTO book_aliases (book_id, alias, lang, alias_type)
                VALUES (?,?,?,?)
            """,
                (title, alias_text.strip(), lang, atype),
            )

    conn.commit()
    log.info(
        "Schemas complete: Kept %d pure base texts | Skipped %d commentary works.",
        inserted,
        skipped,
    )


# ---------------------------------------------------------------------------
# CLTK text ingestion
# ---------------------------------------------------------------------------


def ingest_texts(conn: sqlite3.Connection, cltk_dir: Path):
    log.info("Ingesting base text records from %s", cltk_dir)
    files = list(cltk_dir.rglob("*.json"))

    total_passages = total_chunks = text_files_skipped = 0

    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute("BEGIN TRANSACTION;")
    cur = conn.cursor()

    for file_count, fpath in enumerate(files, start=1):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning("Could not read %s: %s", fpath, e)
            continue

        book_title = data.get("work", "").strip()
        if not book_title:
            continue

        text_obj = data.get("text")
        if not text_obj:
            continue

        # Check if the book was approved in our pure base-text schema phase
        row = cur.execute(
            "SELECT address_types, depth FROM books WHERE id=?", (book_title,)
        ).fetchone()

        if row is None:
            text_files_skipped += 1
            continue

        address_types = json.loads(row[0]) if row[0] else ["Integer"]
        book_depth = row[1] if row[1] is not None else len(address_types)

        parts = fpath.parts
        try:
            lang = parts[-2]
            version_title = fpath.stem.replace("_", " ")
        except IndexError:
            lang = "unknown"
            version_title = fpath.stem

        cur.execute(
            """
            INSERT OR IGNORE INTO versions (book_id, version_title, language, source_path)
            VALUES (?,?,?,?)
        """,
            (book_title, version_title, lang, str(fpath.relative_to(cltk_dir.parent))),
        )

        version_id = cur.execute(
            "SELECT id FROM versions WHERE book_id=? AND version_title=? AND language=?",
            (book_title, version_title, lang),
        ).fetchone()[0]

        file_passages = 0
        file_chunks = 0

        for coords, leaf_text, sec_type in flatten_cltk(
            text_obj, address_types, book_title
        ):
            if not leaf_text or not leaf_text.strip():
                continue

            leaf_text = leaf_text.strip()
            sefaria_ref = build_sefaria_ref(
                book_title,
                address_types,
                coords,
                max_depth=book_depth,
                section_type=sec_type,
            )

            truncated_coords = coords[:book_depth]
            addr_vals = truncated_coords + [None] * (4 - len(truncated_coords))

            try:
                cursor = cur.execute(
                    """
                    INSERT INTO passages (book_id, addr1, addr2, addr3, addr4, sefaria_ref)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sefaria_ref) DO UPDATE SET 
                        book_id=excluded.book_id
                    RETURNING id;
                    """,
                    (
                        book_title,
                        str(addr_vals[0]) if addr_vals[0] is not None else None,
                        str(addr_vals[1]) if addr_vals[1] is not None else None,
                        str(addr_vals[2]) if addr_vals[2] is not None else None,
                        str(addr_vals[3]) if addr_vals[3] is not None else None,
                        sefaria_ref,
                    ),
                )
                passage_id = cursor.fetchone()[0]
            except sqlite3.Error as e:
                log.error(
                    f"Failed handling reference allocation for {sefaria_ref}: {e}"
                )
                continue

            cur.execute(
                """
                INSERT OR REPLACE INTO passage_texts (passage_id, version_id, text)
                VALUES (?,?,?)
            """,
                (passage_id, version_id, leaf_text),
            )

            text_chunks = make_chunks(leaf_text)
            for idx, chunk_text in enumerate(text_chunks):
                cur.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                        (passage_id, version_id, chunk_index, text, token_count)
                    VALUES (?,?,?,?,?)
                """,
                    (passage_id, version_id, idx, chunk_text, token_count(chunk_text)),
                )
                file_chunks += 1

            file_passages += 1

        total_passages += file_passages
        total_chunks += file_chunks

        if file_count % 200 == 0:
            conn.execute("COMMIT;")
            log.info("Processed %d files successfully...", file_count)
            conn.execute("BEGIN TRANSACTION;")

    conn.execute("COMMIT;")
    conn.execute("PRAGMA foreign_keys = ON;")
    log.info(
        "Texts complete: %d passages, %d chunks stored (%d commentary files blocked).",
        total_passages,
        total_chunks,
        text_files_skipped,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build targeted Sefaria SQLite database."
    )
    parser.add_argument(
        "--export-dir",
        default="sefaria_export",
        help="Root of the downloaded export (default: sefaria_export)",
    )
    parser.add_argument(
        "--db", default="sefaria.db", help="Output SQLite file (default: sefaria.db)"
    )
    parser.add_argument("--schemas-only", action="store_true")
    parser.add_argument("--texts-only", action="store_true")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    schemas_dir = export_dir / "schemas"
    cltk_dir = export_dir / "cltk-full"

    conn = sqlite3.connect(args.db)

    conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -1000000;
    """)

    conn.executescript(DDL)
    conn.commit()

    if not args.texts_only:
        if not schemas_dir.exists():
            log.error("schemas/ directory not found under %s", export_dir)
        else:
            ingest_schemas(conn, schemas_dir)

    if not args.schemas_only:
        if not cltk_dir.exists():
            log.error("cltk-full/ directory not found under %s", export_dir)
        else:
            ingest_texts(conn, cltk_dir)

    conn.close()
    log.info("Done. Database written to %s", args.db)


if __name__ == "__main__":
    main()
