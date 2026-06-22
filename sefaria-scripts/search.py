#!/usr/bin/env python3
"""
search.py

Embeds a query string with LaBSE and searches the FAISS index for the
nearest chunks, then resolves them back to full passage / book / ref info
from sefaria.db.

Usage:
    python search.py --query "בראשית ברא אלהים את השמים"
    python search.py --query "in the beginning God created" --top-k 10
    python search.py --query "..." --book-filter Genesis   # restrict to one book
"""

import argparse
import sqlite3
import struct
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def blob_to_vec(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.array(struct.unpack(f"{n}f", b), dtype=np.float32)


class SefariaSearcher:
    def __init__(self, db_path: str, index_path: str, model_name: str = "LaBSE"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA cache_size = -32000")

        print(f"Loading FAISS index from {index_path} ...")
        self.index = faiss.read_index(index_path)
        self.id_map = np.load(index_path + ".idmap.npy")

        print(f"Loading model {model_name} ...")
        self.model = SentenceTransformer(model_name)

        # If IVF index, set nprobe for the speed/accuracy tradeoff
        if isinstance(self.index, faiss.IndexIVFPQ):
            self.index.nprobe = 16  # higher = more accurate, slower

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32).reshape(1, -1)

    def search(
        self,
        query: str,
        top_k: int = 10,
        book_filter: str = None,
        category_filter: str = None,
        unique: bool = None,
        weighted: bool = None,
    ):
        query_vec = self.embed_query(query)

        # Over-fetch if filtering by book, since some results will be discarded
        fetch_k = 10000 if book_filter or category_filter or unique else top_k

        scores, positions = self.index.search(query_vec, fetch_k)
        scores = scores[0]
        positions = positions[0]
        PRIORITY_BOOSTS = {
            "Tanakh": 1.05,  # 5% boost to biblical text
            "Talmud": 1.03,  # 3% boost to the Talmud
            "Mishnah": 1.02,  # 2% boost to the Mishnah
            # Commentaries, Liturgy, Halakha defaults to 1.0 (no boost)
        }
        results = []
        seen = set()
        for score, pos in zip(scores, positions):
            if pos == -1:
                continue
            chunk_id = int(self.id_map[pos])
            row = self._resolve_chunk(chunk_id, book_filter, category_filter)
            if row is None:
                continue
            if unique and row["sefaria_ref"] in seen:
                continue
            boost = 1.0
            if weighted:
                for cat, multiplier in PRIORITY_BOOSTS.items():
                    if cat in row["categories"]:
                        boost = max(boost, multiplier)
            row["score"] = float(score) * boost
            results.append(row)
            if unique:
                seen.add(row["sefaria_ref"])
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _resolve_chunk(
        self, chunk_id: int, book_filter: str = None, category_filter: str = None
    ):
        row = self.conn.execute(
            """
            SELECT
                c.id, c.text, c.chunk_index,
                p.sefaria_ref, p.book_id,
                b.he_title, b.en_title, b.categories,
                v.language, v.version_title
            FROM chunks c
            JOIN passages p ON c.passage_id = p.id
            JOIN books b ON p.book_id = b.id
            JOIN versions v ON c.version_id = v.id
            WHERE c.id = ?
        """,
            (chunk_id,),
        ).fetchone()

        if row is None:
            return None

        result = {
            "chunk_id": row[0],
            "text": row[1],
            "chunk_index": row[2],
            "sefaria_ref": row[3],
            "book_id": row[4],
            "he_title": row[5],
            "en_title": row[6],
            "categories": row[7],  # e.g., "Talmud|Bavli|Seder Nezikin"
            "language": row[8],
            "version_title": row[9],
            "sefaria_url": f"https://www.sefaria.org/{row[3].replace(' ', '_')}",
        }

        # Handle Pipe-Delimited Category Filtering cleanly
        if category_filter:
            # Split "Talmud|Bavli" into ["talmud", "bavli"] for an exact match check
            clean_cats = [c.strip().lower() for c in result["categories"].split("|")]
            if category_filter.strip().lower() not in clean_cats:
                return None

        # Resilient substring match for books
        if book_filter:
            clean_f = book_filter.strip().lower()
            if (
                clean_f not in result["book_id"].lower()
                and clean_f not in result["en_title"].lower()
            ):
                return None

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sefaria.db")
    parser.add_argument("--index", default="sefaria.index")
    parser.add_argument("--model", default="LaBSE")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--book-filter", default=None)
    parser.add_argument("--category-filter", default=None)
    parser.add_argument("--unique", "-u", action="store_true", default=False)
    parser.add_argument("--weighted", "-w", action="store_true", default=False)
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        return
    if not Path(args.index).exists():
        print(f"Index not found: {args.index}. Run build_faiss_index.py first.")
        return

    searcher = SefariaSearcher(args.db, args.index, args.model)
    results = searcher.search(
        args.query,
        top_k=args.top_k,
        book_filter=args.book_filter,
        category_filter=args.category_filter,
        unique=args.unique,
    )

    print(f"\nQuery: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        print(
            f"{i}. [{r['score']:.3f}] {r['sefaria_ref']}  ({r['en_title']}, {r['language']})"
        )
        print(f"   Version: {r['version_title']}")
        print(f"   {r['text'][:150]}")
        print(f"   {r['sefaria_url']}")
        print()


if __name__ == "__main__":
    main()
