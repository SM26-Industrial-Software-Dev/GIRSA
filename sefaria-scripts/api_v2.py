#!/usr/bin/env python3
"""
main.py

Unified Sefaria Semantic Search Engine.
Acts as both a standalone CLI search script and a high-performance FastAPI REST API.

CLI Usage:
    python main.py --query "Don't kill" --debug
Use --debug to print the results before and after cross encoding

API Server Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import argparse
import logging
import os
import sqlite3
import struct
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    torch.set_num_threads(1)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Logging & Environmental Configurations
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "sefaria.db")
INDEX_PATH = os.getenv("INDEX_PATH", "sefaria.index")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/LaBSE")


# ---------------------------------------------------------------------------
# Vector Vectorization Helper
# ---------------------------------------------------------------------------
def blob_to_vec(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.array(struct.unpack(f"{n}f", b), dtype=np.float32)


# ---------------------------------------------------------------------------
# Core Engine Logic
# ---------------------------------------------------------------------------
class SefariaSearcher:
    def __init__(self, db_path: str, index_path: str, model_name: str = "LaBSE",
                 reranker_name: str = "BAAI/bge-reranker-v2-m3"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA cache_size = -32000")

        log.info("Loading FAISS index from %s ...", index_path)
        self.index = faiss.read_index(index_path)
        self.id_map = np.load(index_path + ".idmap.npy")

        log.info("Loading model %s ...", model_name)
        self.model = SentenceTransformer(model_name, device="cpu", trust_remote_code=False)
        self.model.eval()

        log.info("Loading reranker %s ...", reranker_name)
        self.reranker = CrossEncoder(reranker_name, device="cpu", max_length=512)

        if isinstance(self.index, faiss.IndexIVFPQ):
            self.index.nprobe = 16

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            text,
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32).reshape(1, -1)

    def search(
        self,
        query: str,
        top_k: int = 10,
        book_filter: str = None,
        category_filter: str = None,
        unique: bool = False,
        weighted: bool = False,
        addresses=None,
        rerank: bool = True,
        rerank_pool_size: int = 50,
        debug: bool = False,
    ):
        query_vec = self.embed_query(query)

        needs_overfetch = addresses or book_filter or category_filter or unique or rerank
        fetch_k = max(rerank_pool_size, top_k) if needs_overfetch else top_k
        if addresses or book_filter or category_filter:
            fetch_k = max(fetch_k, 10000)

        scores, positions = self.index.search(query_vec, fetch_k)
        scores = scores[0]
        positions = positions[0]

        PRIORITY_BOOSTS = {
            "Tanakh": 1.05,
            "Talmud": 1.03,
            "Mishnah": 1.02,
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

            if addresses and len(addresses) == 4:
                ra1 = row["addr1"] if row["addr1"] is not None else -1
                ra2 = row["addr2"] if row["addr2"] is not None else -1
                ra3 = row["addr3"] if row["addr3"] is not None else -1
                ra4 = row["addr4"] if row["addr4"] is not None else -1
                if addresses[0] != -1 and addresses[0] != ra1:
                    continue
                if addresses[1] != -1 and addresses[1] != ra2:
                    continue
                if addresses[2] != -1 and addresses[2] != ra3:
                    continue
                if addresses[3] != -1 and addresses[3] != ra4:
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

        # Sort by bi-encoder score first, then take a pool for re-ranking
        results.sort(key=lambda x: x["score"], reverse=True)
        results = self._dedupe_by_ref(results)

        if rerank and results:
            pool = results[:rerank_pool_size]
            pool = self._rerank(query, pool, debug=debug)
            return pool[:top_k]

        return results[:top_k]

    def _rerank(self, query: str, results: list, debug: bool = False) -> list:
        if not results:
            return results

        pairs = [(query, r["text"]) for r in results]
        ce_scores = self.reranker.predict(pairs, show_progress_bar=False)

        for r, ce_score in zip(results, ce_scores):
            r["bi_encoder_score"] = r["score"]
            r["score"] = float(ce_score)

        if debug:
            print(f"\n[RERANK DEBUG] query={query!r}  pool_size={len(results)}")
            print(f"{'REF':<30} {'BI-SCORE':>10} {'CE-SCORE':>10}  TEXT")
            for r in results:
                snippet = r["text"][:80].replace("\n", " ")
                print(f"{r['sefaria_ref']:<30} {r['bi_encoder_score']:>10.4f} {r['score']:>10.4f}  {snippet}")

        results.sort(key=lambda x: x["score"], reverse=True)

        if debug:
            print(f"\n[RERANK DEBUG] order AFTER re-ranking:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['sefaria_ref']}  ce_score={r['score']:.4f}")
            print()

        return results

    def _resolve_chunk(
        self, chunk_id: int, book_filter: str = None, category_filter: str = None
    ):
        row = self.conn.execute(
            """
            SELECT
                c.id, c.text, c.chunk_index,
                p.sefaria_ref, p.book_id,
                b.he_title, b.en_title, b.categories,
                v.language, v.version_title, p.addr1, p.addr2, p.addr3, p.addr4
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
            "categories": row[7],
            "language": row[8],
            "version_title": row[9],
            "sefaria_url": f"https://www.sefaria.org/{row[3].replace(' ', '_')}",
            "addr1": row[10],
            "addr2": row[11],
            "addr3": row[12],
            "addr4": row[13],
        }

        if category_filter:
            clean_cats = [c.strip().lower() for c in result["categories"].split("|")]
            if category_filter.strip().lower() not in clean_cats:
                return None

        if book_filter:
            clean_f = book_filter.strip().lower()
            if (
                clean_f != result["book_id"].lower()
                and clean_f != result["en_title"].lower()
            ):
                return None

        return result
    
    def _dedupe_by_ref(self, results: list, prefer_version: str = None) -> list:
        best_by_ref = {}
        for r in results:
            ref = r["sefaria_ref"]
            if ref not in best_by_ref:
                best_by_ref[ref] = r
            elif prefer_version and r["version_title"] == prefer_version:
                best_by_ref[ref] = r
            elif r["score"] > best_by_ref[ref]["score"]:
                best_by_ref[ref] = r
        return list(best_by_ref.values())


# ---------------------------------------------------------------------------
# FastAPI REST Interface
# ---------------------------------------------------------------------------
searcher: Optional[SefariaSearcher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global searcher
    log.info("Loading SefariaSearcher (model=%s)...", MODEL_NAME)
    searcher = SefariaSearcher(DB_PATH, INDEX_PATH, MODEL_NAME)
    log.info("Ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Sefaria Semantic Search",
    description="Semantic search over Sefaria texts for shiur citation hyperlinking.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=100)
    book_filter: Optional[str] = None
    category_filter: Optional[str] = None
    unique: Optional[bool] = False
    weighted: Optional[bool] = False
    addresses: Optional[list[int]] = [-1, -1, -1, -1]
    rerank: Optional[bool] = True
    rerank_pool_size: Optional[int] = Field(100, ge=10, le=500)


class SearchResult(BaseModel):
    chunk_id: int
    score: float
    sefaria_ref: str
    sefaria_url: str
    book_id: str
    en_title: str
    he_title: str
    categories: str
    language: str
    version_title: str
    text: str
    chunk_index: int
    addr1: Optional[int] = None
    addr2: Optional[int] = None
    addr3: Optional[int] = None
    addr4: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


class BookEntry(BaseModel):
    id: str
    en_title: str
    he_title: str
    categories: str


class CategoryEntry(BaseModel):
    category: str
    count: int


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "index": INDEX_PATH}


@app.post("/search", response_model=SearchResponse)
def search_api(req: SearchRequest):
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")
    try:
        raw = searcher.search(
            query=req.query,
            top_k=req.top_k,
            book_filter=req.book_filter,
            category_filter=req.category_filter,
            unique=req.unique,
            weighted=req.weighted,
            addresses=req.addresses,
            rerank=req.rerank,
            rerank_pool_size=req.rerank_pool_size,
        )
    except Exception as e:
        log.exception("Search failed for query %r", req.query)
        raise HTTPException(status_code=500, detail=str(e))

    results = [SearchResult(**r) for r in raw]
    return SearchResponse(query=req.query, results=results, total=len(results))


@app.get("/books", response_model=list[BookEntry])
def list_books(
    q: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    sql = "SELECT id, en_title, he_title, categories FROM books WHERE 1=1"
    params = []

    if q:
        sql += " AND (en_title LIKE ? OR he_title LIKE ? OR id LIKE ?)"
        like = f"%{q}%"
        params += [like, like, like]

    if category:
        sql += " AND categories LIKE ?"
        params.append(f"%{category}%")

    sql += " ORDER BY en_title LIMIT ?"
    params.append(limit)

    rows = searcher.conn.execute(sql, params).fetchall()
    return [
        BookEntry(
            id=r[0], en_title=r[1] or "", he_title=r[2] or "", categories=r[3] or ""
        )
        for r in rows
    ]


@app.get("/categories", response_model=list[CategoryEntry])
def list_categories():
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    rows = searcher.conn.execute("""
        SELECT
            SUBSTR(categories, 1,
                   CASE WHEN INSTR(categories, '|') > 0
                        THEN INSTR(categories, '|') - 1
                        ELSE LENGTH(categories) END
            ) AS top_cat,
            COUNT(*) AS cnt
        FROM books
        GROUP BY top_cat
        ORDER BY cnt DESC
    """).fetchall()
    return [CategoryEntry(category=r[0], count=r[1]) for r in rows]


# ---------------------------------------------------------------------------
# CLI Thread Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sefaria Local Search CLI Toolkit")
    parser.add_argument("--db", default="sefaria_ta.db")
    parser.add_argument("--index", default="sefaria_ta.index")
    parser.add_argument("--model", default="LaBSE")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--book-filter", default=None)
    parser.add_argument("--category-filter", default=None)
    parser.add_argument("--unique", "-u", action="store_true", default=False)
    parser.add_argument("--weighted", "-w", action="store_true", default=False)
    parser.add_argument("--rerank", action="store_true", default=True)
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("--rerank-pool-size", type=int, default=50)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--addrs", "-a", default="-1.-1.-1.-1")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Database file not found: {args.db}")
        exit(1)
    if not Path(args.index).exists():
        print(f"Index file not found: {args.index}")
        exit(1)

    addresses = [int(x) for x in args.addrs.split(".")]
    # Initialize a short-lived instance directly for the terminal session
    cli_searcher = SefariaSearcher(args.db, args.index, args.model)
    results = cli_searcher.search(
        args.query,
        top_k=args.top_k,
        book_filter=args.book_filter,
        category_filter=args.category_filter,
        unique=args.unique,
        weighted=args.weighted,
        addresses=addresses,
        rerank=args.rerank,
        rerank_pool_size=args.rerank_pool_size,
        debug=args.debug,
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
