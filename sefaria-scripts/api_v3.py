#!/usr/bin/env python3
"""
api_v3.py

Unified Sefaria Semantic Search Engine - Cascading Popularity Edition.
Searches through the most popular books first, expanding the search pool 
in tiers (e.g., 500 books at a time) until a high-quality result is found.

CLI Usage:
    python api_v3.py --query "In the beginning God created" --debug
"""

import argparse
import logging
import os
import sqlite3
import struct
import pandas as pd
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "sefaria.db")
INDEX_PATH = os.getenv("INDEX_PATH", "sefaria.index")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/LaBSE")


def blob_to_vec(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.array(struct.unpack(f"{n}f", b), dtype=np.float32)


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

        # Load PageRank scores and sort books by absolute popularity tiering order
        self.ranked_books = []
        pagerank_path = os.getenv("PAGERANK_PATH", "accurate_sefaria_pagerank.csv")

        if os.path.exists(pagerank_path):
            log.info("Loading PageRank scores from %s ...", pagerank_path)
            try:
                df_pr = pd.read_csv(pagerank_path)
                df_pr = df_pr.sort_values(by="PageRank Score", ascending=False)
                # Matches exact formatting from api_v2.py configuration
                self.ranked_books = [b.strip() for b in df_pr['Book Title'].tolist()]
                log.info("Loaded %d books sorted by PageRank for cascading search.", len(self.ranked_books))
            except Exception as e:
                log.error("Failed to load PageRank file: %s. Cascading search will be disabled.", e)
        else:
            log.warning("PageRank file not found at %s. Cascading search disabled.", pagerank_path)

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
        cascade_popularity: bool = True,
        tier_size: int = 500,
        good_score_threshold: float = 0.5, 
        addresses=None,
        rerank: bool = True,
        rerank_pool_size: int = 100,
        debug: bool = False,
    ):
        query_vec = self.embed_query(query)
        fetch_k = 20000  # Pull large context candidate window from FAISS
        
        scores, positions = self.index.search(query_vec, fetch_k)
        scores = scores[0]
        positions = positions[0]

        valid_idx = positions != -1
        retrieved_positions = positions[valid_idx]
        retrieved_scores = scores[valid_idx]

        # Convert to explicit native Python types to guarantee exact database/type matching
        chunk_score_pairs = []
        for pos, sc in zip(retrieved_positions, retrieved_scores):
            chunk_score_pairs.append((int(self.id_map[pos]), float(sc)))

        # Dynamic batch lookup of book_ids for retrieved chunks only.
        # This completely replaces the slow 10-million row mapping on startup.
        retrieved_chunk_book_map = {}
        cids = [p[0] for p in chunk_score_pairs]
        
        # Batch queries into segments of 900 to safely respect SQLite parameters limitations
        for i in range(0, len(cids), 900):
            batch = cids[i:i+900]
            placeholders = ",".join(["?"] * len(batch))
            rows = self.conn.execute(f"""
                SELECT c.id, p.book_id 
                FROM chunks c 
                JOIN passages p ON c.passage_id = p.id 
                WHERE c.id IN ({placeholders})
            """, batch).fetchall()
            for r in rows:
                retrieved_chunk_book_map[r[0]] = r[1].strip()

        do_cascade = cascade_popularity and len(self.ranked_books) > 0
        current_tier = 1
        max_tiers = (len(self.ranked_books) // tier_size) + 1 if do_cascade else 1

        final_results = []

        while current_tier <= max_tiers:
            if do_cascade:
                limit = current_tier * tier_size
                allowed_books = set(self.ranked_books[:limit])
            else:
                allowed_books = None
                
            tier_candidates_metadata = []
            for cid, sc in chunk_score_pairs:
                book_id = retrieved_chunk_book_map.get(cid)
                if not book_id: 
                    continue
                if allowed_books is None or book_id in allowed_books:
                    tier_candidates_metadata.append((cid, sc))
            
            best_score = None
            if tier_candidates_metadata:
                resolve_limit = max(rerank_pool_size * 2, top_k * 5)
                tier_candidates = []
                seen = set()

                for cid, sc in tier_candidates_metadata:
                    if len(tier_candidates) >= resolve_limit:
                        break
                        
                    row = self._resolve_chunk(cid, book_filter, category_filter)
                    if row is None: 
                        continue

                    if unique and row["sefaria_ref"] in seen: 
                        continue

                    if addresses and len(addresses) == 4:
                        ra1 = row["addr1"] if row["addr1"] is not None else -1
                        ra2 = row["addr2"] if row["addr2"] is not None else -1
                        ra3 = row["addr3"] if row["addr3"] is not None else -1
                        ra4 = row["addr4"] if row["addr4"] is not None else -1
                        if addresses[0] != -1 and addresses[0] != ra1: continue
                        if addresses[1] != -1 and addresses[1] != ra2: continue
                        if addresses[2] != -1 and addresses[2] != ra3: continue
                        if addresses[3] != -1 and addresses[3] != ra4: continue
                            
                    row["score"] = sc
                    tier_candidates.append(row)

                    if unique:
                        seen.add(row["sefaria_ref"])
                
                tier_candidates.sort(key=lambda x: x["score"], reverse=True)
                tier_candidates = self._dedupe_by_ref(tier_candidates)

                if rerank and tier_candidates:
                    pool = tier_candidates[:tier_size]
                    pool = self._rerank(query, pool, debug=debug)
                    final_results = pool[:top_k]
                else:
                    final_results = tier_candidates[:top_k]
                    
                if final_results:
                    best_score = final_results[0]["score"]
                    is_good = best_score >= good_score_threshold
                    
                    if is_good or not do_cascade:
                        if debug and do_cascade:
                            print(f"\n[CASCADE DEBUG] Success in Tier {current_tier} (Top {current_tier*tier_size} books). Best Score: {best_score:.4f} >= {good_score_threshold}")
                        return final_results
            else:
                final_results = []
            
            if debug and do_cascade:
                best_str = f"{best_score:.4f}" if best_score is not None else "None"
                print(f"[CASCADE DEBUG] Tier {current_tier} failed. Best Score was {best_str}. Expanding to Tier {current_tier+1}...")
            
            current_tier += 1

        return final_results

    def _rerank(self, query: str, results: list, debug: bool = False) -> list:
        if not results:
            return results

        pairs = [(query, r["text"]) for r in results]
        ce_scores = self.reranker.predict(pairs, show_progress_bar=False)

        for r, ce_score in zip(results, ce_scores):
            r["bi_encoder_score"] = r["score"]
            r["score"] = float(ce_score)

        results.sort(key=lambda x: x["score"], reverse=True)

        if debug:
            print(f"\n[RERANK DEBUG] Order AFTER re-ranking:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['sefaria_ref']}  ce_score={r['score']:.4f} {r['categories']}")
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
    title="Sefaria Semantic Search (Cascading Edition)",
    description="Semantic search over Sefaria texts prioritized by popularity tiers.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=100)
    book_filter: Optional[str] = None
    category_filter: Optional[str] = None
    unique: Optional[bool] = False
    cascade_popularity: Optional[bool] = True
    tier_size: int = Field(500, ge=100, le=5000)
    good_score_threshold: float = 0.0
    addresses: Optional[list[int]] = [-1, -1, -1, -1]
    rerank: Optional[bool] = True
    rerank_pool_size: Optional[int] = Field(25, ge=10, le=500)


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
            cascade_popularity=req.cascade_popularity,
            tier_size=req.tier_size,
            good_score_threshold=req.good_score_threshold,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sefaria Local Search CLI Toolkit (Cascading Edition)")
    parser.add_argument("--db", default="sefaria.db")
    parser.add_argument("--index", default="sefaria.index")
    parser.add_argument("--model", default="LaBSE")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--book-filter", default=None)
    parser.add_argument("--category-filter", default=None)
    parser.add_argument("--unique", "-u", action="store_true", default=False)
    
    parser.add_argument("--no-cascade", action="store_false", dest="cascade", help="Disable tier-based cascading search")
    parser.add_argument("--tier-size", type=int, default=750, help="Number of books to include per cascade tier")
    parser.add_argument("--threshold", type=float, default=0.5, help="Reranker score threshold to stop expanding tiers")
    
    parser.add_argument("--rerank", action="store_true", default=True)
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("--rerank-pool-size", type=int, default=100)
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
    cli_searcher = SefariaSearcher(args.db, args.index, args.model)
    results = cli_searcher.search(
        args.query,
        top_k=args.top_k,
        book_filter=args.book_filter,
        category_filter=args.category_filter,
        unique=args.unique,
        cascade_popularity=args.cascade,
        tier_size=args.tier_size,
        good_score_threshold=args.threshold,
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