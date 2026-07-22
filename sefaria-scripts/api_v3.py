#!/usr/bin/env python3
"""
api_v3.py

Unified Sefaria Semantic Search Engine - Bucket Cascade Edition

CLI Usage:
    python api_v3.py --query "In the beginning God created" --book-prefix "Shulchan Arukh"
"""

import argparse
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import numpy as np
import pandas as pd
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


class SefariaSearcher:
    def __init__(
        self,
        db_path: str,
        index_path: str,
        model_name: str = "LaBSE",
        reranker_name: str = "BAAI/bge-reranker-v2-m3",
    ):
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

        # Caches to eliminate redundant SQL lookups
        self.MAX_CACHE_SIZE = 100000
        self.chunk_cache = {}
        self.cid_to_book_cache = {}

        # Precomputed Popularity Tier Map
        self.book_to_rank = {}
        pagerank_path = os.getenv("PAGERANK_PATH", "accurate_sefaria_pagerank.csv")

        if os.path.exists(pagerank_path):
            log.info("Loading PageRank scores from %s ...", pagerank_path)
            try:
                df_pr = pd.read_csv(pagerank_path)
                df_pr = df_pr.sort_values(by="PageRank Score", ascending=False)
                ranked_books = [b.strip() for b in df_pr["Book Title"].tolist()]
                self.book_to_rank = {book: idx for idx, book in enumerate(ranked_books)}
                log.info("Loaded %d books mapped by PageRank.", len(self.book_to_rank))
            except Exception as e:
                log.error("Failed to load PageRank file: %s. Cascade disabled.", e)
        else:
            log.warning("PageRank file not found at %s. Cascade disabled.", pagerank_path)

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

    # -----------------------------------------------------------------------
    # Phase 2: Batched SQL & Cache Lookups
    # -----------------------------------------------------------------------
    def _get_cid_books(self, cids: list[int]) -> dict[int, str]:
        """Fetch book_ids for given chunk_ids in a single batched query."""
        result = {}
        missing = []
        for cid in cids:
            if cid in self.cid_to_book_cache:
                result[cid] = self.cid_to_book_cache[cid]
            else:
                missing.append(cid)

        if missing:
            for i in range(0, len(missing), 900):
                batch = missing[i : i + 900]
                placeholders = ",".join(["?"] * len(batch))
                rows = self.conn.execute(
                    f"""
                    SELECT c.id, p.book_id 
                    FROM chunks c 
                    JOIN passages p ON c.passage_id = p.id 
                    WHERE c.id IN ({placeholders})
                """,
                    batch,
                ).fetchall()
                for cid, book_id in rows:
                    clean_book_id = book_id.strip()
                    result[cid] = clean_book_id
                    self.cid_to_book_cache[cid] = clean_book_id

        return result

    def _get_metadata(self, cids: list[int]) -> dict[int, dict]:
        """Fetch full chunk metadata natively batched with a local cache."""
        result = {}
        missing = []
        for cid in cids:
            if cid in self.chunk_cache:
                result[cid] = self.chunk_cache[cid]
            else:
                missing.append(cid)

        if missing:
            for i in range(0, len(missing), 900):
                batch = missing[i : i + 900]
                placeholders = ",".join(["?"] * len(batch))
                rows = self.conn.execute(
                    f"""
                    SELECT
                        c.id, c.text, c.chunk_index,
                        p.sefaria_ref, p.book_id,
                        b.he_title, b.en_title, b.categories,
                        v.language, v.version_title, p.addr1, p.addr2, p.addr3, p.addr4
                    FROM chunks c
                    JOIN passages p ON c.passage_id = p.id
                    JOIN books b ON p.book_id = b.id
                    JOIN versions v ON c.version_id = v.id
                    WHERE c.id IN ({placeholders})
                """,
                    batch,
                ).fetchall()

                for row in rows:
                    cid = row[0]
                    data = {
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
                    result[cid] = data

                    if len(self.chunk_cache) > self.MAX_CACHE_SIZE:
                        # Clear oldest 10% of entries
                        for _ in range(self.MAX_CACHE_SIZE // 10):
                            self.chunk_cache.pop(next(iter(self.chunk_cache)))

                    self.chunk_cache[cid] = data

        return result

    # -----------------------------------------------------------------------
    # Phase 4: Core Modular Engine Methods
    # -----------------------------------------------------------------------
    def retrieve_candidates(self, query_vec: np.ndarray, fetch_k: int) -> list[tuple[int, float]]:
        scores, positions = self.index.search(query_vec, fetch_k)
        valid_idx = positions[0] != -1
        cids = [int(self.id_map[p]) for p in positions[0][valid_idx]]
        retrieved_scores = scores[0][valid_idx]
        return list(zip(cids, retrieved_scores))

    def bucket_candidates(self, chunk_score_pairs: list, cascade: bool, tier_size: int) -> dict:
        """Assigns every candidate to a tier exactly once."""
        buckets = defaultdict(list)
        cids = [p[0] for p in chunk_score_pairs]
        cid_books = self._get_cid_books(cids)

        do_cascade = cascade and len(self.book_to_rank) > 0
        total_books = len(self.book_to_rank)

        for cid, sc in chunk_score_pairs:
            book_id = cid_books.get(cid)
            if not book_id:
                continue

            if do_cascade:
                rank = self.book_to_rank.get(book_id)
                if rank is not None:
                    tier = (rank // tier_size) + 1
                else:
                    tier = (total_books // tier_size) + 2  # Lowest priority bucket
            else:
                tier = 1

            buckets[tier].append((cid, sc))

        return buckets

    def _passes_filters(
        self,
        row: dict,
        book_filter: str,
        category_filter: str,
        addresses: list,
        book_prefix_filter: str = None,
    ) -> bool:
        if category_filter:
            clean_cats = [c.strip().lower() for c in row["categories"].split("|")]
            if category_filter.strip().lower() not in clean_cats:
                return False

        if book_filter:
            clean_f = book_filter.strip().lower()
            if clean_f != row["book_id"].lower() and clean_f != row["en_title"].lower():
                return False

        if book_prefix_filter:
            clean_p = book_prefix_filter.strip().lower()
            in_book_id = clean_p in row["book_id"].lower()
            in_en_title = clean_p in row["en_title"].lower()
            in_he_title = clean_p in (row.get("he_title") or "").lower()
            if not (in_book_id or in_en_title or in_he_title):
                return False

        if addresses and len(addresses) == 4:
            ra1 = row["addr1"] if row["addr1"] is not None else -1
            ra2 = row["addr2"] if row["addr2"] is not None else -1
            ra3 = row["addr3"] if row["addr3"] is not None else -1
            ra4 = row["addr4"] if row["addr4"] is not None else -1
            if addresses[0] != -1 and addresses[0] != ra1:
                return False
            if addresses[1] != -1 and addresses[1] != ra2:
                return False
            if addresses[2] != -1 and addresses[2] != ra3:
                return False
            if addresses[3] != -1 and addresses[3] != ra4:
                return False

        return True

    def _dedupe_by_ref(self, results: list) -> list:
        best_by_ref = {}
        for r in results:
            ref = r["sefaria_ref"]
            if ref not in best_by_ref:
                best_by_ref[ref] = r
            elif r["score"] > best_by_ref[ref]["score"]:
                best_by_ref[ref] = r
        return list(best_by_ref.values())

    def _rerank_candidates(self, query: str, results: list, cache: dict, debug: bool = False) -> list:
        """Cached CrossEncoder reranking to prevent recomputing the same chunks."""
        if not results:
            return results

        # Apply already known scores from the cache
        for r in results:
            if r["chunk_id"] in cache:
                r["ce_score"] = cache[r["chunk_id"]]

        pairs_to_eval = [(query, r["text"]) for r in results if "ce_score" not in r]

        if pairs_to_eval:
            ce_scores = self.reranker.predict(pairs_to_eval, show_progress_bar=False)
            score_idx = 0
            for r in results:
                if "ce_score" not in r:
                    r["bi_encoder_score"] = r["score"]
                    r["ce_score"] = float(ce_scores[score_idx])
                    # Store the newly calculated score in the cache
                    cache[r["chunk_id"]] = r["ce_score"]
                    score_idx += 1

        for r in results:
            r["score"] = r["ce_score"]

        results.sort(key=lambda x: x["score"], reverse=True)

        if debug:
            print(f"\n[RERANK DEBUG] Top 3 re-ranked scores:")
            for i, r in enumerate(results[:3], 1):
                prob = float(torch.sigmoid(torch.tensor(r["score"])).item())
                print(f"  {i}. {r['sefaria_ref']}  score={r['score']:.4f} (prob={prob:.4f})")

        return results

    # -----------------------------------------------------------------------
    # Phase 1 & 3: Master Orchestrator (Adaptive Fetch + Buckets + Continuous Queue Reranking)
    # -----------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        book_filter: str = None,
        book_prefix_filter: str = None,
        category_filter: str = None,
        unique: bool = False,
        cascade_popularity: bool = True,
        tier_size: int = 500,
        good_score_threshold: float = 0.5,
        addresses=None,
        rerank: bool = True,
        rerank_pool_size: int = 40,
        max_rerank_budget: int = 200,
        debug: bool = False,
    ):
        query_vec = self.embed_query(query)

        fetch_steps = [4000, 10000, 20000]

        global_evaluated_chunks = {}

        unreranked_queue = []

        query_ce_cache = {}
        
        total_reranked = 0

        for fetch_k in fetch_steps:
            if debug:
                print(f"\n[FETCH TIER] Querying FAISS for top {fetch_k} candidates...")

            chunk_score_pairs = self.retrieve_candidates(query_vec, fetch_k)
            buckets = self.bucket_candidates(chunk_score_pairs, cascade_popularity, tier_size)

            max_tier = max(buckets.keys()) if buckets else 1

            for tier in range(1, max_tier + 1):
                if tier not in buckets:
                    continue

                # Only look at chunks in this tier we haven't already fully evaluated
                new_tier_pairs = [
                    (cid, sc) for cid, sc in buckets[tier] if cid not in global_evaluated_chunks
                ]
                if not new_tier_pairs:
                    continue

                tier_cids = [cid for cid, _ in new_tier_pairs]
                metadata_dict = self._get_metadata(tier_cids)

                tier_resolved = []
                for cid, sc in new_tier_pairs:
                    row = metadata_dict.get(cid)
                    if not row:
                        continue

                    if not self._passes_filters(
                        row, book_filter, category_filter, addresses, book_prefix_filter
                    ):
                        continue

                    # Copy to avoid mutating the core metadata cache
                    row_copy = dict(row)
                    row_copy["score"] = float(sc)
                    tier_resolved.append(row_copy)

                if not tier_resolved:
                    continue

                if rerank:
                    # Append new matching items to our isolated un-reranked tracking queue
                    existing_queued_cids = {r["chunk_id"] for r in unreranked_queue}
                    for r in tier_resolved:
                        if r["chunk_id"] not in existing_queued_cids:
                            unreranked_queue.append(r)

                    # Sort the isolated tracking queue strictly by base FAISS score
                    unreranked_queue.sort(key=lambda x: x["score"], reverse=True)

                    # Continuously drain the queue in batches until threshold is met or budget is reached
                    while unreranked_queue and total_reranked < max_rerank_budget:
                        pull_count = min(rerank_pool_size, max_rerank_budget - total_reranked)
                        if pull_count <= 0:
                            break

                        pool = unreranked_queue[:pull_count]
                        unreranked_queue = unreranked_queue[pull_count:]

                        reranked_pool = self._rerank_candidates(
                            query, pool, query_ce_cache, debug=debug
                        )
                        total_reranked += len(reranked_pool)

                        for r in reranked_pool:
                            global_evaluated_chunks[r["chunk_id"]] = r

                        # Check if any candidate has hit our threshold
                        all_current = list(global_evaluated_chunks.values())
                        if unique:
                            all_current = self._dedupe_by_ref(all_current)
                        all_current.sort(key=lambda x: x["score"], reverse=True)

                        best_score = all_current[0]["score"] if all_current else 0.0
                        if best_score >= good_score_threshold:
                            if debug and cascade_popularity:
                                print(
                                    f"[CASCADE STOP] Threshold met in Tier {tier} (K={fetch_k}). "
                                    f"Score: {best_score:.4f} >= {good_score_threshold}"
                                )
                            return all_current[:top_k]

                    if total_reranked >= max_rerank_budget:
                        if debug:
                            print(
                                f"[RERANK BUDGET EXHAUSTED] Reached maximum rerank budget of {max_rerank_budget}."
                            )
                        all_current = list(global_evaluated_chunks.values())
                        if unique:
                            all_current = self._dedupe_by_ref(all_current)
                        all_current.sort(key=lambda x: x["score"], reverse=True)
                        return all_current[:top_k]

                else:
                    # If rerank is disabled, everything stays on the base FAISS score scale cleanly
                    for r in tier_resolved:
                        global_evaluated_chunks[r["chunk_id"]] = r

                    all_current = list(global_evaluated_chunks.values())
                    if unique:
                        all_current = self._dedupe_by_ref(all_current)
                    all_current.sort(key=lambda x: x["score"], reverse=True)

                    best_score = all_current[0]["score"] if all_current else 0.0
                    if best_score >= good_score_threshold or not cascade_popularity:
                        return all_current[:top_k]

            if debug and cascade_popularity:
                print(f"[CASCADE EXTEND] Threshold not met. Expanding search footprint...")

        # Fallback: return best evaluated candidates
        all_current = list(global_evaluated_chunks.values())
        if unique:
            all_current = self._dedupe_by_ref(all_current)
        all_current.sort(key=lambda x: x["score"], reverse=True)
        return all_current[:top_k]


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
    title="Sefaria Semantic Search (Bucket Cascade Edition)",
    description="Semantic search over Sefaria texts using O(1) tiering, cached metadata, and adaptive fetching.",
    version="0.4.1",
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
    book_prefix_filter: Optional[str] = None
    category_filter: Optional[str] = None
    unique: Optional[bool] = False
    cascade_popularity: Optional[bool] = True
    tier_size: int = Field(500, ge=100, le=5000)
    good_score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    addresses: Optional[list[int]] = [-1, -1, -1, -1]
    rerank: Optional[bool] = True
    rerank_pool_size: Optional[int] = Field(40, ge=10, le=500)
    max_rerank_budget: Optional[int] = Field(200, ge=10, le=1000)


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
            book_prefix_filter=req.book_prefix_filter,
            category_filter=req.category_filter,
            unique=req.unique,
            cascade_popularity=req.cascade_popularity,
            tier_size=req.tier_size,
            good_score_threshold=req.good_score_threshold,
            addresses=req.addresses,
            rerank=req.rerank,
            rerank_pool_size=req.rerank_pool_size,
            max_rerank_budget=req.max_rerank_budget,
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
    parser = argparse.ArgumentParser(
        description="Sefaria Local Search CLI Toolkit (Bucket Edition)"
    )
    parser.add_argument("--db", default="sefaria.db")
    parser.add_argument("--index", default="sefaria.index")
    parser.add_argument("--model", default="LaBSE")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--book-filter", default=None)
    parser.add_argument(
        "--book-prefix-filter",
        "--book-prefix",
        default=None,
        help="Filter results where the book title or ID contains this substring",
    )
    parser.add_argument("--category-filter", default=None)
    parser.add_argument("--unique", "-u", action="store_true", default=False)

    parser.add_argument(
        "--no-cascade",
        action="store_false",
        dest="cascade",
        help="Disable tier-based cascading search",
    )
    parser.add_argument(
        "--tier-size",
        type=int,
        default=750,
        help="Number of books to include per cascade tier",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Reranker score threshold to stop expanding tiers",
    )

    parser.add_argument("--rerank", action="store_true", default=True)
    parser.add_argument("--no-rerank", dest="rerank", action="store_false")
    parser.add_argument("--rerank-pool-size", type=int, default=40)
    parser.add_argument(
        "--max-rerank-budget",
        type=int,
        default=200,
        help="Maximum total candidates to evaluate using CrossEncoder per query",
    )
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
        book_prefix_filter=args.book_prefix_filter,
        category_filter=args.category_filter,
        unique=args.unique,
        cascade_popularity=args.cascade,
        tier_size=args.tier_size,
        good_score_threshold=args.threshold,
        addresses=addresses,
        rerank=args.rerank,
        rerank_pool_size=args.rerank_pool_size,
        max_rerank_budget=args.max_rerank_budget,
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
