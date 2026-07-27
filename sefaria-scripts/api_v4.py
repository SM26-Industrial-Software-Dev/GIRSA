#!/usr/bin/env python3
"""
api_v4.py

Includes weights with PageRank and standalone cross-encoding function, 
plus partial book filtering, sentence-fractional search, and author resolution.
"""

import argparse
import os
import logging
import sqlite3
import struct
import pandas as pd
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder

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
    def __init__(self, db_path: str = "sefaria.db", index_path: str = "sefaria.index", model_name: str = "LaBSE", 
                 reranker_name: str = "BAAI/bge-reranker-v2-m3"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA cache_size = -32000")

        log.info("Loading FAISS index from %s ...", index_path)
        self.index = faiss.read_index(index_path)
        self.id_map = np.load(index_path + ".idmap.npy")

        log.info("Loading model %s ...", model_name)
        self.model = SentenceTransformer(model_name)

        if isinstance(self.index, faiss.IndexIVFPQ):
            self.index.nprobe = 16

        log.info("Loading reranker %s ...", reranker_name)
        self.reranker = CrossEncoder(reranker_name, device="cpu", max_length=512)

        self.pagerank_scores = {}
        pagerank_path = os.getenv("PAGERANK_PATH", "accurate_sefaria_pagerank.csv")

        if os.path.exists(pagerank_path):
            log.info("Loading PageRank scores from %s ...", pagerank_path)
            try:
                df_pr = pd.read_csv(pagerank_path)
                self.pagerank_scores = dict(zip(df_pr['Book Title'].str.strip(), df_pr['PageRank Score']))
                log.info("Loaded PageRank scores for %d books.", len(self.pagerank_scores))
            except Exception as e:
                log.error("Failed to load PageRank file: %s. Defaulting to no boosts.", e)
        else:
            log.warning("PageRank file not found at %s. Weighted searches will fallback to 1.0.", pagerank_path)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32).reshape(1, -1)

    def search(
        self,
        query: str,
        top_k: int = 100,
        book_filter: str = None,
        book_partial_filter: str = None,
        category_filter: str = None,
        unique: bool = False,
        weighted: bool = True,
        rerank: bool = False,
        rerank_pool: int = 100,
        addresses=None,
    ):
        query_vec = self.embed_query(query)

        fetch_k = (
            10000 if addresses or book_filter or book_partial_filter or category_filter or unique else top_k
        )

        scores, positions = self.index.search(query_vec, fetch_k)
        scores = scores[0]
        positions = positions[0]

        results = []
        seen = set()

        for score, pos in zip(scores, positions):
            if pos == -1:
                continue
            chunk_id = int(self.id_map[pos])
            row = self._resolve_chunk(chunk_id, book_filter, book_partial_filter, category_filter)
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

            row["score"] = float(score)
            results.append(row)

            if unique:
                seen.add(row["sefaria_ref"])
        if rerank:
            results = self.rerank(query, results[:rerank_pool])

        if weighted:
            self.weightify(results)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _resolve_chunk(
        self, chunk_id: int, book_filter: str = None, book_partial_filter: str = None, category_filter: str = None
    ):
        row = self.conn.execute(
            """
            SELECT
                c.id, c.text, c.chunk_index,
                p.sefaria_ref, p.book_id,
                b.he_title, b.en_title, b.categories,
                v.language, v.version_title, p.addr1, p.addr2, p.addr3, p.addr4,
                GROUP_CONCAT(abm.author_id, ', ') AS authors
            FROM chunks c
            JOIN passages p ON c.passage_id = p.id
            JOIN books b ON p.book_id = b.id
            JOIN versions v ON c.version_id = v.id
            LEFT JOIN author_book_map abm ON b.id = abm.book_id
            WHERE c.id = ?
            GROUP BY c.id
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
            "authors": row[14] if row[14] else "Unknown",
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

        if book_partial_filter:
            clean_pf = book_partial_filter.strip().lower()
            if (
                clean_pf not in result["book_id"].lower()
                and clean_pf not in result["en_title"].lower()
            ):
                return None

        return result

    def weightify(self, unweighted_results: list[dict])->list[dict]:
        for result in unweighted_results:
            categories = result['categories']
            categories_set = set(categories.split('|'))
            start_score = result['score']
            final_score = start_score

            if categories_set.intersection({"Second Temple", "Reference"}):
                final_score = 0
            elif "Tanakh" in categories_set:
                if "Rishonim on Tanakh" in categories_set: final_score = start_score * 1.05
                elif "Acharonim on Tanakh" in categories_set: final_score = start_score * 1.03
                elif "Modern Commentary on Tanakh" in categories_set: final_score = 0
                elif "Targum" in categories_set: final_score = start_score * 1.04
                else: final_score = start_score * 1.1
            elif "Talmud" in categories_set:
                if "Bavli" in categories_set:
                    if "Rishonim on Talmud" in categories_set: final_score = start_score * 1.03
                    elif "Acharonim on Talmud" in categories_set: final_score = start_score * 1.01
                    elif "Commentary on Minor Tractates" in categories_set: final_score = start_score * 0.95
                    elif "Guides" in categories_set: final_score = 0
                    elif "Minor Tractates" in categories_set: final_score = start_score
                    elif "Modern Commentary on Talmud" in categories_set: final_score = 0
                    else: final_score = start_score * 1.08
                else:
                    final_score = start_score * 0.95 if "Commentary" in categories_set else start_score * 1.02
            elif "Tosefta" in categories_set or "Midrash" in categories_set:
                final_score = start_score * 0.95 if "Commentary" in categories else start_score
            elif "Halakha" in categories_set:
                if "Modern" in categories_set: final_score = start_score * 0.5
                elif "Commentary" in categories: final_score = start_score
                else: final_score = start_score * 1.04
            elif categories_set.intersection({"Responsa", "Musar", "Jewish Thought"}):
                if "Modern" in categories:
                    final_score = 0
                elif "Jewish Thought" in categories_set:
                    final_score = start_score * 0.93
                else:
                    final_score = start_score
            elif "Mishnah" in categories_set:
                if "Rishonim on Mishnah" in categories_set or "Acharonim on Mishnah" in categories_set: final_score = start_score * 1.01
                elif "Modern Commentary on Mishnah" in categories_set: final_score = 0
                else: final_score = start_score * 1.06
            else:
                flat_weights = {"Liturgy": 0.98, "Kabbalah": 0.80, "Chasidut": 0.93}
                has_matched = False
                for cat in categories_set:
                    if cat in flat_weights:
                        final_score = start_score * flat_weights[cat]
                        has_matched = True
                        break
                if not has_matched:
                    final_score = start_score

            if self.pagerank_scores:
                clean_book_id = result['book_id'].strip()
                pr_score = self.pagerank_scores.get(clean_book_id, 0.0)
                pr_boost = 1.0 + (4.5 * pr_score)
                final_score *= pr_boost

            result['score'] = final_score
        return unweighted_results

    def rerank(self, query: str, results: list, top_k: int = 10, debug: bool = False) -> list:
        if not results:
            return results

        if len(results) > 200:
            print(f"\nWarning: {len(results)} results will be cross encoded. This is a large batch and will take a long time.\n")

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

        return results[:top_k]

    def split_search(
        self,
        sentence: str,
        threshold: float = 0.5,
        top_k: int = 10,
        book_filter: str = None,
        book_partial_filter: str = None,
        category_filter: str = None,
        unique: bool = False,
        weighted: bool = True
    ):
        """
        Divides a sentence into full, halves, and quarters.
        Executes a search for each piece, pooling results that clear the threshold score.
        Caps the pool at 100 unique documents, and applies the cross-encoder to rerank.
        """
        words = sentence.split()
        n = len(words)
        
        # If the sentence is too short, just query it whole to avoid meaningless micro-queries
        if n < 4:
            queries = [sentence]
        else:
            q1 = " ".join(words[:n//4])
            q2 = " ".join(words[n//4:n//2])
            q3 = " ".join(words[n//2:3*n//4])
            q4 = " ".join(words[3*n//4:])
            
            h1 = " ".join(words[:n//2])
            h2 = " ".join(words[n//2:])
            
            queries = [sentence, h1, h2, q1, q2, q3, q4]

        pooled_results = {}
        
        for q in queries:
            if not q.strip():
                continue
            
            res = self.search(
                query=q,
                top_k=50, 
                book_filter=book_filter,
                book_partial_filter=book_partial_filter,
                category_filter=category_filter,
                unique=unique,
                weighted=weighted,
                rerank=False
            )
            
            for r in res:
                if r['score'] >= threshold:
                    if r['chunk_id'] not in pooled_results:
                        pooled_results[r['chunk_id']] = r

        pooled_list = list(pooled_results.values())
        pooled_list.sort(key=lambda x: x["score"], reverse=True)
        pooled_list = pooled_list[:100]

        return self.rerank(sentence, pooled_list, top_k=top_k)

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
    book_partial_filter: Optional[str] = None
    category_filter: Optional[str] = None
    unique: Optional[bool] = False
    weighted: Optional[bool] = False
    addresses: Optional[list[int]] = [-1, -1, -1, -1]


class SplitSearchRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=5000)
    threshold: float = Field(0.5)
    top_k: int = Field(10, ge=1, le=100)
    book_filter: Optional[str] = None
    book_partial_filter: Optional[str] = None
    category_filter: Optional[str] = None
    unique: Optional[bool] = False
    weighted: Optional[bool] = False


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
    authors: Optional[str] = None
    addr1: Optional[int] = None
    addr2: Optional[int] = None
    addr3: Optional[int] = None
    addr4: Optional[int] = None
    bi_encoder_score: Optional[float] = None


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
            book_partial_filter=req.book_partial_filter,
            category_filter=req.category_filter,
            unique=req.unique,
            weighted=req.weighted,
            addresses=req.addresses,
        )
    except Exception as e:
        log.exception("Search failed for query %r", req.query)
        raise HTTPException(status_code=500, detail=str(e))

    results = [SearchResult(**r) for r in raw]
    return SearchResponse(query=req.query, results=results, total=len(results))


@app.post("/split_search", response_model=SearchResponse)
def split_search_api(req: SplitSearchRequest):
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")
    try:
        raw = searcher.split_search(
            sentence=req.sentence,
            threshold=req.threshold,
            top_k=req.top_k,
            book_filter=req.book_filter,
            book_partial_filter=req.book_partial_filter,
            category_filter=req.category_filter,
            unique=req.unique,
            weighted=req.weighted
        )
    except Exception as e:
        log.exception("Split search failed for sentence %r", req.sentence)
        raise HTTPException(status_code=500, detail=str(e))

    results = [SearchResult(**r) for r in raw]
    return SearchResponse(query=req.sentence, results=results, total=len(results))


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
    parser.add_argument("--db", default="sefaria.db")
    parser.add_argument("--index", default="sefaria.index")
    parser.add_argument("--model", default="LaBSE")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--book-filter", default=None)
    parser.add_argument("--book-partial-filter", default=None)
    parser.add_argument("--category-filter", default=None)
    parser.add_argument("--unique", "-u", action="store_true", default=False)
    parser.add_argument("--weighted", "-w", action="store_true", default=False)
    parser.add_argument("--split-search", "-s", action="store_true", default=False, 
                        help="Enable fractional sentence search and pooling")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Score threshold when pooling results for split search")
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
    
    if args.split_search:
        print(f"\nRunning Split Search on: {args.query!r}")
        results = cli_searcher.split_search(
            sentence=args.query,
            threshold=args.threshold,
            top_k=args.top_k,
            book_filter=args.book_filter,
            book_partial_filter=args.book_partial_filter,
            category_filter=args.category_filter,
            unique=args.unique,
            weighted=args.weighted
        )
    else:
        results = cli_searcher.search(
            args.query,
            top_k=args.top_k,
            book_filter=args.book_filter,
            book_partial_filter=args.book_partial_filter,
            category_filter=args.category_filter,
            unique=args.unique,
            weighted=args.weighted,
            addresses=addresses,
        )

    print(f"\nQuery: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        score_val = r.get('score', 0.0)
        print(
            f"{i}. [{score_val:.3f}] {r['sefaria_ref']}  ({r['en_title']}, {r['language']})"
        )
        print(f"   Author(s): {r.get('authors', 'Unknown')}")
        print(f"   Version: {r['version_title']}")
        print(f"   {r['text'][:150]}")
        print(f"   {r['sefaria_url']}")
        print()