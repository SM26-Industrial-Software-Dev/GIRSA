#!/usr/bin/env python3
"""
extract_citations.py

Takes classified transcript chunks (Class 0/1/2) and:
1. Groups each Class 0 chunk with any immediately following Class 1 chunks
   into a single "citation body" (ends at the next Class 0 or Class 2)
2. Splits the body into sentences, in order
3. Searches each sentence against the Sefaria index directly (via SefariaSearcher,
   no API call) and applies a small bonus to earlier sentences so that, all else
   close to equal, the earlier matching phrase wins
4. Stops early if a sentence scores above HIGH_SCORE_THRESHOLD
5. Otherwise keeps the best-scoring sentence across the whole body
6. Only keeps a result if it clears MIN_SCORE_THRESHOLD
7. Prints results (DB write is stubbed in but not called yet, for this draft)

Usage:
    python extract_citations.py
    python extract_citations.py --chunks-db classified_chunks.db
    python extract_citations.py --high-threshold 0.85 --min-threshold 0.6
    python extract_citations.py --position-bonus 0.05
"""

import argparse
import re
import sqlite3
import logging
from pathlib import Path

from search import SefariaSearcher  # direct import - no HTTP, no separate API process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
CHUNKS_DB_PATH       = "classified_chunks.db"
TRANSCRIPTS_DB_PATH  = "transcripts.db"
SEFARIA_DB_PATH      = "sefaria.db"
SEFARIA_INDEX_PATH   = "sefaria.index"
SEFARIA_MODEL        = "sentence-transformers/LaBSE"

HIGH_SCORE_THRESHOLD = 0.85  # Stop searching immediately if a sentence scores this high
MIN_SCORE_THRESHOLD  = 0.60  # Minimum score to accept any result as a real citation
POSITION_BONUS       = 0.05  # Weight applied to earlier sentences - tunable
TOP_K                = 3     # Number of Sefaria results to request per sentence


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences on . ? or ! followed by whitespace.

    Note: simple splitter, will be imperfect on messy mixed Hebrew/English
    shiur transcripts. Can be improved later.
    """
    if not text or not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Grouping chunks into citation bodies
# ---------------------------------------------------------------------------
def group_into_bodies(chunks: list[dict]) -> list[list[dict]]:
    """
    Group chunks into citation bodies. A body starts at a Class 0 chunk and
    absorbs all immediately following Class 1 chunks. A body ends when the
    next Class 0 or Class 2 chunk is hit - both mean we've moved on from the
    source that was being introduced.

    Chunks must already be sorted by transcript_id, then position.
    """
    bodies = []
    current_body = []

    for chunk in chunks:
        c = chunk["classification"]

        if c == 0:
            if current_body:
                bodies.append(current_body)
            current_body = [chunk]

        elif c == 1:
            if current_body:
                current_body.append(chunk)
            # A class 1 with no preceding class 0 in this run is unexpected
            # (validation upstream should prevent it) - just drop it silently
            # if it shows up with nothing open.

        else:  # c == 2
            if current_body:
                bodies.append(current_body)
                current_body = []

    if current_body:
        bodies.append(current_body)

    return bodies


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------
def extract_citation_from_body(
    body: list[dict],
    searcher: SefariaSearcher,
    high_threshold: float,
    min_threshold: float,
    position_bonus: float,
    top_k: int,
) -> dict | None:
    """
    Search every sentence across all chunks in a citation body, in order.
    Stop early if a sentence's raw score clears high_threshold.
    Otherwise track the best adjusted score across the whole body.

    Adjusted score = raw_score + position_bonus / (position + 1)
    so earlier sentences get a small boost, but a clearly better later match
    can still win.

    Returns a dict with citing_text, source_text, sefaria_ref, sefaria_url,
    raw_score, adjusted_score, chunk_id - or None if nothing clears min_threshold.
    """
    best = None  # will hold a dict with all fields, keyed by adjusted_score
    position = 0  # running sentence position across the whole body, not per-chunk

    for chunk in body:
        sentences = split_into_sentences(chunk["text"])

        for sentence in sentences:
            if len(sentence.split()) < 3:
                # Too short to plausibly be a citation
                position += 1
                continue

            results = searcher.search(sentence, top_k=top_k, unique=True, weighted=True)

            if not results:
                position += 1
                continue

            top_match = results[0]
            raw_score = top_match["score"]
            adjusted_score = raw_score + (position_bonus / (position + 1))

            log.info(
                "    [pos %d] %r... → raw=%.3f adj=%.3f ref=%s",
                position, sentence[:50], raw_score, adjusted_score,
                top_match.get("sefaria_ref", "?"),
            )

            candidate = {
                "citing_text":    sentence,
                "source_text":    top_match.get("text", ""),
                "sefaria_ref":    top_match.get("sefaria_ref", ""),
                "sefaria_url":    top_match.get("sefaria_url", ""),
                "raw_score":      raw_score,
                "adjusted_score": adjusted_score,
                "chunk_id":       chunk["id"],
            }

            if best is None or adjusted_score > best["adjusted_score"]:
                best = candidate

            # Early stop on a strong raw match - no need to keep searching
            if raw_score >= high_threshold:
                log.info("    High-confidence match found, stopping search early.")
                return best

            position += 1

    if best is None or best["raw_score"] < min_threshold:
        return None

    return best


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def fetch_all_chunks(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, transcript_id, text, classification
        FROM classified_chunks
        ORDER BY transcript_id, id
        """
    ).fetchall()

    return [
        {"id": r[0], "transcript_id": r[1], "text": r[2], "classification": r[3]}
        for r in rows
    ]


def save_citation(conn, transcript_id, citing_text, source_text, sefaria_url):
    """
    Stubbed in but NOT called in this draft - results are printed instead.
    Kept here so writing to the DB is a one-line change once we're ready.
    """
    conn.execute(
        """
        INSERT INTO transcripts_citations (
            transcript_id, start_word_index, end_word_index,
            citing_text, source_text, sefaria_url
        ) VALUES (?, NULL, NULL, ?, ?, ?)
        """,
        (transcript_id, citing_text, source_text, sefaria_url),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    chunks_db_path: str,
    sefaria_db_path: str,
    sefaria_index_path: str,
    sefaria_model: str,
    high_threshold: float,
    min_threshold: float,
    position_bonus: float,
    top_k: int,
):
    if not Path(chunks_db_path).exists():
        log.error("Chunks database not found: %s", chunks_db_path)
        return
    if not Path(sefaria_db_path).exists():
        log.error("Sefaria database not found: %s", sefaria_db_path)
        return
    if not Path(sefaria_index_path).exists():
        log.error("Sefaria FAISS index not found: %s", sefaria_index_path)
        return

    log.info("Loading SefariaSearcher (this loads the model + FAISS index)...")
    searcher = SefariaSearcher(sefaria_db_path, sefaria_index_path, sefaria_model)
    log.info("Ready.")

    chunks_conn = sqlite3.connect(chunks_db_path)
    all_chunks = fetch_all_chunks(chunks_conn)
    log.info("Loaded %d total chunks", len(all_chunks))

    bodies = group_into_bodies(all_chunks)
    log.info("Grouped into %d citation bodies to process", len(bodies))

    confirmed = 0
    no_result = 0

    for i, body in enumerate(bodies, 1):
        transcript_id = body[0]["transcript_id"]
        chunk_ids = [c["id"] for c in body]
        log.info(
            "[%d/%d] Body: %d chunk(s), transcript=%s, chunk ids=%s",
            i, len(bodies), len(body), transcript_id, chunk_ids,
        )

        result = extract_citation_from_body(
            body=body,
            searcher=searcher,
            high_threshold=high_threshold,
            min_threshold=min_threshold,
            position_bonus=position_bonus,
            top_k=top_k,
        )

        if result is None:
            log.info("  → No citation found above threshold.")
            no_result += 1
            continue

        confirmed += 1
        print("\n" + "=" * 70)
        print(f"Transcript:    {transcript_id}")
        print(f"Chunk IDs:     {chunk_ids}  (matched in chunk {result['chunk_id']})")
        print(f"Citing text:   {result['citing_text']}")
        print(f"Sefaria ref:   {result['sefaria_ref']}")
        print(f"Sefaria URL:   {result['sefaria_url']}")
        print(f"Raw score:     {result['raw_score']:.3f}")
        print(f"Adjusted score:{result['adjusted_score']:.3f}")
        print(f"Source text:   {result['source_text'][:150]}")
        print("=" * 70)

        # save_citation(transcripts_conn, transcript_id, result["citing_text"],
        #               result["source_text"], result["sefaria_url"])

    chunks_conn.close()

    log.info("Done.")
    log.info("  Citations found : %d", confirmed)
    log.info("  No result       : %d", no_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract citation phrases from classified transcript chunks."
    )
    parser.add_argument("--chunks-db", default=CHUNKS_DB_PATH)
    parser.add_argument("--sefaria-db", default=SEFARIA_DB_PATH)
    parser.add_argument("--sefaria-index", default=SEFARIA_INDEX_PATH)
    parser.add_argument("--sefaria-model", default=SEFARIA_MODEL)
    parser.add_argument("--high-threshold", type=float, default=HIGH_SCORE_THRESHOLD)
    parser.add_argument("--min-threshold", type=float, default=MIN_SCORE_THRESHOLD)
    parser.add_argument("--position-bonus", type=float, default=POSITION_BONUS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    main(
        chunks_db_path=args.chunks_db,
        sefaria_db_path=args.sefaria_db,
        sefaria_index_path=args.sefaria_index,
        sefaria_model=args.sefaria_model,
        high_threshold=args.high_threshold,
        min_threshold=args.min_threshold,
        position_bonus=args.position_bonus,
        top_k=args.top_k,
    )