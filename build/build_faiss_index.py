#!/usr/bin/env python3
"""
build_faiss_index.py

Reads all rows from chunk_embeddings and builds a FAISS index for fast
approximate nearest neighbor search, plus a sidecar mapping file from
FAISS internal index position -> chunk_id.

Run this any time after an embedding pass to (re)build the search index.
Safe to re-run as embedding coverage grows — it always reads the current
state of chunk_embeddings.

Usage:
    python build_faiss_index.py --db sefaria.db --model LaBSE --out sefaria.index

Dependencies:
    pip install faiss-cpu numpy
    # or faiss-gpu if you have CUDA and want faster index build / search
"""

import argparse
import logging
import sqlite3
import struct
import time
from pathlib import Path

import faiss
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def blob_to_vec(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.array(struct.unpack(f"{n}f", b), dtype=np.float32)


def build_index(db_path: str, model_name: str, out_path: str, index_type: str):
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA cache_size = -64000")

    total = conn.execute(
        "SELECT COUNT(*) FROM chunk_embeddings WHERE model = ?", (model_name,)
    ).fetchone()[0]
    log.info("Found %d total embeddings in database", total)

    if total == 0:
        return

    # Determine dimensionality from the first vector
    first_blob = conn.execute(
        "SELECT vector FROM chunk_embeddings WHERE model = ? LIMIT 1", (model_name,)
    ).fetchone()[0]
    dim = len(blob_to_vec(first_blob))
    log.info("Vector dimensionality: %d", dim)

    # --- THE 32GB RAM OPTIMIZED INDEX CHOICES ---
    if index_type == "flat":
        # Compresses vectors down to 16-bit floats inside RAM (Saves 50% RAM instantly)
        log.info("Building a Memory-Optimized Float16 Flat Index...")
        index = faiss.index_factory(dim, "SQfp16", faiss.METRIC_INNER_PRODUCT)

    elif index_type == "ivf":
        # IVF + Product Quantization (PQ).
        # Breaks each 768-dim vector into 64 sub-vectors encoded as 8-bit bytes.
        # This reduces your memory footprint by ~85%!
        log.info("Building a Hyper-Compressed IVF-PQ Index...")
        nlist = max(1, int(np.sqrt(total)))  # Number of clusters
        m = 64  # Number of centroid sub-spaces
        nbits = 8  # Bits per sub-space byte

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
        )
        log.info(
            f"IVF-PQ configuration parameters set: nlist={nlist}, m={m}, bits={nbits}"
        )
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    id_map = np.zeros(total, dtype=np.int64)
    batch_size = 50000
    offset = 0
    t0 = time.time()

    # IVF-PQ requires a training step to map the cluster centroids
    if index_type == "ivf":
        log.info("Training IVF-PQ quantizer on database text samples...")
        # Pull a clean slice of up to 100,000 rows to build the quantization codebook
        sample_limit = min(total, 100000)
        sample_rows = conn.execute(
            "SELECT vector FROM chunk_embeddings WHERE model = ? LIMIT ?",
            (model_name, sample_limit),
        ).fetchall()
        train_vecs = np.stack([blob_to_vec(r[0]) for r in sample_rows])
        index.train(train_vecs)
        log.info("Training layer calculation complete.")

    # Stream the database into FAISS memory segments
    while offset < total:
        rows = conn.execute(
            """
            SELECT chunk_id, vector FROM chunk_embeddings
            WHERE model = ?
            ORDER BY chunk_id
            LIMIT ? OFFSET ?
        """,
            (model_name, batch_size, offset),
        ).fetchall()

        if not rows:
            break

        chunk_ids = np.array([r[0] for r in rows], dtype=np.int64)
        vecs = np.stack([blob_to_vec(r[1]) for r in rows])

        index.add(vecs)
        id_map[offset : offset + len(rows)] = chunk_ids
        offset += len(rows)

        elapsed = time.time() - t0
        rate = offset / elapsed if elapsed > 0 else 0
        log.info(
            "  %d/%d records successfully added to FAISS index (%.0f/s)",
            offset,
            total,
            rate,
        )

    log.info(
        "Index compression layout complete in %.1f minutes", (time.time() - t0) / 60
    )

    faiss.write_index(index, out_path)
    np.save(out_path + ".idmap.npy", id_map)
    log.info("Saved compressed index to %s", out_path)
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sefaria.db")
    parser.add_argument(
        "--model",
        default="LaBSE",
        help="Model name as stored in chunk_embeddings.model column",
    )
    parser.add_argument("--out", default="sefaria.index")
    parser.add_argument(
        "--index-type",
        default="flat",
        choices=["flat", "ivf"],
        help="'flat' = exact, simpler, fine up to a few million vectors. "
        "'ivf' = approximate, much faster, recommended at 10M+.",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        log.error("Database not found: %s", args.db)
        return

    build_index(args.db, args.model, args.out, args.index_type)


if __name__ == "__main__":
    main()
