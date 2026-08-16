#!/usr/bin/env python3
"""
generate_embeddings_local.py

Highly optimized, fully resumable local vectorization engine
tailored for Intel Arc iGPUs (XPU) using native PyTorch streaming.
"""

import sqlite3
import numpy as np
import torch
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- LOCAL PATHS ---
DB_PATH = Path("sefaria.db")
MODEL_NAME = "sentence-transformers/LaBSE"

# ---------------------------------------------------------------------------
# PyTorch Native Streaming Dataset
# ---------------------------------------------------------------------------


class LocalSQLiteDataset(Dataset):
    """
    Streams row tracking targets instantly using indexed lookups
    to prevent memory bloat on large datasets.
    """

    def __init__(self, db_path, model_name):
        self.db_path = str(db_path)
        self.model_name = model_name

        # Pull only the integer IDs of unindexed chunks into memory
        conn = sqlite3.connect(self.db_path)
        log.info("Scanning database index tracks for remaining chunks...")
        self.ids = [
            row[0]
            for row in conn.execute(
                """
            SELECT id FROM chunks
            EXCEPT
            SELECT chunk_id FROM chunk_embeddings WHERE model = ?
            """,
                (self.model_name,),
            ).fetchall()
        ]
        conn.close()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        chunk_id = self.ids[idx]
        # Open an isolated connection per worker thread for clean concurrency
        conn = sqlite3.connect(self.db_path)
        text_str = conn.execute(
            "SELECT text FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()[0]
        conn.close()
        return int(chunk_id), str(text_str)


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------


def main():
    if not DB_PATH.exists():
        log.error(f"Database file '{DB_PATH}' not found!")
        return

    # Native Intel XPU Handshake Verification
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    else:
        log.warning("Intel XPU acceleration not detected. Falling back to CPU.")
        device = "cpu"

    log.info(f"Loading transformer model onto local device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.eval()

    # Load streaming dataset tracks
    dataset = LocalSQLiteDataset(DB_PATH, MODEL_NAME)
    total_remaining = len(dataset)

    if total_remaining == 0:
        log.info("Database is already 100% vectorized. Exiting.")
        return

    log.info(f"Remaining chunks to process locally: {total_remaining}")

    # Local Optimization Tuning:
    # We increase the outer stream batch size to 4096.
    # num_workers=2 forks separate background CPU threads on your laptop to
    # read text strings asynchronously while the GPU focuses entirely on matrix operations.
    dl = DataLoader(
        dataset,
        batch_size=4096,
        num_workers=2,
        pin_memory=True if device == "xpu" else False,
        shuffle=False,
        drop_last=False,
    )

    # Establish main persistent writer thread
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("PRAGMA journal_mode = WAL; PRAGMA synchronous = OFF;")
    conn.execute("BEGIN TRANSACTION;")

    uncommitted_count = 0
    commit_threshold = 100000

    log.info("Launching parallel execution engine loop...")

    for chunk_ids, texts in dl:
        chunk_ids = [int(x) for x in chunk_ids]

        with torch.no_grad():
            # The inner encoding batch size can now safely crawl up to 512
            embeddings = model.encode(
                list(texts),
                batch_size=512,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        insert_data = []
        for c_id, vec in zip(chunk_ids, embeddings):
            vector_blob = vec.astype(np.float32).tobytes()
            insert_data.append((c_id, MODEL_NAME, vector_blob))

        conn.executemany(
            """
            INSERT OR REPLACE INTO chunk_embeddings (chunk_id, model, vector)
            VALUES (?, ?, ?)
            """,
            insert_data,
        )

        uncommitted_count += len(chunk_ids)

        if uncommitted_count >= commit_threshold:
            conn.execute("COMMIT;")
            log.info(
                f"--- Local Milestone: Committed {uncommitted_count} records to disk ---"
            )
            conn.execute("BEGIN TRANSACTION;")
            uncommitted_count = 0
        else:
            log.info(
                f"Vectorized batch of {len(chunk_ids)} chunks (Uncommitted: {uncommitted_count})..."
            )

    # Flush remaining trails
    conn.execute("COMMIT;")
    conn.executescript("PRAGMA synchronous = ON;")
    conn.close()
    log.info("Local vectorization run finished successfully!")


if __name__ == "__main__":
    main()
