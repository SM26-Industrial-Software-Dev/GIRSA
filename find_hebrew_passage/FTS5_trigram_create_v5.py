import sqlite3
import re

DB_FILE = "sefaria.db"
HEBREW_REGEX = re.compile(r"[\u0590-\u05ff]")

def clean_whole_chunk(text):
    """
    Prepares whole chunks for FTS5 trigram indexing.
    Strips raw HTML markup and adds padding around periods/colons 
    to prevent tokens from getting permanently fused together.
    """
    if not text: 
        return ""
        
    # 1. Strip structural HTML markup (like <b> and </b> tags)
    cleaned = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Add padding spaces around periods and colons 
    # This prevents "ועריה.ולמראה" from fusing into a broken token
    cleaned = re.sub(r'([\.:׃])', r' \1 ', cleaned)
    
    # 3. Collapse whitespace down into a clean single string
    return " ".join(cleaned.split()).strip()

def rebuild_database_v5():
    """Builds a database architecture that indexes intact whole chunks."""
    print(f"Connecting to '{DB_FILE}' to build resilient whole-chunk V5 structures...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Maximize disk transaction performance
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA cache_size = -4000000;") 
    cursor.execute("PRAGMA temp_store = MEMORY;")
    cursor.execute("PRAGMA synchronous = OFF;")
    
    # Drop existing version 5 lookup structures
    cursor.execute("DROP TABLE IF EXISTS chunks_fts_v5;")
    cursor.execute("DROP TABLE IF EXISTS sentences_lookup_v5;")
    conn.commit()

    cursor.execute("""
    CREATE TABLE sentences_lookup_v5 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        passage_id INTEGER,
        sentence_text TEXT
    );
    """)
    cursor.execute("CREATE INDEX idx_sentences_passage_v5 ON sentences_lookup_v5(passage_id);")
    conn.commit()

    # Extract all text segments directly from the source table
    cursor.execute("SELECT id, text FROM chunks WHERE text IS NOT NULL;")
    all_chunks = cursor.fetchall()
    
    prepared_chunks = []
    for chunk_id, text in all_chunks:
        # Index rows only if they contain valid Hebrew text
        if HEBREW_REGEX.search(text):
            sanitized_text = clean_whole_chunk(text)
            if len(sanitized_text) > 5:
                prepared_chunks.append((chunk_id, sanitized_text))
            
    cursor.executemany(
        "INSERT INTO sentences_lookup_v5 (passage_id, sentence_text) VALUES (?, ?);", 
        prepared_chunks
    )
    conn.commit()

    print("Creating NATIVE FTS5 TRIGRAM Virtual Table across raw chunks...")
    cursor.execute("""
    CREATE VIRTUAL TABLE chunks_fts_v5 USING fts5(
        sentence_text,
        content='sentences_lookup_v5',
        content_rowid='id',
        tokenize='trigram'
    );
    """)
    conn.commit()

    print("Building native FTS index matrix...")
    cursor.execute("""
        INSERT INTO chunks_fts_v5(rowid, sentence_text)
        SELECT id, sentence_text FROM sentences_lookup_v5;
    """)
    conn.commit()
    conn.close()
    print("Ingestion complete! Entire intact chunks are now indexed.\n")

if __name__ == "__main__":
    rebuild_database_v5()
