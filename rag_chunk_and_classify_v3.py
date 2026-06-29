import sqlite3
from groq import Groq
import json
import time
import re
import os
from dotenv import load_dotenv
import sys

load_dotenv()

client = Groq(api_key="YOUR_API_KEY_HERE")

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def is_transcript_already_processed(output_db_path: str, transcript_id) -> bool:
    try:
        conn = sqlite3.connect(output_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT transcript_id FROM classified_chunks WHERE transcript_id = ? LIMIT 1",
            (transcript_id,)
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return False
        raise e

def fetch_transcript_from_db(input_db_path: str, output_db_path: str, num_of_transcripts: int = 1) -> dict:
    conn = sqlite3.connect(input_db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, text FROM full_transcripts")
    
    new_transcripts = {}
    
    # Loop through rows one by one until we have enough unused transcripts
    while len(new_transcripts) < num_of_transcripts:
        row = cursor.fetchone()
        if not row:
            break  # Reached the absolute end of the database
            
        transcript_id, raw_text = row[0], row[1]
        
        if not is_transcript_already_processed(output_db_path, transcript_id):
            new_transcripts[transcript_id] = raw_text

    conn.close()
    
    if not new_transcripts:
        raise ValueError("❌ No unprocessed transcripts remaining in the database!")
        
    if len(new_transcripts) < num_of_transcripts:
        print(f"⚠️ Warning: only found {len(new_transcripts)}/{num_of_transcripts} unprocessed transcripts.")
        
    cleaned_transcripts = {}
    for transcript_id, text_content in new_transcripts.items():
        clean_text = re.sub(r"<i>|</i>", "", text_content)
        cleaned_transcripts[transcript_id] = clean_text
        
    return cleaned_transcripts


# ─────────────────────────────────────────────
# STEP 1 — BOUNDARY DETECTION
# ─────────────────────────────────────────────

SYSTEM_PROMPT_1 = """
You are a semantic text optimizer for Torah transcripts.
I am going to give you a transcript broken into numbered lines (sentences), along with a list of "candidate_boundaries" where we mathematically plan to split the text.

Your Job:
Review each candidate line number. If that line number cuts off a continuous thought, a specific Torah source citation, or an active story, shift that boundary line number slightly higher or lower (by 1-2 lines) to place the break at a natural semantic transition point.

The Rules:
- Only adjust the numbers slightly to fix semantic cuts.
- Do not delete boundaries or merge massive sections together.
- Your output must be a valid JSON object with a single key "optimized_boundaries" containing the finalized array of integers.

Example Output:
{
  "optimized_boundaries": [5, 12, 17, 23]
}
"""

def split_into_sentences(text: str) -> list:
    if not text:
        return []
    sentences = re.split(r'(?<=[\.!\?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_baseline_boundaries(sentences: list) -> list:
    boundaries = []
    current_words = 0
    word_counts = [len(s.split()) for s in sentences]
    total_words = sum(word_counts)
    words_spent = 0

    for i, sentence in enumerate(sentences):
        words = word_counts[i]
        words_remaining = total_words - words_spent

        if current_words + words > 100 and current_words >= 75:
            if words_remaining >= 75:
                boundaries.append(i + 1)
                current_words = words
            else:
                if current_words + words_remaining > 125:
                    remaining_sentences_count = len(sentences) - i
                    halfway_offset = remaining_sentences_count // 2
                    if halfway_offset > 0:
                        boundaries.append(i + halfway_offset + 1)
                    break
                else:
                    current_words += words
        else:
            current_words += words

        words_spent += words

    return boundaries


def get_chunks(transcript_text: str) -> tuple:
    """
    1. Pre-chunks text mathematically.
    2. Sends to LLM to semantically optimize boundaries.
    3. Returns the raw sentences and the finalized boundary array.
    """
    sentences = split_into_sentences(transcript_text)
    if not sentences:
        return [], []

    baseline_boundaries = get_baseline_boundaries(sentences)
    print(f"🔢 Baseline boundaries: {baseline_boundaries}")

    numbered_lines = [f"[{i+1}] {s}" for i, s in enumerate(sentences)]
    user_data = {
        "candidate_boundaries": baseline_boundaries,
        "transcript": numbered_lines
    }

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_1},
                {"role": "user", "content": json.dumps(user_data, ensure_ascii=False)}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        raw_json = completion.choices[0].message.content
        parsed = json.loads(raw_json)
        boundaries = parsed.get("optimized_boundaries", baseline_boundaries)
        print(f"🎯 Optimized boundaries: {boundaries}")

    except Exception as e:
        print(f"❌ LLM boundary optimization failed ({e}). Falling back to math boundaries.")
        boundaries = baseline_boundaries

    final_boundaries = sorted(list(set([1] + [int(b) for b in boundaries])))
    return sentences, final_boundaries


# ─────────────────────────────────────────────
# STEP 2 — SEQUENTIAL CLASSIFICATION
# ─────────────────────────────────────────────

SYSTEM_PROMPT_2 = """
You are a Torah transcript classifier. Your job is to classify a single text chunk.

You will receive:
- "current_chunk": the text to classify
- "previous_chunk": the text of the chunk immediately before this one (null if first chunk)

CLASSIFICATION CODES:
- 0 = current_chunk introduces or quotes a Torah source that does NOT appear in previous_chunk.
      Torah sources include: Torah, Tanach, Mishnah, Gemara, Midrash, Rambam, Rashi, Ritva,
      Tosafos, Shulchan Aruch, or any named Rishon or Acharon.
- 1 = current_chunk continues a quotation of a Torah source from the preceding chunk.
- 2 = No Torah source is being quoted or cited in current_chunk. This is the speaker's own
      commentary, a story, a transition, or a conceptual explanation with no active citation.

DECISION PROCESS (follow in order):
1. Read current_chunk fully. Name the specific Torah source being cited, if any (e.g. "Rambam Hilchos Avodah Zara", "Gemara Eiruvin"). If none, go to step 4.
2. Check: Is this source quoted in the previous chunk?
   - YES → classification 1 (returning to a known source)
   - NO  → classification 0 (brand new source)
3. Stop. Do not proceed to step 4.
4. No source found in current_chunk → classification 2.

OUTPUT FORMAT (JSON only, no extra text):
{
  "source_in_current": "name of source, or null",
  "reasoning": "One specific sentence explaining the classification and naming the source.",
  "classification": 0
}
"""

def classify_single_chunk(current_text: str, previous_text: str | None) -> dict:
    """Classifies one chunk using the full source history accumulated so far."""
    payload = {
        "current_chunk": current_text,
        "previous_chunk": previous_text,
    }
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_2},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            response_format={"type": "json_object"},
            temperature=0.0   # Fully deterministic
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ Classification failed for chunk: {e}")
        return None


def classify_chunks_sequentially(prepared_chunks: list) -> list:
    results = []
    previous_text = None

    for chunk in prepared_chunks:
        print(f"  🔍 Classifying chunk {chunk['chunk_index']}")

        result = classify_single_chunk(
            current_text=chunk["text"],
            previous_text=previous_text
        )

        if result is not None:

            source = result.get("source_in_current")
            classification = result.get("classification", 2)

            results.append({
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "classification": classification,
                "reasoning": result.get("reasoning", ""),
                "source_in_current": source
            })

        previous_text = chunk["text"]
        time.sleep(0.5)   # Avoid rate limits

    return results


# ─────────────────────────────────────────────
# STEP 3 — POST-PROCESSING VALIDATION
# ─────────────────────────────────────────────

def validate_continuations(classified_chunks: list) -> list:
    validated = []
    for idx, chunk in enumerate(classified_chunks):
        if chunk["classification"] == 1 and (idx == 0 or classified_chunks[idx - 1]["classification"] == 2):
            print(f"⚠️ Invalid continuation at chunk {chunk['chunk_index']}. Skipping this chunk...")
        else:
            validated.append(chunk)
    return validated


# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────

def generate_conceptual_chunks(transcript: str) -> list:
    print(f"\n📡 Step 1: Mapping Boundaries...")
    sentences, boundaries = get_chunks(transcript)
    if not sentences:
        return []

    # Reconstruct text chunks locally in Python (guaranteed correct splitting)
    prepared_chunks = []
    for idx, start_pos in enumerate(boundaries):
        start_idx = max(0, start_pos - 1)
        end_idx = (boundaries[idx + 1] - 1) if idx + 1 < len(boundaries) else len(sentences)
        chunk_text = " ".join(sentences[start_idx:end_idx]).strip()
        if chunk_text:
            prepared_chunks.append({
                "chunk_index": idx + 1,
                "text": chunk_text
            })

    time.sleep(2)

    print(f"\n📡 Step 2: Classifying {len(prepared_chunks)} chunks sequentially...")
    classified_chunks = classify_chunks_sequentially(prepared_chunks)

    print(f"\n📡 Step 3: Validating continuations...")
    validated_chunks = validate_continuations(classified_chunks)

    return validated_chunks


# ─────────────────────────────────────────────
# DATABASE OUTPUT
# ─────────────────────────────────────────────

def save_chunks_to_db(chunks: list, db_path: str = "classified_chunks.db") -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Standardized to 'transcript_id' to align with verification checks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classified_chunks (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id     TEXT NOT NULL,
            classification    INTEGER NOT NULL,
            text              TEXT NOT NULL
        )
    """)

    rows = [
        (
            chunk["transcript_id"],
            chunk["classification"],
            chunk["text"]
        )
        for chunk in chunks
    ]

    cursor.executemany("""
        INSERT INTO classified_chunks
            (transcript_id, classification, text)
        VALUES (?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()
    print(f"💾 Saved {len(rows)} chunks to '{db_path}'")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(num_of_transcripts: int = 1):
    INPUT_DB_PATH  = "transcripts_full_text.db"
    OUTPUT_DB_PATH = "classified_chunks.db"
    start_time = time.time()

    try:
        dict_of_transcripts = fetch_transcript_from_db(INPUT_DB_PATH, OUTPUT_DB_PATH, num_of_transcripts)
        all_chunks = []

        for i, (transcript_id, transcript_text) in enumerate(dict_of_transcripts.items()):
            transcript_number = i + 1

            print(f"\n{'='*50}")
            print(f"📖 Processing transcript {transcript_number} of {len(dict_of_transcripts)}")
            print(f"{'='*50}")

            final_chunks = generate_conceptual_chunks(transcript_text)
            for chunk in final_chunks:
                chunk["transcript_id"] = transcript_id

            all_chunks.extend(final_chunks)

        print("\n\n================ FINAL OUTPUT ================")
        for chunk in all_chunks:
            word_count = len(chunk.get("text", "").split())
            label = {0: "🆕 New Source", 1: "➡️  Continuation", 2: "💬 Commentary"}.get(chunk["classification"], "❓ Unknown")
            print(f"\n🏷️  Chunk {chunk['chunk_index']} | {word_count} words | {label} (Class {chunk['classification']})")
            print("-" * 50)

        print(f"\n📡 Step 4: Saving to database...")
        save_chunks_to_db(all_chunks, OUTPUT_DB_PATH)

        print(f'\n⏱️  Total time: {time.time() - start_time:.2f}s')

    except Exception as e:
        print(f"Error in main: {e}")
        raise


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main()
