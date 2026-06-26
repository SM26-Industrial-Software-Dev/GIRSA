import sqlite3
from groq import Groq
import json
import time
import re
import os
from dotenv import load_dotenv
import sys

# Add number at end to choose how many transcripts to chunk. Ex: python3 rag_chunk_and_classify.py 5

load_dotenv()

# Use your own Groq API key.
client = Groq(api_key=YOUR_API_KEY_HERE)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def fetch_transcript_from_db(db_path: str, num_of_transcripts: int = 1) -> list:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT text FROM full_transcripts LIMIT {num_of_transcripts}")
    rows = cursor.fetchmany(num_of_transcripts)
    conn.close()
    if not rows:
        raise ValueError("❌ No transcript found")
    cleaned_transcripts = []
    for row in rows:
        clean_text = re.sub(r"<i>|</i>", "", row[0])
        cleaned_transcripts.append(clean_text)
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
            model="llama-3.1-8b-instant",
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
- "last_active_source": the most recent Torah source cited anywhere in the transcript so far (null if none yet)
- "all_seen_sources": a list of every Torah source cited anywhere in the transcript so far

CLASSIFICATION CODES:
- 0 = current_chunk introduces or quotes a Torah source that does NOT appear in all_seen_sources.
      Torah sources include: Torah, Tanach, Mishnah, Gemara, Midrash, Rambam, Rashi, Ritva,
      Tosafos, Shulchan Aruch, or any named Rishon or Acharon.
- 1 = current_chunk continues a quotation of a Torah source from the preceding chunk.
- 2 = No Torah source is being quoted or cited in current_chunk. This is the speaker's own
      commentary, a story, a transition, or a conceptual explanation with no active citation.

DECISION PROCESS (follow in order):
1. Read current_chunk fully. Name the specific Torah source being cited, if any (e.g. "Rambam Hilchos Teshuvah", "Gemara Brachos"). If none, go to step 4.
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

def classify_single_chunk(
    current_text: str,
    previous_text: str | None,
    last_active_source: str | None,
    all_seen_sources: list[str]
) -> dict:
    """Classifies one chunk using the full source history accumulated so far."""
    payload = {
        "current_chunk": current_text,
        "previous_chunk": previous_text,
        "last_active_source": last_active_source,
        "all_seen_sources": all_seen_sources
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
        return {
            "classification": 2,
            "reasoning": "Fallback — API error.",
            "source_in_current": None,
        }


def classify_chunks_sequentially(prepared_chunks: list) -> list:
    """
    Iterates through chunks one at a time, maintaining a running list of every
    source seen so far. The LLM uses this history to correctly classify
    returning sources as class 1 even after intervening commentary chunks.
    """
    results = []
    previous_text = None
    last_active_source = None
    all_seen_sources = []   # Grows as new sources are discovered

    for chunk in prepared_chunks:
        print(f"  🔍 Classifying chunk {chunk['chunk_index']} | Known sources so far: {all_seen_sources or 'none'}")

        result = classify_single_chunk(
            current_text=chunk["text"],
            previous_text=previous_text,
            last_active_source=last_active_source,
            all_seen_sources=all_seen_sources
        )

        source = result.get("source_in_current")
        classification = result.get("classification", 2)

        # Update source history in Python (not left to the LLM)
        if source and classification == 0:
            # Brand new source — add to the seen list
            if source not in all_seen_sources:
                all_seen_sources.append(source)
            last_active_source = source
        elif source and classification == 1:
            # Returning source — update last active but don't duplicate list
            last_active_source = source

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
    """
    Class 1 is valid whenever the source appeared anywhere earlier in the
    transcript — not just in the immediately previous chunk. The only invalid
    case is class 1 on the very first chunk (nothing could have been seen yet).
    """
    for i, chunk in enumerate(classified_chunks):
        if chunk["classification"] == 1 and i == 0:
            print(f"⚠️  Correcting chunk {chunk['chunk_index']}: class 1 on first chunk is impossible → setting to 0")
            chunk["classification"] = 0
            chunk["reasoning"] += " [Post-processing: first chunk cannot be a continuation.]"
    return classified_chunks


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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classified_chunks (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_number INTEGER NOT NULL,
            classification    INTEGER NOT NULL,
            text              TEXT NOT NULL
        )
    """)

    # Clear any previous run so re-runs don't duplicate data
    cursor.execute("DELETE FROM classified_chunks")

    rows = [
        (
            chunk["transcript_number"],
            chunk["classification"],
            chunk["text"]
        )
        for chunk in chunks
    ]

    cursor.executemany("""
        INSERT INTO classified_chunks
            (transcript_number, classification, text)
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
        list_of_transcripts = fetch_transcript_from_db(INPUT_DB_PATH, int(num_of_transcripts))
        all_chunks = []

        for i, transcript in enumerate(list_of_transcripts):
            transcript_number = i + 1

            print(f"\n{'='*50}")
            print(f"📖 Processing transcript {transcript_number} of {len(list_of_transcripts)}")
            print(f"{'='*50}")

            final_chunks = generate_conceptual_chunks(transcript)

            # Record which transcript each chunk came from
            for chunk in final_chunks:
                chunk["transcript_number"] = transcript_number

            all_chunks.extend(final_chunks)

        print("\n\n================ FINAL OUTPUT ================")
        for chunk in all_chunks:
            word_count = len(chunk.get("text", "").split())
            label = {0: "🆕 New Source", 1: "➡️  Continuation", 2: "💬 Commentary"}.get(chunk["classification"], "❓ Unknown")
            print(f"\n🏷️  Chunk {chunk['chunk_index']} | {word_count} words | {label} (Class {chunk['classification']})")
            print(f"📚 Source: {chunk.get('source_in_current') or 'None'}")
            print(f"🧠 Reasoning: {chunk['reasoning']}")
            print(f"📜 Text: {chunk['text']}")
            print("-" * 50)

        print(f"\n📡 Step 4: Saving to database...")
        save_chunks_to_db(all_chunks, OUTPUT_DB_PATH)

        print(f'\n⏱️  Total time: {time.time() - start_time:.2f}s')

    except Exception as e:
        print(f"Error in main: {e}")
        raise


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
