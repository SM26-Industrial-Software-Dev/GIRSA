import os
import re
import json
import random
import sqlite3
from bs4 import BeautifulSoup

# 1. Define configuration settings
OUTPUT_FOLDER = "byt5_dataset_prep"
HOLD_OUT_PERCENTAGE = 0.10  # 10% of documents
WORD_LIMIT = 25  # Exclude phrases greater than 25 words

# Regex pattern to match all Hebrew characters (including points/nikkud)
HEBREW_PATTERN = re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F]")

# Ensure the target directory folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 2. Connect to your SQLite database file
db_path = r"C:\Users\Adir\SoferAi\AI-Project\transcripts.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch all unique document IDs to calculate the holdout pool
cursor.execute("SELECT id FROM full_transcripts")
all_doc_ids = [row[0] for row in cursor.fetchall()]

# 3. Deterministically select 10% of documents to hold out completely
random.seed(42)
random.shuffle(all_doc_ids)
split_point = int(len(all_doc_ids) * HOLD_OUT_PERCENTAGE)

# Separate documents into allowed and excluded sets
excluded_doc_ids = all_doc_ids[:split_point]
excluded_docs_set = set(excluded_doc_ids)

# Re-fetch both 'id' and 'text' rows for full extraction processing
cursor.execute("SELECT id, text FROM full_transcripts")
rows = cursor.fetchall()

phrase_to_docs = {}

# 4. Extract, clean, and map (skipping the excluded holdout docs)
for doc_id, text in rows:
    if doc_id in excluded_docs_set:
        continue  # Skip entirely to prevent these words from being sent to ByT5
        
    if text:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup.find_all("i"):
            raw_text = tag.get_text()
            
            # Remove periods, commas, and forward slashes
            cleaned_text = raw_text.replace(",", "").replace(".", "").replace("/", "")
            
            # Remove all Hebrew letters/characters
            cleaned_text = HEBREW_PATTERN.sub("", cleaned_text).strip()
            
            # Count words by splitting on remaining whitespace
            word_count = len(cleaned_text.split())
            
            # Filter: Skip if empty or GREATER than 25 words
            if cleaned_text and word_count <= WORD_LIMIT:
                if cleaned_text not in phrase_to_docs:
                    phrase_to_docs[cleaned_text] = set()
                phrase_to_docs[cleaned_text].add(doc_id)

conn.close()

# 5. Export all extracted phrases into one unified file inside the folder
phrases_list = sorted(list(phrase_to_docs.keys()))
phrases_output_path = os.path.join(OUTPUT_FOLDER, "all_extracted_phrases.txt")

with open(phrases_output_path, "w", encoding="utf-8") as f:
    for phrase in phrases_list:
        f.write(phrase + "\n")

# 6. Export the exact list of document IDs that were NOT sent to ByT5
excluded_ids_output_path = os.path.join(OUTPUT_FOLDER, "excluded_document_ids.txt")

with open(excluded_ids_output_path, "w", encoding="utf-8") as f:
    for doc_id in sorted(excluded_doc_ids):
        f.write(str(doc_id) + "\n")

# 7. Export the data manifest for looking up sources later
manifest_output_path = os.path.join(OUTPUT_FOLDER, "data_manifest.json")
manifest = {phrase: list(doc_ids) for phrase, doc_ids in phrase_to_docs.items()}

with open(manifest_output_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=4)