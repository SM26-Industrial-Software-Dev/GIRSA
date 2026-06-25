"""
build_db.py

Ingests Sofer.Ai transcript json files into SQLite database
organized into 4 databases:
    1) full text
    2) split into sentences
    3) split into words
    4) set up for linking citations

"""

import sqlite3
import logging
import json
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Schema creation
def create_schema(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS full_transcripts(
        id TEXT PRIMARY KEY,
        title TEXT,
        text TEXT,
        language TEXT,
        duration INTEGER,
        num_speakers INTEGER,
        lang_for_hebrew_words TEXT,
        link TEXT,
        shiur_id INTEGER
    );""")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS transcript_sentences(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id TEXT REFERENCES full_transcripts(id),
        sentance TEXT,
        sentance_index INTEGER,
        total_sentences INTEGER
    );""")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS transcript_words(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id TEXT REFERENCES full_transcripts(id),
        word TEXT,
        hebrew_format TEXT,
        word_index INTEGER,
        total_words INTEGER
    );""")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS transcripts_citations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id TEXT REFERENCES full_transcripts(id),
        start_word_index INTEGER,
        end_word_index INTEGER,
        citing_text TEXT,
        source_text TEXT,
        sefaria_url TEXT
    );""")

    conn.commit()

def fill_db(conn: sqlite3.Connection, transcripts_dir: Path):

    """
    Walk the cltk-full directory tree and ingest every JSON text file.
    Hyper-optimized for file transaction batch boundaries and raw performance.
    """
    log.info("Ingesting texts from %s", transcripts_dir)
    files = list(transcripts_dir.rglob("*.json"))
    log.info("Found %d transcript text files", len(files))

    cur = conn.cursor()

    for file_num, fpath in enumerate(files, start=1):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning("Could not read %s: %s", fpath, e)
            continue

        try:
            with conn:
                # save reused parts of json in memory
                transcript_id = data["id"]
                full_text = data["text"]
                title = data["title"]

                # full transcript database creation
                # get language for hebrew words
                lang_list = data.get("lang_for_hebrew_words", [])
                if isinstance(lang_list, list):
                    lang_for_hebrew_words = ",".join(lang_list)
                else:
                    lang_for_hebrew_words = str(lang_list)

                # insert full transcript
                comand = """INSERT INTO full_transcripts(
                        id,
                        title,
                        text,
                        language,
                        duration,
                        num_speakers,
                        lang_for_hebrew_words,
                        link,
                        shiur_id
                    ) VALUES (?,?,?,?,?,?,?,?,?)"""

                insert = (
                    transcript_id,
                    title,
                    full_text,
                    data["primary_language"],
                    data["duration"],
                    data["num_speakers"],
                    lang_for_hebrew_words,
                    data["link"],
                    data["shiur_id"],
                )

                cur.execute(comand, insert)

                # sentences database creation
                # split into sentences
                sentences = re.split(r"(?<=[?.])\s+", full_text)
                sentance_length = len(sentences)

                for index, sentance in enumerate(sentences, start=1):

                    # insert sentences
                    comand = """INSERT INTO transcript_sentences(
                            transcript_id,
                            sentance,
                            sentance_index,
                            total_sentences
                        ) VALUES (?,?,?,?)"""

                    insert = (transcript_id, sentance, index, sentance_length)

                    cur.execute(comand, insert)

                # word database creation
                # split into words
                words = re.split(r"\s+", full_text)
                words_length = len(words)

                for index, word in enumerate(words, start=1):

                    # check language
                    if bool(re.search(r"[\u0590-\u05FF]", word)):
                        language = "he"
                    elif "<i>" in word:
                        language = "hybrid"
                    else:
                        language = "en"

                    # insert words
                    comand = """INSERT INTO transcript_words(
                            transcript_id,
                            word,
                            hebrew_format,
                            word_index,
                            total_words
                        ) VALUES (?,?,?,?,?)"""
                    insert = (transcript_id, word, language, index, words_length)

                    cur.execute(comand, insert)

                if file_num % 50 == 0:
                    log.info("Prossesed shiur: %s, %s", file_num, title)

        except sqlite3.Error as e:
            log.warning("Error inserting content %s: %s", fpath, e)
            continue

def main():
    transcripts_dir = Path("./transcripts")

    conn = sqlite3.connect("transcripts.db")
    create_schema(conn)

    if not transcripts_dir.exists():
        log.error("transcripts/ directory not found under")
    else:
        fill_db(conn, transcripts_dir)

    conn.close()
    log.info("Done. Database written to %s", "transcripts.db")

if __name__ == "__main__":
    main()
