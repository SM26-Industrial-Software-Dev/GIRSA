"""
build_db.py

Ingests Sofer.Ai transcript json files into SQLite database
organized into 4 databases:
    1) full text
    2) split into sentances
    3) split into words
    4) set up for linking citations

"""


import sqlite3
import json
import re
from pathlib import Path

# Initilize databases
connection1 = sqlite3.connect("transcripts_full_text.db")
cursor1 = connection1.cursor()

connection2 = sqlite3.connect("transcripts_sentances.db")
cursor2 = connection2.cursor()

connection3 = sqlite3.connect("transcripts_words.db")
cursor3 = connection3.cursor()

connection4 = sqlite3.connect("transcripts_citations.db")
cursor4 = connection4.cursor()

# create tables
table1 = """CREATE TABLE IF NOT EXISTS full_transcripts(
        id TEXT PRIMARY KEY,
        title TEXT,
        text TEXT,
        language TEXT,
        duration INTEGER,
        num_speakers INTEGER,
        lang_for_hebrew_words TEXT,
        link TEXT,
        shiur_id INTEGER
    )"""

connection1.commit()
cursor1.execute(table1)

table2 = """CREATE TABLE IF NOT EXISTS transcript_sentances(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id Text REFERENCES full_transcripts(id),
        sentance TEXT,
        sentance_index INTEGER,
        total_sentances INTEGER
    )"""

cursor2.execute(table2)
connection2.commit()

table3 = """CREATE TABLE IF NOT EXISTS transcript_words(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id Text REFERENCES full_transcripts(id),
        word TEXT,
        hebrew_format TEXT,
        word_index INTEGER,
        total_words INTEGER
    )"""

cursor3.execute(table3)
connection3.commit()

table4 = """CREATE TABLE IF NOT EXISTS transcripts_citations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id Text REFERENCES full_transcripts(id),
        start_word_index INTEGER,
        end_word_index INTEGER,
        citing_text TEXT,
        source_text TEXT,
        sefaria_url TEXT
    )"""

cursor4.execute(table4)
connection4.commit()

# itterate through .json files
directory = Path("./transcripts")

file_count = 0

for path in directory.iterdir():
    if path.is_file() and path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            try:
                with connection1, connection2, connection3:

                    # save reused parts of json in memory
                    transcript_id = data["id"]
                    full_text = data["text"]

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
                        data["title"],
                        full_text,
                        data["primary_language"],
                        data["duration"],
                        data["num_speakers"],
                        lang_for_hebrew_words,
                        data["link"],
                        data["shiur_id"],
                    )

                    cursor1.execute(comand, insert)

                    # sentances database creation
                    # split into sentances
                    sentences = re.split(r"(?<=[?.])\s+", full_text)
                    sentance_length = len(sentences)

                    for index, sentance in enumerate(sentences, start=1):

                        # insert sentances
                        comand = """INSERT INTO transcript_sentances(
                                transcript_id,
                                sentance,
                                sentance_index,
                                total_sentances
                            ) VALUES (?,?,?,?)"""

                        insert = (transcript_id, sentance, index, sentance_length)

                        cursor2.execute(comand, insert)

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

                        cursor3.execute(comand, insert)

                    file_count += 1

                print(f"{file_count} {data["title"]}")

            except sqlite3.Error as e:
                print(f"Database error while processing {path.name}: {e}")

# save table
connection1.close()
connection2.close()
connection3.close()
connection4.close()

print(f"\nSuccessfully added {file_count:,} shiurim to the databases.")
