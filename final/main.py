from HebrewTransliterationEngine import HebrewTransliterationEngine
from bs4 import BeautifulSoup
import logging
import requests
import book_alias_lookup
import sqlite3

# from process_text import process_sentence
from CitationClassifier import CitationClassifier

from FuzzyMatch import SefariaEngineV5
import re


def split_into_sentences(transcript: str):
    # Clean text and break into indexed sentences
    transcript = re.sub(r"\s+", " ", transcript).strip()
    sentences = re.split(r"(?<=[.!?])\s+", transcript)
    return [
        {"index": i, "text": sentence.strip()}
        for i, sentence in enumerate(sentences)
        if sentence.strip()
    ]


def are_titles_same_book(db_cursor, title_a: str, title_b: str) -> bool:
    # Quick string check first to save a DB query if they match exactly
    if title_a.strip().lower() == title_b.strip().lower():
        return True

    possible_titles_b = find_in_aliases(title_b)

    query = """
        SELECT DISTINCT b.id
        FROM books b
        LEFT JOIN book_aliases ba ON b.id = ba.book_id
        WHERE :title IN (b.id, b.en_title, b.he_title, ba.alias);
    """

    # Get canonical ID for title A
    db_cursor.execute(query, {"title": title_a})
    res_a = db_cursor.fetchone()

    for title_b in possible_titles_b:
        # Get canonical ID for title B
        db_cursor.execute(query, {"title": title_b})
        res_b = db_cursor.fetchone()

        # If both resolved and match, it's the same book
        if res_a and res_b:
            return res_a[0] == res_b[0]

    return False

def clean_hebrew(text):
    return "".join(re.findall(r"[\u0590-\u05ff]", text))

def match_authors(str1, str2):
    clean1 = clean_hebrew(str1)
    clean2 = clean_hebrew(str2)
    
    smaller = clean1 if len(clean1) <= len(clean2) else clean2
    larger = clean2 if len(clean1) <= len(clean2) else clean1
    
    if not smaller:
        return True
        
    smaller_idx = 0
    larger_idx = 0
    
    while larger_idx < len(larger) and smaller_idx < len(smaller):
        if larger[larger_idx] == smaller[smaller_idx]:
            smaller_idx += 1
        larger_idx += 1
        
    return smaller_idx == len(smaller)


def score_candidate(candidate: dict, window_ner: list[dict], db_cursor) -> float:
    candidate_author = candidate["author"] if candidate["author"] else None
    candidate_title = candidate["book_id"]
    candidate_original_score = candidate["score"]
    matches = 0
    for entry in window_ner:
        if not entry:
            continue
        if entry["TITLE"]:
            if candidate_author is not None and match_authors(candidate_author, entry["TITLE"]):
                matches += 1
            elif are_titles_same_book(
                db_cursor=db_cursor, title_a=candidate_title, title_b=entry["TITLE"]
            ):
                matches += 1
        if entry["PATH"] and candidate["addr1"]:
            if entry["PATH"] == candidate["addr1"]:
                matches += 1
        if entry["PATH"] and candidate["addr2"]:
            if entry["PATH"] == candidate["addr2"]:
                matches += 1
        if entry["PATH"] and candidate["addr3"]:
            if entry["PATH"] == candidate["addr3"]:
                matches += 1
        if entry["PATH"] and candidate["addr4"]:
            if entry["PATH"] == candidate["addr4"]:
                matches += 1

def extract_hebrew_chunks(text, min_words=5):
    # Match sequences of Hebrew characters and spaces/punctuation between them
    hebrew_pattern = r"[\u0590-\u05FF]+(?:\s+[\u0590-\u05FF]+)*"
    # Find all continuous Hebrew blocks
    raw_chunks = re.findall(hebrew_pattern, text)
    # Filter to keep only chunks with at least `min_words` words
    result = [chunk for chunk in raw_chunks if len(chunk.split()) >= min_words]
    return result


def find_links(transcript: str):
    log = logging.getLogger("source_pipeline")
    sentences = split_into_sentences(transcript)
    # Transliterate all sentences
    transliterator = HebrewTransliterationEngine()
    transliterated_trancsript = transliterator.process_transcript(transcript)
    transliterated_sentences = split_into_sentences(transliterated_trancsript)
    # Filter to only citation sentences
    classifier = CitationClassifier()
    citation_sentences = classifier.classify_sentences(transliterated_sentences)
    fuzzy = SefariaEngineV5()
    for sentence in citation_sentences:
        # Find Hebrew chunks of certain size to fuzzy find against
        hebrew_chunks = extract_hebrew_chunks(sentence["text"], 5)
        fuzzy_results = []
        for chunk in hebrew_chunks:
            # add found sources to source field
            result = fuzzy.execute_search(chunk)
            fuzzy_results.append(result)
        if fuzzy_results:
            sentence["fuzzyMatches"] = fuzzy_results
        results_for_sentence = []
        for query in sentence["queries"]:
            payload = {
                "query": query,
                "top_k": 20,
                "unique": True,
                "rerank_pool_size": 100,
            }
            try:
                response = requests.post(
                    url="http://localhost:8000/search", json=payload
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                # Only add to local list if results are non-empty
                if results:
                    results_for_sentence.append(results)
            except requests.RequestException as e:
                log.warning("Search failed for query=%r: %s", query, e)
        # Only create the key in the dictionary if we found non-empty outputs
        if results_for_sentence:
            sentence["APIOutputs"] = results_for_sentence
        # NER and Naive data extraction
        # Normalize NER and Naive data to match API/Fuzzy outputs
    # Get NER output for the last 4-5 sentences
    max_trail_size = 4
    citation_sentences.sort(key=lambda item: item["index"])
    for i, current_sent in enumerate(citation_sentences):
        # 1. Build rolling window of NER items (current + up to 3 preceding sentences)
        start = max(0, i - max_trail_size + 1)
        window = citation_sentences[start : i + 1]
        window_ner = [item.get("NER") for item in window if item.get("NER")]

        links = []

        # 2. Score API outputs
        for api_call_results in current_sent.get("APIOutputs", []):
            if not api_call_results:
                continue
            for hit in api_call_results:
                boosted_score = score_candidate(hit, window_ner, db_cursor)
                hit_copy = hit.copy()
                hit_copy["final_score"] = boosted_score
            api_call_results.sort(key=lambda c: c["final_score"], reverse=True)
            links.append(api_call_results[0])

        # 3. Score Fuzzy matches
        for fuzzy_match_results in current_sent.get("fuzzyMatches", []):
            if not fuzzy_match_results:
                continue
            for hit in fuzzy_match_results:
                boosted_score = score_candidate(hit, window_ner, db_cursor)
                hit_copy = hit.copy()
                hit_copy["final_score"] = boosted_score
            fuzzy_match_results.sort(key=lambda c: c["final_score"], reverse=True)
            links.append(fuzzy_match_results[0])
    # convert to html with footnotes
    return merge_footnotes()


def merge_footnotes(sentences, citation_sentences):
    if len(sentences) != len(citation_sentences):
        raise RuntimeError(
            "citation_sentences must contain the same number of sentences as the original text"
        )

    if not isinstance(citation_sentences, list):
        raise TypeError("citation_sentences must be a list")

    for index, item in enumerate(citation_sentences):
        if not isinstance(item, dict):
            raise TypeError(
                f"Item at index {index} in citation_sentences must be a dictionary"
            )
        if "link" not in item:
            raise KeyError(
                f"Dictionary at index {index} in citation_sentences is missing the required 'link' key"
            )

    processed_sentences = []
    footnotes = []
    footnote_count = 1

    for index, text in enumerate(sentences):
        links = citation_sentences[index]["link"]

        for link in links:
            text += f'<sup><a href="#fn{footnote_count}">[{footnote_count}]</a></sup>'
            footnotes.append(f'<li id="fn{footnote_count}">{link}</li>')
            footnote_count += 1

        processed_sentences.append(text)

    final_html = f"<p>{' '.join(processed_sentences)}</p>\n<hr>\n<ol>\n"
    final_html += "\n".join(footnotes)
    final_html += "\n</ol>"

    return BeautifulSoup(final_html, "html.parser")


def main():
    transcript = """
    """
    find_links(transcript)


if __name__ == "__main__":
    main()
