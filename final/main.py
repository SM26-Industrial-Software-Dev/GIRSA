from HebrewTransliterationEngine import HebrewTransliterationEngine
from bs4 import BeautifulSoup

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
def extract_hebrew_chunks(text, min_words=5):
    # Match sequences of Hebrew characters and spaces/punctuation between them
    hebrew_pattern = r'[\u0590-\u05FF]+(?:\s+[\u0590-\u05FF]+)*'
    # Find all continuous Hebrew blocks
    raw_chunks = re.findall(hebrew_pattern, text)
    # Filter to keep only chunks with at least `min_words` words
    result = [chunk for chunk in raw_chunks if len(chunk.split()) >= min_words]
    return result

def find_links(transcript: str):
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
        for chunk in hebrew_chunks:
            # add found sources to source field
            result = fuzzy.execute_search(chunk)
        # NER and Naive data extraction
            # Any source path containing "Daf" must become a addr1 gemara ref
    for sentence in citation_sentences:
        # API calls to check against the source field and extracted data
    # convert to html with footnotes
    return merge_footnotes()

def merge_footnotes(sentences, citation_sentences):
    if len(sentences) != len(citation_sentences):
        raise RuntimeError("citation_sentences must contain the same number of sentences as the original text") 1

    if not isinstance(citation_sentences, list):
        raise TypeError("citation_sentences must be a list")
        
    for index, item in enumerate(citation_sentences):
        if not isinstance(item, dict):
            raise TypeError(f"Item at index {index} in citation_sentences must be a dictionary")
        if "link" not in item:
            raise KeyError(f"Dictionary at index {index} in citation_sentences is missing the required 'link' key")
        
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
