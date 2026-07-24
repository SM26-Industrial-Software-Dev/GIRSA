from HebrewTransliterationEngine import HebrewTransliterationEngine

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


def main():
    transcript = """
    """
    find_links(transcript)


if __name__ == "__main__":
    main()
