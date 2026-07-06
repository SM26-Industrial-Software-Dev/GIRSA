import nltk
from build_search_queries import build_search_queries
from build_sefaria_link import create_link, connect
from api_call import get_best_source_link


def halveSentence(sentence: str):
    words = sentence.split()
    mid = len(words) // 2
    a = " ".join(words[:mid])
    b = " ".join(words[mid:])
    return a, b


def chunkTranscript(transcriptText: str):
    sentences = nltk.sent_tokenize(transcriptText)
    halfSentences = []
    for sentence in sentences:
        a, b = halveSentence(sentence)
        halfSentences.append(a)
        halfSentences.append(b)
    return sentences, halfSentences


def main():
    """
    Get the output of the RAG as an array of lists of dictionaries
    """
    transcriptInput = getSources()
    output = []
    for sentence in transcriptInput:
        if sentence is None:
            output.append(None)
            continue
        sources = []
        for source in sentence:
            if source.get("text") is None:
                sources.append(create_link(source))
            else:
                call = build_search_query(source)
                best_link = get_best_source_link("sefaria.db", call)
                sources.append(best_link)
        output.append(sources)

    # nltk.download("punkt")
    # Get text
    # sentences, halves = chunkTranscript(text)
    # results per sentence
    # for sentence in sentences:
    # get citation
    # results per half sentence
    # for half in halves:
    # get citation
    # results per sentence and a half
    # for i in range(len(halves)-2):
    #    halves[i].join(halves[i+1]).join(halves[i+2])
    # get citation
    # results per two sentences
    # for i in range(len(sentences)-1):
    #    sentences[i].join(sentences[i+1])
    # get citation
