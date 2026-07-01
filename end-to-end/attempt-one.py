import nltk


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
    nltk.download("punkt")
    # Get text
    sentences, halves = chunkTranscript(text)
    #results per sentence
    for sentence in sentences:
        #get citation
    #results per half sentence
    for half in halves:
        #get citation
    #results per sentence and a half
    for i in range(len(halves)-2):
        halves[i].join(halves[i+1]).join(halves[i+2])
        #get citation
    #results per two sentences
    for i in range(len(sentences)-1):
        sentences[i].join(sentences[i+1])
        #get citation
