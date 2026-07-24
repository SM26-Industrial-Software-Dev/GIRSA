import re
import json
from fastai.text.all import load_learner


class CitationClassifier:
    def __init__(self, model_path="citation_classifier.pkl"):
        """
        Load the trained FastAI classifier once.
        """
        self.learner = load_learner(model_path)

    def prepare_sentences(self, text):
        """
        Clean transcript text, split into sentences,
        and attach an index to each sentence.
        """

        # Remove transliteration tags
        text = text.replace("<i>", "").replace("</i>", "")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        return [
            {
                "index": i,
                "text": sentence.strip()
            }
            for i, sentence in enumerate(sentences)
            if sentence.strip()
        ]

    def classify_sentence(self, sentence):
        """
        Classify a single sentence.

        Returns:
            tuple: (prediction, confidence)
        """
        _, pred_idx, probs = self.learner.predict(sentence)

        return int(pred_idx), float(probs[pred_idx])

    def classify_sentences(self, sentences):
        """
        Run the citation classifier on each sentence.
        Adds prediction and confidence to each dictionary.
        """

        for sentence in sentences:
            prediction, confidence = self.classify_sentence(sentence["text"])

            sentence["prediction"] = prediction
            sentence["confidence"] = confidence

        return sentences

    def classify_transcript(self, transcript_text):
        """
        Complete pipeline:
          transcript -> cleaned/indexed sentences -> classified output
        """

        sentences = self.prepare_sentences(transcript_text)
        return self.classify_sentences(sentences)

    @staticmethod
    def get_citation_sentences(sentences):
        """
        Return only the predicted citation sentences.
        """
        return [s for s in sentences if s["prediction"] == 1]


if __name__ == "__main__":

    classifier = CitationClassifier()

    transcript = """
    Your transcript goes here...
    """

    # Classify transcript
    results = classifier.classify_transcript(transcript)

    # Keep only citation sentences
    citations = classifier.get_citation_sentences(results)

    with open("citations.json", "w", encoding="utf-8") as f:
        json.dump(citations, f, indent=2, ensure_ascii=False)

    print("Saved citations to citations.json")