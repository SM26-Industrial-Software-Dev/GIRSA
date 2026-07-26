import json
import os
from google import genai
from google.genai import types


class GeminiQuoteExtractor:

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3.5-flash-lite"
    
    def extract_queries(self, sentences, batch_size=25):
        batches = self._chunk_sentences(sentences, batch_size)
        all_results = []

        for i, batch in enumerate(batches):
            print(f"Processing batch {i + 1}/{len(batches)} ({len(batch)} sentences)")
            batch_results = self._extract_queries_batch(batch)
            all_results.extend(batch_results)

        return all_results

    def _chunk_sentences(self, sentences, batch_size=25):
        return [
            sentences[i:i + batch_size]
            for i in range(0, len(sentences), batch_size)
        ]

    def extract_queries_batch(self, sentences):
        """
        Parameters
        ----------
        sentences : list[dict]

        Each dictionary must contain at least:
            {
                "text": ...
            }

        Returns
        -------
        The same list of sentence dictionaries with a new field:

            sentence["queries"] = [ ... ]
        """

        texts = [sentence["text"] for sentence in sentences]

        prompt = f"""
You are helping build a citation retrieval pipeline for Torah lectures.

Your task is to extract one or more search queries from each sentence.

Each query should contain the actual Torah content that should be sent to a semantic search engine to identify the original Jewish source.

A query is the teaching, ruling, argument, explanation, statement, or idea attributed to a source.

Important distinction:
Extract the CONTENT from the source, not the source itself.

Rules:

1. A sentence may produce zero, one, or multiple search queries.

2. Each query should express a complete teaching or idea whenever possible.
   Prefer complete statements over isolated words or short phrases.

3. Remove attribution phrases such as:
   - The Gemara says...
   - The Rambam writes...
   - Rashi explains...
   - The Midrash teaches...
   - The Mishnah says...
   - The Shulchan Aruch paskens...
   - He says...
   - We learn...
   - and similar attribution phrases.

4. Copy the query exactly from the sentence whenever possible.
   - Do not rewrite.
   - Do not summarize.
   - Do not translate.
   - Do not correct spelling or grammar.
   - Preserve Hebrew, English, and transliteration exactly as written.

5. If the sentence paraphrases a source rather than directly quoting it, extract the paraphrased teaching.

6. Include:
   - direct quotations from sources
   - paraphrased teachings, rulings, arguments, or explanations attributed to a source

7. Do NOT return:
   - names of rabbis or authors by themselves
   - names of books by themselves
   - masechta names by themselves
   - daf numbers, simanim, or chapter references by themselves
   - statements that only mention that a source exists
   - descriptions of what a source discusses without stating the actual teaching
   - conversational filler
   - questions
   - incomplete phrases
   - isolated Hebrew terms

8. If a sentence only mentions a source but does not contain the teaching itself, return an empty list.

Examples:

Input:
"The Gemara says דברים שבלב אינם דברים."

Output:
[
    "דברים שבלב אינם דברים"
]

Input:
"There is a Gemara that talks about רבי שמעון בר יוחאי on Daf Lamed Gimmel of Masechet Shabbos."

Output:
[]

Input:
"Rav Aharon Kotler is quoted as saying that you are not allowed to switch unless it is an absolute case of necessity."

Output:
[
    "you are not allowed to switch unless it is an absolute case of necessity"
]

Return ONLY valid JSON.

The output must contain exactly one list for every input sentence.

The number of output lists MUST equal the number of input sentences.

Even if a sentence contains no useful query, return:
[]

Do not skip, combine, merge, or remove sentences.

The output order must exactly match the input order.

Input sentences:

{json.dumps(texts, indent=2)}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        results = json.loads(response.text)
        
        for sentence, queries in zip(sentences, results):
            sentence["queries"] = queries

        print("Expected:", len(sentences))
        print("Received:", len(results))

        return sentences


def main():

    API_KEY = os.environ["GEMINI_API_KEY"]

    extractor = GeminiQuoteExtractor(API_KEY)
    sentences = [
    {
        "index": 0,
        "text": "And then also interesting is that the הרמבם later mentions as a separate category in י״א, ולא שבועת שוא בלבד היא אסורה אלא אפילו להזכיר שם מן השמות המיוחדים לבטלה אסור ואף על פי שלא נשבע הרי הכתוב מצווה ואומר ליראה את השם הנכבד והנורא ובכלל יראתו שלא יזכירו לבטלה."
    },
    {
        "index": 1,
        "text": "And there the הרמבם mentions a cure of sorts."
    },
    {
        "index": 2,
        "text": "He says לפיכך אם טעה הלשון והוציא שם שמים לבטלה ימהר מיד וישבח ויפאר ויהדר לו כדי שלא יזכר לבטלה."
    },
    {
        "index": 3,
        "text": "He says, right, so immediately if he made a mistake and he said one of the שמות המיוחדים לבטלה, he should say words of שבח immediately."
    },
    {
        "index": 4,
        "text": "And he could say וכיצד אמר השם אומר ברוך הוא לעולם ועד או גדול הוא ומהולל מאד וכיוצא בזה כדי שלא יהא לבטלה."
    },
    {
        "index": 5,
        "text": "And here at least in שבועות he doesn't mention that kind of a response for the first cases."
    },
    {
        "index": 6,
        "text": "So that's חילול השם, like the רמבם says in the הלכה א of פרק י״ב."
    },
    {
        "index": 7,
        "text": "Okay, כי לא ינקה השם, אין לזה ניקיון מדין שמיים עד שייפרע ממנו על שם הגדול שחילל, שנאמר וחיללת את שם אלוהיך אני השם."
    },
    {
        "index": 8,
        "text": "ולפיכך צריך אדם להיזהר מעוון זה יתר מכל העבירות."
    },
    {
        "index": 9,
        "text": "This is an עוון זה מן החמורות, like the הרמבם said in the beginning of הלכות תשובה."
    },
    {
        "index": 10,
        "text": "That the הרמבם is learning that the מעשה ברכה, a מעשה of a ברכה, is really at its essence, I'm not saying what its function is, but its essence is to substantiate something."
    },
    {
        "index": 11,
        "text": "Shulchan Aruch paskens that it runs from the beginning of Sefirah and it goes until..."
    },
    {
        "index": 12,
        "text": "The Shulchan Aruch paskens for Sefardim and in that in the shita of the Beis Yosef, the Shulchan Aruch, there is no celebration on Lag Baomer."
    },
    {
        "index": 13,
        "text": "So you're not allowed to celebrate on Lag Baomer and that's based on a Midrash that says that the plague that caused the deaths of Rabbi Akiva's talmidim ended fifteen days before Shavuos."
    },
    {
        "index": 14,
        "text": "So according to the Shulchan Aruch the last day of the plague was the thirty-fourth day of the Omer, so miktzas hayom kekulo, so on the Lamed-Dalet Baomer that's when the aveilus stops."
    },
    {
        "index": 15,
        "text": "Rav Neuburger shlita likes to point out that the Rema was a great unifier of the Jewish people because the Rema died on Lag Baomer, so it became his yartzeit, and the Rema is the one that said that the aveilus ends on Lag Baomer."
    },
    {
        "index": 16,
        "text": "And so the Rema brought it all together by saying no we could end the aveilus on Lag Baomer and therefore everyone could feel comfortable except for Sefardim having a celebration on Lag Baomer."
    },
    {
        "index": 17,
        "text": "Again it doesn't say that in the Gemara, but the Aruch Hashulchan brings down such a tradition."
    },
    {
        "index": 18,
        "text": "The Chasam Sofer writes in a teshuva in Yoreh Deah סימן רלג, the Chasam Sofer writes about Lag Baomer it's a day you don't say tachanun."
    },
    {
        "index": 19,
        "text": "Chasam Sofer was not particularly excited about Lag Baomer being a special day, but it's a day you have a right not to say tachanun."
    },
    {
        "index": 20,
        "text": "Maharil and the Rema would hold that the evening of Lag Baomer would still be aveilus."
    },
    {
        "index": 21,
        "text": "That's a kasha is better than the teretz, because according to the Rema that's assur."
    },
    {
        "index": 22,
        "text": "And therefore the Alter Rebbe in Shulchan Aruch HaRav says that really there are only lamed-beis days of aveilus, that there aren't lamed-gimmel days of aveilus."
    },
    {
        "index": 23,
        "text": "But there is a Gemara that appears in the second Perek of Masechet Kiddushin, Daf Mem-Tes, which is a critical Gemara in terms of our whole understanding of the relationship of Dibur and Machshava as relates to many many Halachos in the Torah."
    },
    {
        "index": 24,
        "text": "So the Gemara says the conclusion of the Sugya was that דברים שבלב אינם דברים."
    }
]


    results = extractor.extract_queries(sentences)

    print("=" * 80)

    for sentence in results:
        print(f"Sentence : {sentence['text']}")
        print(f"Queries  : {sentence['queries']}")
        print("-" * 80)


if __name__ == "__main__":
    main()