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
            batch_results = self.extract_queries_batch(batch)
            all_results.extend(batch_results)

        return all_results

    def _chunk_sentences(self, sentences, batch_size=25):
        return [
            sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)
        ]

    def extract_queries_batch(self, sentences):
        """
        Parameters
        ----------
        sentences : list[dict]

        Each dictionary must contain at least:
            {
                "index": ...
                "text": ...
            }

        Returns
        -------
        The same list of sentence dictionaries with a new field:

            sentence["queries"] = [ ... ]
        """

        indexed_input = [{"index": s["index"], "text": s["text"]} for s in sentences]

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
{{"index": 4, "text": "The Gemara says דברים שבלב אינם דברים."}}

Output:
{{"index": 4, "queries": ["דברים שבלב אינם דברים"]}}

Input:
{{"index": 6, "text": "The Rambam was a doctor."}}

Output:
{{"index": 6, "queries": []}}

Input:
{{"index": 10, "text": "Rav Aharon Kotler is quoted as saying that you are not allowed to switch unless it is an absolute case of necessity."}}

Output:
{{"index": 10, "queries": ["you are not allowed to switch unless it is an absolute case of necessity"]}}

Return ONLY valid JSON: an array of objects, one per input sentence, of the form:
{{"index": <the sentence's index from the input>, "queries": [...]}}

Every index from the input must appear exactly once in your output, even if its queries list is empty. The index in each output object must exactly match the index of the sentence it corresponds to. Do not skip, combine, or merge sentences.

Input sentences:

{json.dumps(indexed_input, indent=2, ensure_ascii=False)}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

            results = json.loads(response.text)
        except Exception as e:
            print(f"Gemini call failed for this batch: {e}")
            results = []

        results_by_index = {r["index"]: r.get("queries", []) for r in results}

        missing = []
        for sentence in sentences:
            i = sentence["index"]
            if i in results_by_index:
                sentence["queries"] = results_by_index[i]
            else:
                sentence["queries"] = []
                missing.append(i)

        if missing:
            print(f"Missing sentence indices in this batch: {missing}")

        return sentences


def main():

    API_KEY = os.environ["GEMINI_API_KEY"]

    extractor = GeminiQuoteExtractor(API_KEY)
    sentences = [
        {
            "index": 0,
            "text": "And then also interesting is that the הרמבם later mentions as a separate category in י״א, ולא שבועת שוא בלבד היא אסורה אלא אפילו להזכיר שם מן השמות המיוחדים לבטלה אסור ואף על פי שלא נשבע הרי הכתוב מצווה ואומר ליראה את השם הנכבד והנורא ובכלל יראתו שלא יזכירו לבטלה.",
        },
        {"index": 1, "text": "And there the הרמבם mentions a cure of sorts."},
        {
            "index": 2,
            "text": "He says לפיכך אם טעה הלשון והוציא שם שמים לבטלה ימהר מיד וישבח ויפאר ויהדר לו כדי שלא יזכר לבטלה.",
        },
        {
            "index": 3,
            "text": "He says, right, so immediately if he made a mistake and he said one of the שמות המיוחדים לבטלה, he should say words of שבח immediately.",
        },
        {
            "index": 4,
            "text": "And he could say וכיצד אמר השם אומר ברוך הוא לעולם ועד או גדול הוא ומהולל מאד וכיוצא בזה כדי שלא יהא לבטלה.",
        },
        {
            "index": 7,
            "text": "So let's take a step back for a second before we continue.",
        },
        {"index": 8, "text": "The Gemara in Berachos discusses this at length."},
        {
            "index": 10,
            "text": "The Shulchan Aruch paskens that one who is a ba'al keri may still learn Torah and daven, and there is no need to be machmir on the old minhag.",
        },
        {
            "index": 13,
            "text": "Rashi explains שאין אדם עשוי להשליך מעותיו לאיבוד, meaning a person doesn't typically throw away his money for nothing.",
        },
        {
            "index": 14,
            "text": "This whole sugya is found on Daf Beis of Maseches Kiddushin.",
        },
        {
            "index": 15,
            "text": "The Mishnah Berurah writes that you shouldn't rely on this heter unless there's a real need, and the Chazon Ish adds that even then you should try to find an alternative first.",
        },
        {
            "index": 18,
            "text": "Okay, any questions so far before we move to the next halacha?",
        },
        {
            "index": 20,
            "text": "The Chofetz Chaim is quoted as saying that a person should judge others favorably even when the facts seem to point the other way.",
        },
        {
            "index": 21,
            "text": "Rav Moshe Feinstein discusses a similar case in his teshuvos.",
        },
        {
            "index": 24,
            "text": "We learn from here that כבוד הבריות דוחה לא תעשה שבתורה, meaning the dignity of a person overrides a Torah prohibition in certain circumstances.",
        },
        {"index": 27, "text": "There's a famous Ramban on this pasuk."},
        {
            "index": 30,
            "text": "The Rambam in Hilchos Deos says that a person should always try to walk in the middle path, not going to extremes in either direction.",
        },
        {
            "index": 31,
            "text": "So basically what happened next is really fascinating if you think about it.",
        },
        {"index": 33, "text": "The Vilna Gaon."},
        {
            "index": 36,
            "text": "The Taz holds that this only applies bedieved, not lechatchila, while the Shach argues it applies even lechatchila.",
        },
        {
            "index": 39,
            "text": "It says in the Midrash אין לך כל עשב ועשב שאין לו מזל ברקיע שמכה אותו ואומר לו גדל, that every blade of grass has a heavenly force telling it to grow.",
        },
        {
            "index": 40,
            "text": "That's covered extensively in Even HaEzer siman chof gimmel.",
        },
        {
            "index": 42,
            "text": "Rabbeinu Yona writes that true teshuva requires not just regret but a genuine change in one's ways going forward.",
        },
        {"index": 45, "text": "Anyway, let's move on to the next topic."},
        {
            "index": 48,
            "text": "The Gemara says כל המקיים נפש אחת מישראל כאילו קיים עולם מלא, that whoever sustains a single Jewish soul is considered as if he sustained an entire world.",
        },
        {
            "index": 52,
            "text": "The Rama adds that this is only the case in a she'as hadchak, not under normal circumstances.",
        },
        {
            "index": 58,
            "text": "There's a whole machlokes rishonim about this that goes back to the Rif and the Rosh.",
        },
        {
            "index": 65,
            "text": "The Gemara in Kesubos says האומר תנו גט זה לאשתי צריך שיאמר בפני עדים, that one who instructs to give a get to his wife must say so in front of witnesses.",
        },
        {
            "index": 66,
            "text": "So that's the basic halacha, and now let's get into some of the nuances.",
        },
        {
            "index": 71,
            "text": "The Aruch HaShulchan holds like the Rama here, and explains that the reason is because we're worried about aginus.",
        },
        {"index": 79, "text": "Rav Elyashiv."},
        {
            "index": 80,
            "text": "The Chazon Ish is quoted as saying that in matters of health, one should always follow the majority of doctors rather than seeking out a lenient minority opinion.",
        },
        {
            "index": 88,
            "text": "This is discussed at length by the Achronim in siman tes vav.",
        },
        {
            "index": 95,
            "text": "It's brought down in the Yerushalmi as well, with a slightly different lashon.",
        },
        {
            "index": 96,
            "text": "The Yerushalmi says כל המצמצם עצמו מלכעוס מוחלין לו על כל עונותיו, that one who restrains himself from anger is forgiven for all his sins.",
        },
        {
            "index": 103,
            "text": "Let's take a quick break and we'll pick this up in a few minutes.",
        },
        {
            "index": 110,
            "text": "The Ritva writes that the reason for this din is not because of hezek, but rather because of the general principle of ein adam oseh besias chaveiro chinam.",
        },
        {"index": 118, "text": "Tosfos on this daf."},
        {
            "index": 119,
            "text": "The Bach disagrees with the Beis Yosef here, and holds that the minhag should follow the stricter opinion, while the Taz sides with the Beis Yosef.",
        },
        {
            "index": 127,
            "text": "We see a similar idea by the din of pikuach nefesh doche shabbos, where the Gemara says כמעט כל דיני שבת נדחים מפני פיקוח נפש חוץ משלש עבירות.",
        },
        {
            "index": 128,
            "text": "Okay so let's summarize what we covered before moving forward.",
        },
        {
            "index": 135,
            "text": "The Steipler is quoted as saying that emunah peshuta is often stronger than emunah built purely on intellectual proofs.",
        },
        {
            "index": 144,
            "text": "This whole topic is covered in the sixth perek of the masechta.",
        },
        {
            "index": 150,
            "text": "The Sfas Emes writes that every yom tov contains within it a hearah that is unique to that specific time of year, and it's up to a person to tap into it.",
        },
        {"index": 160, "text": "Rav Chaim Kanievsky."},
    ]

    results = extractor.extract_queries(sentences)

    print("=" * 80)

    for sentence in results:
        print(f"Index : {sentence['index']}")
        print(f"Sentence : {sentence['text']}")
        print(f"Queries  : {sentence['queries']}")
        print("-" * 80)


if __name__ == "__main__":
    main()

