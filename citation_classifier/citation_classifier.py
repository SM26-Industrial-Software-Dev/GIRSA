import re
from fastai.text.all import load_learner

MODEL_PATH = "citation_classifier.pkl"
learner = load_learner(MODEL_PATH)


def prepare_sentences(text):
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

    # Create indexed sentence dictionaries
    return [
        {
            "index": i,
            "text": sentence.strip()
        }
        for i, sentence in enumerate(sentences)
        if sentence.strip()
    ]


def classify_sentences(sentences, learner):
    """
    Run the citation classifier on each sentence.
    Adds prediction and confidence to each dictionary.
    """

    for sentence in sentences:
        _, pred_idx, probs = learner.predict(sentence["text"])

        sentence["prediction"] = int(pred_idx)
        sentence["confidence"] = float(probs[pred_idx])

    return sentences


def classify_transcript(transcript_text):
    """
    Complete pipeline:
      transcript -> cleaned/indexed sentences -> classified output
    """

    sentences = prepare_sentences(transcript_text)

    return classify_sentences(sentences, learner)

def get_citation_sentences(sentences):
    return [s for s in sentences if s["prediction"] == 1]

if __name__ == "__main__":

    transcript = """
    "Beginning of this <i>parsha</i>, <i>Rashi</i> asks why the Torah had to count <i>shemos b'nei Yisrael habaim Mitzraimah</i> if it already did so in <i>Parshas Vayigash</i>. And the <i>Ramban agav</i> answers with a literary answer: there are two different books, you have to end one book and then you have to start the next book by repeating to show where it starts compared to the last book. But <i>Rashi</i> gives a <i>midrashic</i> answer, he says: <i>af al pi shemenamam bechayehem bishmosam, chazar umenamam bemisasan, l'hodia chibasom</i>. He counted them when they were alive, now he's counting them after they passed away to show their <i>chiba</i>, <i>shenimshalu l'kochavim</i>. The reason Hashem counts the Jews so much is to show <i>chiba</i>, that they're compared to the stars, <i>shemotzi'am umachnisam bemispar uvishmosam</i>. Hashem brings the stars out every night and brings them in every morning with their counting by their number and their names, <i>shene'emar</i>: <i>hamotzi bemispar tzva'am l'chulam b'shem yikra</i>. So the Jews are compared to stars. Why <i>davka</i> here in the beginning of <i>Parshas Shemos</i> does the Torah choose to hint that the Jews are compared specifically to stars? So the <i>Sfas Emes</i> says that here we're talking about the Jews going into <i>galus</i>. Right? The Jews going into <i>galus</i>. Like someone at first might look at it and say, \\\"Oh, the Jews are going into <i>galus</i>. What does that mean? That means that Hashem doesn't care about them, that means Hashem doesn't need them anymore, it means Hashem doesn't want them anymore, they're not important anymore.\\\" And he says: no, <i>davka</i> the Jews are compared to stars. What's unique about stars? Right now you look out the window, you see the million and billions of star... no, you don't. You can't see the stars during the day. You can only see the stars at night. When it's light outside, you don't see the stars, when they make no <i>nafka mina</i>. <i>Davka</i> they're there, but you don't notice them. <i>Davka</i> when it's dark, that's when the stars shine, that's when they do their thing. He says: no, <i>davka</i> when it's dark, then the Jews shine even brighter. <i>Davka</i> in the <i>galus</i>, not that the <i>galus</i> means the Jews aren't <i>choshuv</i>—putting out the lights, putting the Jews in <i>galus</i>. No, <i>davka</i> in <i>galus</i> is giving the Jews the opportunity to shine, because the tougher things are, the more they can rise to the challenge and the more they can pass <i>nisyonos</i> and the more they can accomplish tremendous things. He even goes so far if you want to be a little <i>Chasidish</i> say: <i>galus</i> is <i>milashon giluy</i>, <i>giluy Shechina</i>. <i>Galus</i> is when they're <i>megaleh</i> the <i>Shechina</i> in the world. But I think that's a very... it's a very important <i>yesod</i>. It's something we've definitely seen in the last couple of months here in <i>Eretz Yisrael</i> that, you know, tough times, tough times, you know, whatever, tough times obviously mean a lot of negative things, we're not getting into that now. But <i>davka</i> one of the reasons Hashem brings tough times on us is because it gives us the opportunity to shine, that since times have been tough the last few months, Jews have shown levels of <i>chessed</i> and levels of <i>emunah</i> and levels of <i>achdus</i> and levels of <i>gevurah</i> that they would never have a chance to shine so much if it was <i>shana k'tikuna</i>, if everything was smooth, if everything... if everything was easy. The darkness comes, says the <i>Sfas Emes</i> the very beginning of <i>Sefer Galus uGeula</i>, says <i>Rashi</i> tells us the darkness comes to give us an opportunity, to give us an opportunity to shine. And he also says that that's <i>davka</i> what it means that Hashem brings each of the <i>kochavim</i> with their <i>shemos</i>. What does a <i>shem</i>... why does he need a <i>shem</i> for all the <i>kochavim</i>, why not just a <i>mispar</i>? Why is it \\\"<i>v'eileh shemos</i>\\\"? A <i>mispar</i> means I need a lot of things. You know what? It's better to have a hundred gold coins than fifty gold coins. It's better to have, you know, thirty kilo of flour than to have twenty kilo of flour. But a name is different. A name means that no, no, no, each one is unique, each one is different. When Hashem gives a name to the <i>kochavim</i>, say I just don't want a lot of <i>kochavim</i>, I have a specific <i>tafkid</i>—he says a name is a <i>tafkid</i>—I have a specific <i>tafkid</i> in mind for each one. Each one is different and has its own <i>tafkid</i>. And he says that's also us in <i>galus</i>. When things are tough, when the Jews have to shine, it's not just that he wants, \\\"Oh, you know, I'll make twice as many Jews so I'll have twice as much light, it'll be a fifty-watt Jewish people instead of a twenty-five-watt Jewish people.\\\" No, each Jew has their own... has their own name. When it's dark, when it's time for the Jews to shine, each Jew has their own <i>tafkid</i> and their own name, their own way to shine, and if you say, \\\"Come on, there's millions and millions of them, really each one has their own <i>tafkid</i>?\\\" Just like there's millions of stars and Hashem didn't make it <i>stam</i>, but there's a special <i>tafkid</i> for each one. Likewise, there might be millions of Jews and it might say, \\\"What difference does one make?\\\" But no, <i>v'eileh shemos b'nei Yisrael</i>. <i>Davka</i> in the tough times is when you're ready to shine, and Hashem has a special <i>tafkid</i> for each person to shine in their own way that he expects of us. And I think that's, you know, both in... when things are tough for <i>Klal Yisrael</i>. It's important to remember that one of the reasons things are tough is so that we can shine, and we don't all shine in the exact same way, we each have our own <i>tafkid</i> to shine. And I think when things are tough, even if it's someone's personal business, someone's community, someone's family, there's a lot of tough times in the world and it's important to remember that sometimes one aspect of the tough times is <i>Hashem</i> darkens to give us the opportunity to shine, to bring out what we can contribute, and <i>vekulam beshem yikra</i>, and that each one of us has our own <i>tafkid</i>, our own way to shine in the darkness, and <i>be'ezrat Hashem</i> if we realize that, then we can shine like the stars that we really are, and we can go from the initial <i>Perek Aleph</i> of <i>Galus</i> to the main theme of <i>Sefer Shmos</i>, which is the ultimate <i>Geulah</i>. <i>Shabbat Shalom</i>."
    """

    #gets all classifications
    results = classify_transcript(transcript)
    #gets just citations
    citations = get_citation_sentences(results)

    for sentence in citations:
        print(sentence)