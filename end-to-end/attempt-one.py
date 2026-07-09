from build_search_queries import build_search_query
from grab_refs import extract_sources_from_transcript
from build_sefaria_link import create_link
from api_call import get_best_source_link


def main():
    """
    Get the output of the RAG as an array of lists of dictionaries
    """
    transcript = """
    let's talk about it in a more practical example I'll give you an interesting case the <i>kli</i> becoming <i>assur</i> that's what it says right let's imagine I want to do the following I want to add some water to my <i>cholent</i> okay let's say it's <i>erev Shabbos</i> it's cold water we'll talk about on Shabbos and adding hot water in a minute but cold water so I take my I don't know my <i>pareve</i> measuring cup and I pour the cold water into my <i>cholent</i> or my chicken soup or something like that so and there's a stream so somehow or another my chicken soup made my <i>pareve</i> measuring cup or maybe <i>milchig</i> cup <i>fleishig</i> in other words the <i>fleishig bleos</i> somehow or another went upstream right this is reverse <i>irui</i> till now we've always been talking about <i>irui</i> right where the thing on top assurs the thing below but the <i>Rama</i> says you know what the thing below could go upstream and assur the thing on top that's what he says right but then he says yeah that's <i>l'chatchila</i> not <i>bedieved</i> <i>bedieved ein lachosh</i> so what in the world is he talking about over here so this comes from the <i>Mordechai</i> again it's in the <i>Rama</i> at the end of <i>sif-gimmel</i> comes from the <i>Mordechai</i> in <i>Chullin</i> תשטו and he brings a <i>raya</i> from the <i>Mishna</i> in <i>Machshirin</i> <i>perek yud mishna</i> פרק ה משנה י it says over there המערה טהור לטמא מחם לחם ומצונן לצונן <i>mutar</i> or <i>tahor</i> <i>slicha</i> <i>mitzonen l'cham tameh</i> so you're pouring something <i>tahor</i> into something <i>tameh</i> so if it's hot into hot or cold into cold it's okay but cold into hot it becomes <i>tameh</i> so why let's first talk let's analyze the <i>Mishna</i> on its own and then we'll see how to apply it to our <i>deiah</i> so if we look at the <i>Rambam</i> over there he says <i>hatam amru</i> this is <i>peirush hamishnayos</i> הטעם אמרו מצונן לחם טמא why is it that when you pour the cold water into the <i>tsonem l'chamtam</i>, the hot water, it's <i>tameh</i>. לפי שאמר המשקה הנקרא ניצוק, in other words, so there is this <i>nitzuk</i>, there is this continuous stream from the upper one to the lower one. <i>mitchamem</i>, it gets heated, because remember going cold into hot. So the hot below heats up the stream, right again, this is reverse <i>iruy</i>. מחמת שקציו בכלי חם, because part of it is touching the hot thing. So the heat goes upstream to the top. But then he adds something: ויעלו ההבל עם הניצוק. So going upstream, there seems to be two factors according to the <i>Rambam</i>: <i>hevel</i> plus <i>nitzuk</i>. מתוך הכלי חם שהוא טמא ויחמם את המשקה אשר בכלי העליון. And this will heat up the <i>mashke</i> in the upper <i>kli</i>. ולפיכך יהא טמא מפני שהאד החם חיבר ביניהם והוא עולה מן הטמא לטהור. And I think the <i>Taz</i> is more or less by us <i>sif katan vav</i> is taking the same approach again סימן ק\"ה סיף קטן ו', quoting from the <i>Issur v'Heter</i> that it's because דסא קבר ומחברו בניצוק. Now if you look at the <i>Rav</i> over there in the <i>Mishna</i>, he says צונן לחמתן טמא שהתחתון חם מעלה הבל לעליון הצונן ומטמאו. He just says <i>hevel</i>, he didn't say <i>nitzuk</i>. And that's sort of the <i>mashma'us</i> here of the <i>Shach</i> סיף קטן י\"א. He just says <i>hevel</i>, he doesn't say anything about a <i>nitzuk</i>. So anyway, <i>al kol panim</i>, what does this mean? So the <i>Chochmas Adam</i> over here כלל נ\"ט סיף ה' ד' ה' says, therefore, if you go from hot to hot, it's going to be okay. Right? The opposite of what the <i>Yad</i> said. Because the <i>Mishna</i> was only concerned about cold to hot. So in this case, if you wanted to go, let's say we talked about this last year in <i>Hilchos Shabbos</i>, you know your cholent is burning, we talked about a question: could you go ahead and I don't know let's say you put your urn under your cholent pot full of drying out cholent under your urn.
    """
    sentences = extract_sources_from_transcript(transcript)
    output = {}
    for sentence in sentences:
        sentenceID = sentence["id"]
        sourceInput = sentence["sources_found"]
        sourceOutput = []
        for source in sourceInput:
            if source.get("text") is None:
                sourceOutput.append(create_link(source))
            else:
                call = build_search_query(source)
                best_link = get_best_source_link("http://localhost:8000", call)
                sourceOutput.append(best_link)
        output[sentenceID] = sourceOutput
    print(output)


if __name__ == "__main__":
    main()
# import nltk
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
# def halveSentence(sentence: str):
#   words = sentence.split()
#   mid = len(words) // 2
#   a = " ".join(words[:mid])
#   b = " ".join(words[mid:])
#   return a, b
#
#
# def chunkTranscript(transcriptText: str):
#    sentences = nltk.sent_tokenize(transcriptText)
#    halfSentences = []
#    for sentence in sentences:
#        a, b = halveSentence(sentence)
#        halfSentences.append(a)
#        halfSentences.append(b)
#    return sentences, halfSentences