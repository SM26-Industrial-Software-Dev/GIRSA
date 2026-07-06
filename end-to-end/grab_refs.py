import os
import re
import json
from typing import List, Dict, Any
from groq import Groq

def split_into_sentences(transcript: str) -> List[str]:
    print("[1/3] Splitting transcript into sentences...")
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    valid_sentences = [s.strip() for s in sentences if s.strip()]
    print(f" Found {len(valid_sentences)} total sentences.")
    return valid_sentences

def extract_sources_from_transcript(transcript: str, batch_size: int = 5, context_window: int = 2) -> List[Dict[str, Any]]:
    print("[2/3] Initializing Groq client...")
    
    if not os.environ.get("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY environment variable is not set!")
        return []
        
    client = Groq()
    sentences = split_into_sentences(transcript)
    all_extracted_data: List[Dict[str, Any]] = []
    
    print(f"[3/3] Processing batches of size {batch_size} (with context window of {context_window})...")
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        batch_dict = {i + idx: text for idx, text in enumerate(batch)}
        
        prev_context = sentences[max(0, i - context_window) : i]
        context_before_dict = {i - len(prev_context) + idx: text for idx, text in enumerate(prev_context)}
        
        prompt = f"""
You are an expert in Jewish texts and Rabbinic literature.
Analyze the following batch of numbered sentences from a shiur transcript and identify quotations of valid Torah books.

You MUST respond with a single JSON object matching this exact structural schema:
{{
  "results": [
    {{
      "id": int,
      "sources_found": [
        {{
          "category": "string",
          "book": "string",
          "addr1": int or null,
          "addr2": int or null,
          "text": "string or null"
        }}
      ]
    }}
  ]
}}

CRITICAL INSTRUCTIONS & RULES:
1. NO BOOK = SKIP: If a sentence doesn't explicitly name a book, SKIP IT entirely. Do not include it in the results array.
2. ADDR1 & ADDR2 MUST BE STRICT INTEGERS: Never put text, quotes, or alphabetic suffixes in 'addr1' or 'addr2'. Only one exception: A daf of gemara can have a letter 'a' or 'b' at the end.
3. text: If a complete reference to a book is given, text should be left null. Otherwise, text should be the quotation of the book quoted in the sentence
4. Rule: NEVER make up anything. If it's not mentioned, leave it null. Addresses may be left null, but if they are both null the text cannot be null

--- CONTEXT BEFORE ---
{json.dumps(context_before_dict)}

--- SENTENCES TO ANALYZE ---
{json.dumps(batch_dict)}
"""
        print(f" ⏳ Sending batch (Sentences {i} to {i + len(batch) - 1}) to Groq...")
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a highly logical assistant that strictly outputs JSON and mathematically converts all Hebrew addresses to Base-10 integers."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            raw_json = response.choices[0].message.content
            parsed_data = json.loads(raw_json)
            
            batch_results = parsed_data.get("results", [])
            all_extracted_data.extend(batch_results)
            print(f" ✅ Batch completed! Appended {len(batch_results)} results.")

        except Exception as e:
            print(f" ❌ API Error encountered during this batch: {e}")

    print(" Processing complete.")
    return all_extracted_data

# ==========================================
# Example Execution (Testing Block)
# ==========================================
if __name__ == "__main__":
    def decode_talmudic_addr(val: int) -> str:
        if val is None:
            return "null"
        daf = val // 10
        side_digit = val % 10
        amud = "a" if side_digit == 1 else "b"
        return f"{daf}{amud}"

    sample_transcript = """
let's talk about it in a more practical example I'll give you an interesting case the <i>kli</i> becoming <i>assur</i> that's what it says right let's imagine I want to do the following I want to add some water to my <i>cholent</i> okay let's say it's <i>erev Shabbos</i> it's cold water we'll talk about on Shabbos and adding hot water in a minute but cold water so I take my I don't know my <i>pareve</i> measuring cup and I pour the cold water into my <i>cholent</i> or my chicken soup or something like that so and there's a stream so somehow or another my chicken soup made my <i>pareve</i> measuring cup or maybe <i>milchig</i> cup <i>fleishig</i> in other words the <i>fleishig bleos</i> somehow or another went upstream right this is reverse <i>irui</i> till now we've always been talking about <i>irui</i> right where the thing on top assurs the thing below but the <i>Rama</i> says you know what the thing below could go upstream and assur the thing on top that's what he says right but then he says yeah that's <i>l'chatchila</i> not <i>bedieved</i> <i>bedieved ein lachosh</i> so what in the world is he talking about over here so this comes from the <i>Mordechai</i> again it's in the <i>Rama</i> at the end of <i>sif-gimmel</i> comes from the <i>Mordechai</i> in <i>Chullin</i> תשטו and he brings a <i>raya</i> from the <i>Mishna</i> in <i>Machshirin</i> <i>perek yud mishna</i> פרק ה משנה י it says over there המערה טהור לטמא מחם לחם ומצונן לצונן <i>mutar</i> or <i>tahor</i> <i>slicha</i> <i>mitzonen l'cham tameh</i> so you're pouring something <i>tahor</i> into something <i>tameh</i> so if it's hot into hot or cold into cold it's okay but cold into hot it becomes <i>tameh</i> so why let's first talk let's analyze the <i>Mishna</i> on its own and then we'll see how to apply it to our <i>deiah</i> so if we look at the <i>Rambam</i> over there he says <i>hatam amru</i> this is <i>peirush hamishnayos</i> הטעם אמרו מצונן לחם טמא why is it that when you pour the cold water into the <i>tsonem l'chamtam</i>, the hot water, it's <i>tameh</i>. לפי שאמר המשקה הנקרא ניצוק, in other words, so there is this <i>nitzuk</i>, there is this continuous stream from the upper one to the lower one. <i>mitchamem</i>, it gets heated, because remember going cold into hot. So the hot below heats up the stream, right again, this is reverse <i>iruy</i>. מחמת שקציו בכלי חם, because part of it is touching the hot thing. So the heat goes upstream to the top. But then he adds something: ויעלו ההבל עם הניצוק. So going upstream, there seems to be two factors according to the <i>Rambam</i>: <i>hevel</i> plus <i>nitzuk</i>. מתוך הכלי חם שהוא טמא ויחמם את המשקה אשר בכלי העליון. And this will heat up the <i>mashke</i> in the upper <i>kli</i>. ולפיכך יהא טמא מפני שהאד החם חיבר ביניהם והוא עולה מן הטמא לטהור. And I think the <i>Taz</i> is more or less by us <i>sif katan vav</i> is taking the same approach again סימן ק\"ה סיף קטן ו', quoting from the <i>Issur v'Heter</i> that it's because דסא קבר ומחברו בניצוק. Now if you look at the <i>Rav</i> over there in the <i>Mishna</i>, he says צונן לחמתן טמא שהתחתון חם מעלה הבל לעליון הצונן ומטמאו. He just says <i>hevel</i>, he didn't say <i>nitzuk</i>. And that's sort of the <i>mashma'us</i> here of the <i>Shach</i> סיף קטן י\"א. He just says <i>hevel</i>, he doesn't say anything about a <i>nitzuk</i>. So anyway, <i>al kol panim</i>, what does this mean? So the <i>Chochmas Adam</i> over here כלל נ\"ט סיף ה' ד' ה' says, therefore, if you go from hot to hot, it's going to be okay. Right? The opposite of what the <i>Yad</i> said. Because the <i>Mishna</i> was only concerned about cold to hot. So in this case, if you wanted to go, let's say we talked about this last year in <i>Hilchos Shabbos</i>, you know your cholent is burning, we talked about a question: could you go ahead and I don't know let's say you put your urn under your cholent pot full of drying out cholent under your urn.
"""
    
    final_list: List[Dict[str, Any]] = extract_sources_from_transcript(sample_transcript, batch_size=4, context_window=2)
    sentences_lookup = split_into_sentences(sample_transcript)
    
    print("\n" + "="*60)
    print("      FINAL EXTRACTED RESULTS (EVALUATION VIEW)")
    print("="*60)
    print(f"Total valid entries returned by LLM: {len(final_list)}\n")
    
    for result in final_list:
        sentence_id = result.get('id')
        
        if sentence_id is not None and sentence_id < len(sentences_lookup):
            sentence_text = sentences_lookup[sentence_id]
        else:
            sentence_text = "Unknown Sentence (ID Mismatch)"
            
        print(f"Sentence ID: {sentence_id}")
        print(f"Transcript Text: \"{sentence_text}\"")
        
        sources = result.get("sources_found", [])
        if not sources:
            print("  ↳  Sources Found: []")
        else:
            print("  ↳  Sources Found:")
            for idx, source in enumerate(sources, 1):
                cat = source.get("category")
                book = source.get("book")
                a1 = source.get("addr1")
                a2 = source.get("addr2")
                txt = source.get("text")
                
                if cat == "Talmud" and isinstance(a1, int):
                    display_a1 = decode_talmudic_addr(a1)
                else:
                    display_a1 = str(a1)
                
                print(f"      [{idx}] Category: {cat}")
                print(f"          Book:     {book}")
                print(f"          Address:  Primary: {display_a1} | Secondary: {a2}")
                print(f"          Text:     {repr(txt)}")
        print("-" * 50)