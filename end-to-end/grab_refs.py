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
Put transcript here
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