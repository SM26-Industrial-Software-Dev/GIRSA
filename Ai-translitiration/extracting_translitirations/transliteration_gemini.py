import os, json, time
from google import genai
from google.genai import types
from google.genai.errors import APIError

client = genai.Client() # Expects GEMINI_API_KEY environment variable

INPUT_FILE = "byt5_dataset_prep/extracted_phrases.json"
OUTPUT_FILE = "byt5_dataset_prep/English_Hebrew.jsonl"

# 1. CRASH RECOVERY: Checks what is already done so we can pause/resume safely
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    completed.add(json.loads(line)["input"].strip().lower())
                except: continue
print(f"Safety Engine: Skipping {len(completed)} lines already saved on disk.")

# 2. LOAD DATA (Top-Down Sequence)
if not os.path.exists(INPUT_FILE):
    print(f"Error: Could not find '{INPUT_FILE}'.")
    exit()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    all_phrases = json.load(f)

# Keep only the rows that haven't been completed yet, maintaining top-down order
phrases = [str(p).strip() for p in all_phrases if str(p).strip() and str(p).strip().lower() not in completed]

print(f"Loaded database. {len(phrases)} new elements remaining to process sequentially from the top.")

if not phrases:
    print("Success! All phrases have already been processed.")
    exit()

# 3. TEXT-STREAM INSTRUCTIONS (Saves Input Tokens)
system_instruction = (
    "You are a raw text processor. Convert the provided numbered list of English phrases "
    "into standard Hebrew characters without vowel points (Nikud). For each item, output "
    "the original English phrase, followed by a pipe character '|', followed by the Hebrew translation. "
    "Do not provide markdown, headers, intro, or wrap text. Example format:\n"
    "Abaye amar assur|אביי אמר אסור\n"
    "Abaye amar pasul|אביי אמר פסול"
)

# 4. CHUNK PROCESSING LOOP (400-item chunks fits your remaining daily requests)
chunk_size = 750  
with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
    for i in range(0, len(phrases), chunk_size):
        chunk = phrases[i : i + chunk_size]
        print(f"Sending batch lines {i} to {i + len(chunk)} from the top down...")

        formatted_list = "\n".join([f"{idx+1}. {item}" for idx, item in enumerate(chunk)])
        user_prompt = f"Convert this list:\n{formatted_list}"
        
        retries = 5
        wait_time = 4
        success = False
        
        while retries > 0 and not success:
            try:
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=user_prompt,
                    config={"system_instruction": system_instruction, "temperature": 0.0}
                )
                
                # 5. PYTHON INTERNAL TEXT PARSING TO JSONL
                lines = response.text.strip().split("\n")
                valid_saved = 0
                
                for line in lines:
                    if "|" in line:
                        parts = line.split("|", 1)
                        eng = parts[0].strip()
                        
                        # Clean up any leftover list numbering anomalies (e.g. "1. Abaye" -> "Abaye")
                        if eng and eng[0].isdigit():
                            eng = eng.split(".", 1)[-1].strip()
                            
                        heb = parts[1].strip()
                        
                        if eng and heb:
                            row = {
                                "input": eng,
                                "target": heb
                            }
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                            valid_saved += 1
                
                out.flush() # Secure progress instantly to your hard drive
                success = True
                print(f"Successfully processed batch! Saved {valid_saved} records directly to disk.")
                time.sleep(4) # Rate-limiting minute safety buffer
                
            except APIError as api_err:
                err_str = str(api_err)
                if "429" in err_str or "Quota" in err_str or "503" in err_str:
                    print(f"Server Throttled ({api_err.code}). Backing off for {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2
                    retries -= 1
                else:
                    print(f"Gemini API critical failure: {api_err}")
                    break
            except Exception as e:
                print(f"Parsing error on index {i}: {e}. Retrying execution block...")
                time.sleep(2)
                retries -= 1
                
        if not success:
            print("\nShutting down pipeline loop to preserve progress stability.")
            break

print("Processing loop finalized successfully.")