import re
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ==========================================
# 1. HARDWARE ACCELERATION CONFIG
# ==========================================
BATCH_SIZE = 512  
MODEL_PATH = "final_hebrew_transliteration_model"

print("Loading model directly into 16-bit Tensor Cores...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
torch.backends.cudnn.benchmark = True


# ==========================================
# 2. DATA PREPARATION PIPELINE
# ==========================================
class TranscriptDataset(Dataset):
    def __init__(self, db_path):
        self.segments_list = []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("Reading complete text data from full_transcripts table...")
        cursor.execute("SELECT id, text, shiur_id FROM full_transcripts")
        rows = cursor.fetchall()
        conn.close()
        
        # Capture raw matches inside <i></i> tags
        tag_regex = re.compile(r"<i>(.*?)</i>", re.DOTALL)
        
        for row_id, text_content, shiur_id in rows:
            if not text_content:
                continue
                
            matches = tag_regex.findall(text_content)
            for match in matches:
                cleaned_match = match.strip()
                word_count = len(cleaned_match.split())
                
                # Filter Condition: Keep segments with fewer than 25 words
                if 0 < word_count < 25: 
                    self.segments_list.append({
                        "id": str(row_id),
                        "shiur_id": int(shiur_id) if shiur_id is not None else 0,
                        "original_full_text": text_content,
                        "tag_raw_match": match,             
                        "text_to_process": cleaned_match    
                    })
        
        print(f"✅ Text preparation complete. Found {len(self.segments_list)} total segments to run.")

    def __len__(self):
        return len(self.segments_list)

    def __getitem__(self, idx):
        return self.segments_list[idx]


def smart_collate_fn(batch):
    texts_to_process = [item["text_to_process"] for item in batch]
    ids = [item["id"] for item in batch]
    shiur_ids = [item["shiur_id"] for item in batch]
    full_texts = [item["original_full_text"] for item in batch]
    raw_matches = [item["tag_raw_match"] for item in batch]
    
    inputs = tokenizer(texts_to_process, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return inputs, ids, shiur_ids, full_texts, raw_matches


# ==========================================
# 3. FULL PRODUCTION PROCESSING ENGINE
# ==========================================
def run_production_gpu_pipeline(db_path):
    # STEP 1: IMMEDIATELY DROP THE INTERMEDIATE TABLE
    print("🧹 Wiping any existing table to ensure a clean database state...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS transliterations_removed;")
    conn.commit()
    conn.close()
    
    dataset = TranscriptDataset(db_path)
    if len(dataset) == 0:
        print("No matching text segments found. Exiting.")
        return
        
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=smart_collate_fn,
        num_workers=4,          
        pin_memory=True,        
        shuffle=False
    )
    
    reconstructed_rows = {}
    print(f"\n🚀 Deploying Production Pipeline across all data rows...")
    
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for batch_idx, (inputs, ids, shiur_ids, full_texts, raw_matches) in enumerate(dataloader):
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            
            outputs = model.generate(**inputs, max_length=64, num_beams=3, early_stopping=True)
            decoded_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for i in range(len(ids)):
                row_id = ids[i]
                shiur_id = shiur_ids[i]
                ai_output = decoded_results[i]
                raw_match = raw_matches[i]
                
                if row_id not in reconstructed_rows:
                    reconstructed_rows[row_id] = {
                        "shiur_id": shiur_id,
                        "current_text": full_texts[i] 
                    }
                
                # In-place substitution: swap <i>english</i> with hebrew
                target_string = f"<i>{raw_match}</i>"
                reconstructed_rows[row_id]["current_text"] = reconstructed_rows[row_id]["current_text"].replace(
                    target_string, ai_output
                )
            
            if (batch_idx + 1) % 10 == 0:
                print(f"GPU Engine Progress: Finished batch {batch_idx + 1}... Hardware saturated.")

    # STEP 2: OPEN SQL PIPELINE AND COMMIT EVERYTHING IN ONE BULK OPERATION
    print("\n📦 Generating final database table and configuring ultra-fast WAL streams...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=OFF;")
    
    cursor.execute("""
        CREATE TABLE transliterations_removed (
            id TEXT PRIMARY KEY,
            shiur_id INTEGER,
            transliterated_hebrew TEXT
        )
    """)
    
    print("Writing reconstructed production text records directly to your hard drive...")
    
    sql_insert_payload = []
    for row_id, content in reconstructed_rows.items():
        sql_insert_payload.append((row_id, content["shiur_id"], content["current_text"]))
    
    # Bulk speed insert execution
    cursor.executemany(
        "INSERT INTO transliterations_removed (id, shiur_id, transliterated_hebrew) VALUES (?, ?, ?)",
        sql_insert_payload
    )
    
    conn.commit()
    conn.close()
    print("\n🏁 PRODUCTION RUN COMPLETE! The entire table has been successfully processed and updated.")


if __name__ == "__main__":
    # Supply your master database file path here
    run_production_gpu_pipeline("transcripts.db")
