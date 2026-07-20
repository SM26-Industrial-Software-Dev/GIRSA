import os
import json
import torch
import numpy as np
from pathlib import Path
from ner_trainer import load_and_merge_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report

# ==========================================
# 1. Setup Architecture Foundations
# ==========================================
local_dir = Path(__file__).parent / "best_rabbinic_ner_model"
model_path = str(local_dir.resolve())

label_list = [
    "O", 
    "B-AUTHORITY_OR_SOURCE", "I-AUTHORITY_OR_SOURCE",
    "B-BOOK_LOCATION", "I-BOOK_LOCATION",
    "B-SOURCE_PATH", "I-SOURCE_PATH"
]
num_valid_classes = len(label_list)

print(f"[SYSTEM] Loading fine-tuned local weights to CPU: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True).cpu()
model.eval()

# ==========================================
# 2. Extract the Isolated Test Data Split
# ==========================================
raw_dataset = load_and_merge_dataset("ner_data.json", "transcripts.db")
# Using the exact same seed (42) guarantees we evaluate on the exact same 20% validation split your trainer used!
split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
test_data = split_dataset["test"]

all_true_labels = []
all_pred_labels = []

print(f"[SYSTEM] Running direct evaluation loops over {len(test_data)} test transcripts...")

# ==========================================
# 3. Safe Sentence-Chunk Evaluation Loop
# ==========================================
# We process token by token to prevent index overflows (512 token boundary) [INDEX]
for i in range(len(test_data)):
    words = test_data[i]["tokens"]
    true_tags = test_data[i]["ner_tags"]
    
    # Sentencepiece tokenization matching training configurations [INDEX]
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt")
    
    # Boundary guard: Truncate sentence if it somehow hits the 512-word model wall [INDEX]
    if inputs["input_ids"].shape[1] > 512:
        inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        
    with torch.no_grad():
        outputs = model(**inputs)
        # Constrain output logits to our valid 7 index positions to prevent out of bounds loops [INDEX]
        clean_logits = outputs.logits[:, :, :num_valid_classes]
        predictions = torch.argmax(clean_logits, dim=2).squeeze(0).numpy()
        
    word_ids = inputs.word_ids()
    previous_word_idx = None
    
    for idx, word_idx in enumerate(word_ids):
        # Ignore special structural packaging tokens ([CLS], [SEP], padding) [INDEX]
        if word_idx is None or word_idx == previous_word_idx:
            continue
            
        pred_id = predictions[idx]
        actual_id = true_tags[word_idx]
        
        # Append target names directly if they sit inside valid boundaries [INDEX]
        if actual_id < num_valid_classes and pred_id < num_valid_classes:
            all_true_labels.append(label_list[actual_id])
            all_pred_labels.append(label_list[pred_id])
            
        previous_word_idx = word_idx

# ==========================================
# 4. Print Category-by-Category Report
# ==========================================
print("\n" + "="*60)
print("       PER-CLASS METRIC EVALUATION REPORT")
print("="*60)

# Generate a detailed classification metrics matrix report [INDEX]
report = classification_report(
    y_true=all_true_labels,
    y_pred=all_pred_labels,
    labels=[l for l in label_list if l != "O"], # Filter out 'O' to see true class mechanics clearly [INDEX]
    digits=4
)
print(report)
print("="*60)
