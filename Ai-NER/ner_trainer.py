import json
import sqlite3
import torch
import numpy as np
from datasets import Dataset
from torch import nn
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate

# 1. Map Labels to Integers (BIO Schema)
label_list = [
    "O", 
    "B-AUTHORITY_OR_SOURCE", "I-AUTHORITY_OR_SOURCE",
    "B-BOOK_LOCATION", "I-BOOK_LOCATION",
    "B-SOURCE_PATH", "I-SOURCE_PATH"
]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}

def tokenize_words(text: str):
    """The exact same word splitting tokenization used in your original script."""
    return text.split()

# ========================================================
# 2. Dynamic Merging: JSON + SQLite Database
# ========================================================
def load_and_merge_dataset(json_path, db_path):
    """
    Reads your existing JSON progress file, queries transcripts.db to pull 
    the raw text strings, and builds the tokens list on the fly.
    """
    print("[SYSTEM] Reading existing JSON progress records...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_json_data = json.load(f)
        
    print("[SYSTEM] Fetching raw text mappings from transcripts.db...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Fetch all raw text rows to map against our processed JSON keys
        cursor.execute("SELECT id, transliterated_hebrew FROM transliterations_removed")
        db_rows = cursor.fetchall()
        # Convert to an efficient lookup dictionary: { "record_id": "raw text block" }
        db_lookup = {str(row[0]): row[1] for row in db_rows}
    finally:
        conn.close()

    formatted_data = {"tokens": [], "ner_tags": []}
    skipped_count = 0

    print("[SYSTEM] Merging arrays and building word tag sequences...")
    for record_id, content in raw_json_data.items():
        # Grab original text from DB using the record_id key
        transcript_text = db_lookup.get(record_id)
        
        if not transcript_text:
            skipped_count += 1
            continue
            
        # Reconstruct the precise word tokens array dynamically
        tokens = tokenize_words(transcript_text)
        
        # Initialize all tags to 'O'
        tags = [label_to_id["O"]] * len(tokens)
        
        # Overlay the labeled entities using your calculated index bounds
        for entity in content.get("entities", []):
            # Double-check that your keys map correctly to your JSON structure
            start = entity.get("start_word_idx", -1)
            end = entity.get("end_word_idx", -1)
            label = entity.get("label") # <--- Make sure this points to 'label'
            
            # Skip invalid extractions or missing labels
            if start == -1 or end == -1 or not label:
                continue
                
            # Apply BIO tagging strategy safely
            if f"B-{label}" in label_to_id:
                tags[start] = label_to_id[f"B-{label}"]
                for i in range(start + 1, end + 1):
                    if i < len(tags) and f"I-{label}" in label_to_id:
                        tags[i] = label_to_id[f"I-{label}"]
                    
        formatted_data["tokens"].append(tokens)
        formatted_data["ner_tags"].append(tags)
        
    print(f"[SYSTEM] Cleanly mapped {len(formatted_data['tokens'])} rows. (Skipped {skipped_count} missing from DB context)")
    return Dataset.from_dict(formatted_data)

# ========================================================
# 3. Tokenize & Align Subwords Safely for XLM-RoBERTa
# ========================================================
def tokenize_and_align_labels(examples, tokenizer):
    """
    Safely tokenizes word lists, automatically chunking documents longer 
    than 512 sub-tokens with a sliding window so no trailing text is lost.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
        stride=32,                          # Keeps a 32-token overlap between chunks
        return_overflowing_tokens=True,     # Splits long text into multiple rows
    )
    
    # Map each chunk back to its original document index
    sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
    labels = []
    
    # --- FIX: Iterate using a range to pass the precise batch index ---
    num_chunks = len(tokenized_inputs["input_ids"])
    for i in range(num_chunks):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # <--- Explicit index fixes NoneType bug
        original_sample_idx = sample_mapping[i]
        label = examples["ner_tags"][original_sample_idx]
        
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Ignore special padding tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) # First sub-token gets the tag
            else:
                label_ids.append(-100) # Ignore subsequent sub-tokens
            previous_word_idx = word_idx
            
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# ========================================================
# 4. Evaluation Metrics & Main Trainer Entrypoint
# ========================================================
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    model_checkpoint = "xlm-roberta-base"
    json_path = "ner_data.json"
    db_path = "transcripts.db"
    
    # 1. Initialize tokenizer and model structures
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, 
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id
    )

    # 2. Build the dataset using the dynamic DB lookup merge function
    raw_dataset = load_and_merge_dataset(json_path, db_path)
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
    
    # 3. Process sub-word tokens
    tokenized_datasets = split_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), 
        batched=True,
        remove_columns=split_dataset["train"].column_names  # <--- THIS LINE FIXES THE ARROW INVALID ERROR
    )

    # 4. Configure training arguments
    training_args = TrainingArguments(
        output_dir="./xlm-roberta-rabbinic-ner",
        learning_rate=1e-5,                 # <--- Lowered (from 1.5e-5) to fine-tune boundaries carefully
        per_device_train_batch_size=4,      
        per_device_eval_batch_size=4,
        num_train_epochs=6,                 # <--- Increased (from 4) to give the model time to balance out
        weight_decay=0.03,                  # <--- Slightly higher to prevent overfitting on the extra epochs
        eval_strategy="epoch",      
        save_strategy="epoch",              
        save_total_limit=1,                 
        load_best_model_at_end=True,
        metric_for_best_model="f1",         
        logging_steps=10,
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[SYSTEM] Starting XLM-RoBERTa Training Loop...")
    trainer.train()
    
    print("[SYSTEM] Training complete! Saving final model weights...")
    trainer.save_model("./best_rabbinic_ner_model")
    tokenizer.save_pretrained("./best_rabbinic_ner_model")

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Define weights proportional to inverse class frequencies
        # (Giving significantly higher math rewards to BOOK_LOCATION and SOURCE_PATH)
        # Order maps directly to your label_list index layout
        class_weights = torch.tensor([1.0, 1.0, 1.0, 2.2, 2.2, 6.6, 6.6]).cuda()
        
        loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    main()