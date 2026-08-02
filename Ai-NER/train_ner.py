import json
import sys
import torch
import numpy as np
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# ---------------------------------------------------------
# 0. CUDA Hardware Check
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Checking Compute Device: {device}")

if device.type != "cuda":
    print("⚠️ WARNING: CUDA GPU was not detected! Training will fall back to CPU (much slower).")
    print("If you have an NVIDIA GPU, make sure PyTorch with CUDA support is installed.")
else:
    print(f"🚀 CUDA Detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ---------------------------------------------------------
# 1. Label Mappings
# ---------------------------------------------------------
LABEL_LIST = ["O", "B-TITLE", "I-TITLE", "B-PATH", "I-PATH"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

MODEL_CHECKPOINT = "xlm-roberta-base"

# ---------------------------------------------------------
# 2. Dataset Preparation & Token Alignment
# ---------------------------------------------------------
def load_labeled_data(data_path: str) -> Dataset:
    """Reads output_bio_ner.json and converts to Hugging Face Dataset format."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    formatted_data = {
        "id": [item["sentence_id"] for item in data],
        "tokens": [item["tokens"] for item in data],
        "ner_tags": [[LABEL2ID[tag] for tag in item["bio_tags"]] for item in data]
    }
    return Dataset.from_dict(formatted_data)


def tokenize_and_align_labels(examples, tokenizer, max_length: int = 128):
    """Aligns word-level BIO tags with XLM-RoBERTa SentencePiece subwords."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# ---------------------------------------------------------
# 3. Metrics Calculation (Seqeval)
# ---------------------------------------------------------
seqeval_metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# ---------------------------------------------------------
# 4. Main Fine-Tuning Execution
# ---------------------------------------------------------
def train():
    print(f"\nLoading Tokenizer & Model ({MODEL_CHECKPOINT})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # Load model and send parameters directly to CUDA device
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    # 1. Load Dataset
    raw_dataset = load_labeled_data("output_bio_ner.json")
    
    # 2. Train/Validation Split (80% Train, 20% Val)
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
    
    # 3. Tokenize and align labels
    tokenized_dataset = split_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names
    )

    # 4. CUDA-Optimized Training Arguments
    use_cuda = device.type == "cuda"
    
    training_args = TrainingArguments(
        output_dir="./xlm_roberta_ner_cuda",
        eval_strategy="epoch",            # Changed from evaluation_strategy to avoid deprecation warnings
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        
        # --- CUDA Optimization Flags ---
        fp16=use_cuda,                    # Enables mixed precision for faster GPU training
        dataloader_pin_memory=use_cuda,   # Speeds up memory transfers from CPU to GPU
        report_to="none"
    )

    # 5. Initialize Trainer
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. Run Training on GPU
    print("\nStarting Training on CUDA GPU...")
    trainer.train()

    print("\nSaving Fine-Tuned Model...")
    trainer.save_model("./xlm_roberta_ner_cuda/final")
    tokenizer.save_pretrained("./xlm_roberta_ner_cuda/final")
    print(" Model saved to './xlm_roberta_ner_cuda/final'")

if __name__ == "__main__":
    train()