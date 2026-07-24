import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

# 1. Setup Device Globally
device = torch.device("cpu")
print(f"[SYSTEM] Using device: {device}")

# Initialize local model objects
local_dir = Path(__file__).parent / "NER"
model_path = str(local_dir.resolve())

label_list = [
    "O",
    "B-AUTHORITY_OR_SOURCE",
    "I-AUTHORITY_OR_SOURCE",
    "B-BOOK_LOCATION",
    "I-BOOK_LOCATION",
    "B-SOURCE_PATH",
    "I-SOURCE_PATH",
]
num_valid_classes = len(label_list)

print(f"[SYSTEM] Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(
    model_path, local_files_only=True
).to(device)
model.eval()


def process_sentence(sentence: str):
        words = sentence.split()
        inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt")
        # Guard against single massive sentences overflowing 512 bounds
        if inputs["input_ids"].shape[1] > 512:
            inputs = tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            clean_logits = outputs.logits[:, :, :num_valid_classes]
            predictions = torch.argmax(clean_logits, dim=2).squeeze(0).cpu().numpy()

        word_ids = (
            tokenizer(words, is_split_into_words=True, truncation=True, max_length=512).word_ids()
            if inputs["input_ids"].shape[1] == 512
            else tokenizer(words, is_split_into_words=True).word_ids()
        )
        previous_word_idx = None

        entities_found = False
        current_entity_tokens = []
        current_entity_label = None

        def flush_entity_block():
            nonlocal current_entity_tokens, current_entity_label, entities_found
            if current_entity_tokens and current_entity_label:
                full_word = " ".join(current_entity_tokens)
                print(f" {current_entity_label}] -> '{full_word}'")
                entities_found = True
            current_entity_tokens = []
            current_entity_label = None

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue

            pred_id = predictions[token_idx]
            label_string = label_list[pred_id]
            raw_word = words[word_idx]

            if label_string.startswith("B-"):
                flush_entity_block()
                current_entity_label = label_string.split("-")[1]
                current_entity_tokens = [raw_word]
            elif label_string.startswith("I-"):
                expected_label = label_string.split("-")[1]
                if current_entity_label == expected_label:
                    current_entity_tokens.append(raw_word)
                else:
                    flush_entity_block()
            else:
                flush_entity_block()

            previous_word_idx = word_idx

        flush_entity_block()

# 2. Execution Block
if __name__ == "__main__":
    print("\n" + "=" * 60)
    # Replace with the actual path to your structured JSON file
    text = """
    The רמב"ן in the איגרת הרמב"ן writes to his son that he advises him להתבושש מכל אדם ולהתפחד ממנו ומן החטא.
"""
    process_sentence(text)
    print("=" * 60 + "\n")
