from pathlib import Path
import re
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


class NERExtractor:
    def __init__(self, model_dir: Path = None):
        if model_dir is None:
            model_dir = Path(__file__).parent / "NER"

        self.device = torch.device("cpu")
        self.label_list = [
            "O",
            "B-AUTHORITY_OR_SOURCE",
            "I-AUTHORITY_OR_SOURCE",
            "B-BOOK_LOCATION",
            "I-BOOK_LOCATION",
            "B-SOURCE_PATH",
            "I-SOURCE_PATH",
        ]
        self.num_valid_classes = len(self.label_list)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            str(model_dir), local_files_only=True
        ).to(self.device)
        self.model.eval()

    def process_sentence(self, sentence: str) -> dict:
        words = sentence.split()
        if not words:
            return {"AUTHORITY_OR_SOURCE": [], "BOOK_LOCATION": [], "SOURCE_PATH": []}

        inputs = self.tokenizer(words, is_split_into_words=True, return_tensors="pt")
        if inputs["input_ids"].shape[1] > 512:
            inputs = self.tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            clean_logits = outputs.logits[:, :, : self.num_valid_classes]
            predictions = torch.argmax(clean_logits, dim=2).squeeze(0).cpu().numpy()

        word_ids = (
            self.tokenizer(
                words, is_split_into_words=True, truncation=True, max_length=512
            ).word_ids()
            if inputs["input_ids"].shape[1] == 512
            else self.tokenizer(words, is_split_into_words=True).word_ids()
        )

        extracted = {
            "AUTHORITY_OR_SOURCE": [],
            "BOOK_LOCATION": [],
            "SOURCE_PATH": [],
        }

        previous_word_idx = None
        current_entity_tokens = []
        current_entity_label = None

        def flush_entity():
            nonlocal current_entity_tokens, current_entity_label
            if current_entity_tokens and current_entity_label:
                full_word = " ".join(current_entity_tokens)
                if current_entity_label in extracted:
                    extracted[current_entity_label].append(full_word)
            current_entity_tokens = []
            current_entity_label = None

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue

            pred_id = predictions[token_idx]
            label_string = self.label_list[pred_id]
            raw_word = words[word_idx]

            if label_string.startswith("B-"):
                flush_entity()
                current_entity_label = label_string.split("-")[1]
                current_entity_tokens = [raw_word]
            elif label_string.startswith("I-"):
                expected_label = label_string.split("-")[1]
                if current_entity_label == expected_label:
                    current_entity_tokens.append(raw_word)
                else:
                    flush_entity()
            else:
                flush_entity()

            previous_word_idx = word_idx

        flush_entity()
        return extracted


def convert_location_to_addresses(loc_str: str) -> list[int]:
    """
    Parses numerical segments in BOOK_LOCATION strings to map to [addr1, addr2, addr3, addr4].
    """
    nums = [int(n) for n in re.findall(r"\d+", loc_str)]
    addrs = [-1, -1, -1, -1]
    for i in range(min(len(nums), 4)):
        addrs[i] = nums[i]
    return addrs
