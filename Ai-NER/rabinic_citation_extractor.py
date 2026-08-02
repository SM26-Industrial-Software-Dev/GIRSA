import json
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from gematria import hebrew_to_values, word_value
import regex


class RabbinicCitationExtractor:

    def __init__(
        self, relative_model_path: str = "./xlm_roberta_ner_cuda/final"
    ):
        self.model_path = os.path.abspath(relative_model_path)

        # Check if the trained model directory exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model path '{self.model_path}' does not exist! "
                f"Make sure you run the training script ('train_ner.py') to completion first."
            )

        print(f"Loading model from: {self.model_path}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path
        )

        # CUDA Device selection (device 0 for GPU, -1 for CPU)
        self.device_idx = 0 if torch.cuda.is_available() else -1

        # Initialize pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device_idx,
        )


    def extract(self, text: str) -> list[dict[str, str]]:
        results = self.ner_pipeline(text)
        output = merge_adjacent_same_type(results)

        formated_output = []
        for item in output:
            item["text"] = item["text"].strip().replace('\\', '').replace('"', '').replace('\'', '')
            item["label"] = item["label"].upper()
            if item["label"] == "PATH":

                target = "דף"
                pattern = rf"\b{target}\b\s+(\p{{L}}+)"
                match = regex.search(pattern, text)

                if match:
                    daf = word_value(match.group(1))

                    target = "עמוד"
                    pattern = rf"\b{target}\b\s+(\p{{L}}+)"
                    match = regex.search(pattern, text)

                    if match:
                        amud = word_value(match.group(1))   
                        item["text"] = daf*2 - 3 + amud
                        formated_output.append({item["label"]: item["text"]})
                    else:
                        item["text"] = daf*2 - 2
                        formated_output.append({item["label"]: item["text"]})
                        item["text"] = daf*2 - 1
                        formated_output.append({item["label"]: item["text"]})

                else:
                    gematria = hebrew_to_values(item["text"])
                    for number in gematria:
                        if number is not -1:
                            item["text"] = number -1
                            formated_output.append({item["label"]: item["text"]})

        return formated_output



def merge_adjacent_same_type(results):
    if not results:
        return []
        
    merged_entities = []
    current_span = {
        "text": results[0]["word"].strip(),
        "label": results[0]["entity_group"].upper(),
        "start": results[0]["start"],
        "end": results[0]["end"],
        "combined": False
    }
    
    for entity in results[1:]:
        word = entity["word"].strip()
        label = entity["entity_group"].upper()
        start = entity["start"]
        end = entity["end"]
        
        same_label = (current_span["label"] == label)
        is_adjacent = (start <= current_span["end"] + 1)
        
        if same_label and is_adjacent:
            gap = start - current_span["end"]
            current_span["text"] += (" " * gap) + word
            current_span["end"] = end
            current_span["combined"] = True
        else:
            merged_entities.append(current_span)
            current_span = {
                "text": word,
                "label": label,
                "start": start,
                "end": end
            }
            
    merged_entities.append(current_span)
    return merged_entities
        

if __name__ == "__main__":
    pipeline = RabbinicCitationExtractor()
    with open("text_01.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                print("\n", line)
                line_data = json.loads(line)
                output = pipeline.extract(line_data)
                print(output)