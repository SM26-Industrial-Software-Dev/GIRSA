import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class HebrewTransliterationEngine:
    def __init__(self, model_path="final_hebrew_transliteration_model", batch_size=512):
        """
        Phase 1: Initialization. 
        Loads the model and tokenizer into memory once.
        """
        self.batch_size = batch_size
        print("Loading model directly into 16-bit Tensor Cores...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        torch.backends.cudnn.benchmark = True
        
        # Matches EITHER <i>(.*?)</i> OR *(.*?)?* while handling multi-line patterns safely
        self.tag_regex = re.compile(r"<i>(.*?)</i>|(?<!\*)\*([^*]+?)\*(?!\*)", re.DOTALL)
        
        # Sub-patterns used to completely delete tags AND their inner text if bypass is active
        self.html_strip_regex = re.compile(r"<i>.*?</i>", re.DOTALL)
        self.md_strip_regex = re.compile(r"(?<!\*)\*[^*]+?\*(?!\*)", re.DOTALL)

        self.double_space_regex = re.compile(r" {2,}")

    def process_transcript(self, input_text: str, isEnHe: bool = False) -> str:
        if not input_text:
            return ""

        if isEnHe:
            cleaned_text = self.html_strip_regex.sub("", input_text)
            cleaned_text = self.md_strip_regex.sub("", cleaned_text)
            cleaned_text = self.double_space_regex.sub(" ", cleaned_text)
            return cleaned_text

        # 1. Extract and filter matching segments for AI inference
        raw_matches = self.tag_regex.findall(input_text)
        segments_to_process = []
        
        for html_match, md_match in raw_matches:
            match = html_match if html_match else md_match
            is_markdown = bool(md_match)
            
            cleaned_match = match.strip()
            word_count = len(cleaned_match.split())
            
            # Filter Condition: Keep segments with fewer than 25 words
            if 0 < word_count < 25:
                segments_to_process.append({
                    "raw_match": match, 
                    "clean_text": cleaned_match,
                    "is_markdown": is_markdown
                })

        if not segments_to_process:
            return input_text

        # 2. Batch process targets through the GPU inference engine
        decoded_results = []
        for i in range(0, len(segments_to_process), self.batch_size):
            batch = segments_to_process[i:i + self.batch_size]
            texts = [item["clean_text"] for item in batch]
            
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            
            with torch.inference_mode(), torch.cuda.amp.autocast():
                outputs = self.model.generate(**inputs, max_length=64, num_beams=3, early_stopping=True)
                
            batch_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_results.extend(batch_decoded)

        # 3. Perform in-place reconstruction substitution
        output_text = input_text
        for segment, ai_output in zip(segments_to_process, decoded_results):
            if segment["is_markdown"]:
                target_string = f"*{segment['raw_match']}*"
            else:
                target_string = f"<i>{segment['raw_match']}</i>"
                
            output_text = output_text.replace(target_string, ai_output)

        return output_text