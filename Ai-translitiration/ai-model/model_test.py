import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Path to your saved fine-tuned model folder
model_path = "./final_hebrew_transliteration_model"

print("Loading your fine-tuned ByT5 model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Move the model over to your RTX GPU for rapid generation
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def transliterate(english_text):
    # Process text input into raw bytes
    inputs = tokenizer(english_text, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate the Hebrew character tokens using beam search
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=64, 
            num_beams=4,          # Evaluates multiple paths for accurate letter combos
            early_stopping=True
        )
    
    # Decode raw bytes back into readable Hebrew text
    hebrew_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return hebrew_text

# 2. Interactive Terminal Testing Loop
print("\n" + "="*50)
print("ByT5 TRANSLITERATION TESTER READY")
print("Type any English transliteration and press Enter.")
print("Type 'exit' at any time to quit.")
print("="*50 + "\n")

# Prompt user for manual inputs
while True:
    user_input = input("Enter English phrase: ")
    if user_input.lower() == 'exit':
        print("Closing tester. Great job training your model!")
        break
    if user_input.strip() == '':
        continue
        
    result = transliterate(user_input)
    print(f"Hebrew Output:  {result}\n")