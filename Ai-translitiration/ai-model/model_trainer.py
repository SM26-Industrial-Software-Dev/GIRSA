import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# 1. Load and parse your JSONL dataset
raw_data = []
with open("en_to_he.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            raw_data.append(json.loads(line))

# 2. Split into train and validation sets (90/10 split)
train_data, val_data = train_test_split(raw_data, test_size=0.1, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# 3. Load ByT5 Tokenizer and Model
model_id = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)

from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_id)
config.tie_word_embeddings = False

model = AutoModelForSeq2SeqLM.from_pretrained(model_id, config=config)

# 4. Preprocessing function
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["target"]
    
    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=64, truncation=True)
    labels = tokenizer(text_target=targets, max_length=64, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization across datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["input", "target"])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["input", "target"])

# Set up the data collator to handle byte padding dynamically
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


training_args = Seq2SeqTrainingArguments(
    output_dir="byt5_to_hebrew",
    eval_strategy="epoch",       
    learning_rate=1e-4,                  
    lr_scheduler_type="cosine",          
    warmup_ratio=0.10,                   
         
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    gradient_checkpointing=True,
    optim="adamw_torch",
    
    weight_decay=0.01,                   
    num_train_epochs=5,                  
    predict_with_generate=False,              
         
    fp16=False,                               
    bf16=True,  
    max_grad_norm=1.0,                             
    logging_steps=20,                    
    report_to="none" 
)

# 6. Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,  
)

# 7. Start Training
trainer.train()

# 8. Save your fine-tuned model
trainer.save_model("./final_hebrew_transliteration_model")
tokenizer.save_pretrained("./final_hebrew_transliteration_model")
print("Training complete! Model saved successfully.")
