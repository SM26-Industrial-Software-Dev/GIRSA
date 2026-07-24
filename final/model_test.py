import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Path to your saved fine-tuned model folder (using absolute path to prevent Hub errors)
model_path = os.path.abspath(".")

print("Loading your fine-tuned ByT5 model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

# Keep your Arc GPU device target
device = "xpu" if torch.xpu.is_available() else "cpu"
model = model.to(device)
model.eval()


def transliterate(text_list, batch_size=4):
    """
    Accepts a list of strings and processes them in batches.
    """
    if isinstance(text_list, str):
        text_list = [text_list]

    all_results = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]

        # Process text input into raw bytes (added padding=True for batching)
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=64, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate the Hebrew character tokens using beam search
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=64,
                num_beams=4,  # Evaluates multiple paths for accurate letter combos
                early_stopping=True,
            )

        # Decode raw bytes back into readable Hebrew text for the entire batch
        hebrew_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_results.extend(hebrew_texts)

    return all_results


if __name__ == "__main__":
    # Example usage:
    sentences = [
        "But this <i>Aliya</i> may have unusual importance, as let's see, it may arguably be the most important <i>Aliya</i> in the <i>Torah</i>.",
        "So you may not have been expecting that.",
        "So we certainly want to address this <i>Aliya</i>, and then we have another <i>Aliya</i> to discuss, which in its own way may be the most important <i>Aliya</i> in the <i>Torah</i> because it may be the only time there's a <i>Mitzvah min-Hatorah</i> to hear that <i>Aliya</i>.",
        "So I'd like to make time now to speak both about <i>Tetzaveh</i> and <i>Zachor</i>.",
        "<i>Zachor</i> is actually a pretty massive topic because there are a lot of dimensions to consider.",
        "So we'll use the clock to guide how much we can say about it and presumably even after <i>Shabbos</i>, given that there's a lot to discuss and it's <i>Purim</i> almost a whole week later, so we'll probably also find other opportunities to add and supplement, but we'll see how much we can squeeze in today, especially that which can help us fulfill the <i>Mitzvah</i> of <i>Shabbos</i> of <i>Zakhor</i>.",
        "So first <i>Tetzaveh</i>.",
        "So the <i>Aliya</i> starts off talking about the <i>Korban Tamid</i>.",
        "וזה אשר תעשה על המזבח כבשים בני שנה שנים ליום תמיד.",
        "So this is what you do for the <i>Korban Tamid</i>.",
        "Two lambs that are a year old every day.",
        "So here, first of all, <i>Rav</i> Soloveitchik notes that the same idea appears in <i>Parshas Pinchas</i> and that's itself worthy of focus, but there the language is also <i>asuyah be-Har Sinai</i>.",
        "So the reference to <i>Har Sinai</i> here, what's that talking about?"
        "So according to one <i>deah</i> in <i>Rashi</i>, the <i>Korban</i> that was brought at that time was a <i>Tamid</i>.",
        "So that's good.",
        "But in <i>Chagigah</i> on <i>daf vov</i>, there's actually two <i>deos</i> about this, whether it was an <i>Olas Re'iyah</i> or a <i>Tamid</i>.",
        "So then we have to figure out what the relationship here is.",
        "The <i>Gemara Masechet Keritut</i> on <i>daf tes</i> tells us that <i>Har Sinai</i> was the model for <i>geirus</i>.",
    ]
    results = transliterate(sentences, batch_size=2)
    for eng, heb in zip(sentences, results):
        print(f"{eng} -> {heb}")
