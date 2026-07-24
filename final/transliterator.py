import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


def batch_process_sentences(model_dir, sentences, batch_size=4):
    print("Loading model and tokenizer...")

    # 1. Load the tokenizer from the local directory only
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # 2. Load the architecture explicitly from the local directory
    model = T5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)

    # Move to GPU if available
    device = "xpu" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    results = []

    # 3. Process sentences in batches
    print(f"Processing {len(sentences)} sentences...")
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        # Tokenize the batch
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,  # Adjust based on your expected output length
                num_beams=4,  # Standard for seq2seq tasks
                early_stopping=True,
            )

        # Decode the byte/character tokens back into text
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded_outputs)

    return results


if __name__ == "__main__":
    # Resolve to an absolute path to prevent Hub validation from guessing it's a URL
    MODEL_DIRECTORY = os.path.abspath(".")

    # Input data
    input_sentences = [
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
    # Run the batch processing
    try:
        predictions = batch_process_sentences(
            MODEL_DIRECTORY, input_sentences, batch_size=2
        )

        # Print results
        for original, pred in zip(input_sentences, predictions):
            print(f"\n[Input]:  {original}")
            print(f"[Output]: {pred}")

    except Exception as e:
        print(f"\nExecution failed: {e}")
