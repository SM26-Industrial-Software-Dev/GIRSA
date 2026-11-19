# Shiur Citation Finder

Automatically find and link the Torah sources quoted or referenced in a shiur (Torah lecture) transcript.

## Demo Video

Watch our team walk through how the project was built, how it works, and a live demo:

**[Watch the demo on YouTube](https://youtu.be/l5mgeqNjs1s?si=ezavYpNoViwU52cv)**

## Project Goal

Shiurim are full of references to Tanach, Talmud, halachic codes, commentaries, and later works. A speaker may quote a source word for word in Hebrew, paraphrase it in English, use a transliterated Hebrew phrase ("the *Rambam* in *Hilchos Tzitzis*"), or name a source several sentences before actually quoting it. Tracking all of these down by hand takes a long time.

This project takes a plain-text transcript of a shiur and returns the same transcript annotated with footnotes. Each footnote links directly to the source on [Sefaria](https://www.sefaria.org), so a listener or reader can jump straight to the text being discussed.

## How It Works

The system has two parts: a local copy of the Sefaria library with a search engine built on top of it, and a pipeline that reads a transcript and resolves its citations against that library.

### 1. Sefaria library and search engine

- The full public Sefaria export is downloaded and loaded into a SQLite database of books, authors, aliases, passages, and text chunks.
- Every chunk is embedded with the multilingual [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) sentence-transformer model, and the embeddings are stored in a FAISS index. This allows semantic search: an English paraphrase can match the original Hebrew or Aramaic.
- A SQLite FTS5 trigram index is built over the Hebrew text for fast fuzzy matching of direct Hebrew quotes.
- A FastAPI search service (`final/api_v3.py`) exposes the semantic search. It searches the most popular books first and widens the pool in tiers, then reranks the top candidates with a cross-encoder.

### 2. Transcript pipeline

When a transcript is submitted, `final/main.py` runs these steps:

1. **Transliteration.** A fine-tuned seq2seq model converts transliterated Hebrew in the transcript (for example *Kriyat Shema*) back into Hebrew script.
2. **Citation classification.** A FastAI text classifier flags the sentences that quote, paraphrase, or mention a source. The rest of the transcript is skipped.
3. **Query extraction.** Flagged sentences are sent in batches to Google Gemini, which extracts clean search queries for the quoted or paraphrased content.
4. **Candidate search.**
   - Each query is sent to the semantic search API.
   - Runs of five or more Hebrew words are fuzzy-matched against the FTS5 index to catch direct quotes.
5. **Named entity recognition.** A fine-tuned token-classification model pulls out source titles and locations (for example "*Shulchan Aruch*, *Siman Tes*, *Seif Heh*").
6. **Scoring.** Candidates are scored against NER results from a rolling window of nearby sentences, so a source named a few sentences earlier still counts. Book names are resolved through the database's alias and author tables, and Hebrew numerals are converted with gematria. Candidates that match the referenced book or location are boosted, and only results above a confidence threshold are kept.
7. **Output.** The transcript is returned as HTML, with superscript footnote markers and a numbered list of Sefaria links.

### Web front end

`final/index.html` is a simple page where you upload a `.txt` transcript and see the annotated result. It calls a small FastAPI wrapper (`final/frontend.py`) that runs the pipeline.

## Repository Layout

| Path                                                     | Contents                                                                          |
| -------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `final/`                                                 | The finished application: pipeline, search API, front end, and model loaders      |
| `build/`                                                 | Scripts that download Sefaria and build the database, embeddings, and FAISS index |
| `find_hebrew_passage/`                                   | Builds and tests the FTS5 trigram index used for fuzzy Hebrew matching            |
| `citation_classifier/`                                   | Training data and training script for the citation classifier                     |
| `Ai-NER/`                                                | Training scripts for the NER model                                                |
| `Ai-translitiration/`                                    | Data extraction and training scripts for the transliteration model                |
| `sefaria-scripts/`, `end-to-end/`, `transcript-scripts/` | Earlier experiments and prototypes kept for reference                             |

## Setup

### Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/) (or pip)
- A Google Gemini API key
- Plenty of disk space: the Sefaria database, embeddings, and index together take well over 14 GB
- A GPU is strongly recommended for generating embeddings and for running the transliteration model

### 1. Install dependencies

```bash
cd final
uv sync
```

All dependencies are listed in `final/pyproject.toml`. If you use pip instead of uv, create a virtual environment and install the packages listed there.

### 2. Build the Sefaria database and indexes

Run these from the `build/` directory. This step takes a long time and only has to be done once.

```bash
cd build

# Download the Sefaria export (texts and schemas)
python download-sefaria.py --out ./sefaria_export --workers 8

# Load it into SQLite
python build_db.py --export-dir ./sefaria_export --db sefaria.db

# Embed every chunk with LaBSE (reads and writes sefaria.db in the current directory)
python generate_embeddings.py

# Build the FAISS index (writes sefaria.index and sefaria.index.idmap.npy)
python build_faiss_index.py --db sefaria.db --out sefaria.index
```

Then build the FTS5 trigram tables used for fuzzy matching. The script opens `sefaria.db` in the current directory:

```bash
python ../find_hebrew_passage/FTS5_trigram_create_v5.py
```

`download-sefaria.py` skips any versions listed in `removable_versions.csv` by default. Use `--no-exclude` to download everything, or `--exclude-csv` to point at a different file.

A prebuilt Docker image of the search service is also available: `esheinfe/sefaria-search:v1`.

### 3. Train the models

The trained model files are not included in this repository, so you will need to build them yourself. The pipeline expects three models inside `final/`:

| Path                            | Model                                                             | Training script                                |
| ------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------- |
| `final/citation_classifier.pkl` | Citation sentence classifier (FastAI)                             | `citation_classifier/train_classifier.py`      |
| `final/transliterate/`          | Transliteration to Hebrew (Hugging Face seq2seq)                  | `Ai-translitiration/ai-model/model_trainer.py` |
| `final/NER/`                    | Source title and location NER (Hugging Face token classification) | `Ai-NER/train_ner.py`                          |

Train each one with the listed script, then place the output at the path shown.

### 4. Configure paths and keys

The code ships with placeholder values that you need to replace before running:

| File | Setting | Placeholder | Replace with |
| --- | --- | --- | --- |
| `final/main.py` | `GEMINI_API_KEY` | `YOUR_GEMINI_API_KEY` | Your Google Gemini API key |
| `final/main.py` | `DB_PATH` | `/path/to/sefaria.db` | Path to your `sefaria.db` |
| `final/FuzzyMatch.py` | `DB_FILE` | `/path/to/sefaria.db` | Path to your `sefaria.db` |
| `final/book_alias_lookup.py` | `PATH_TO_SEFARIA_DB` | `/path/to/sefaria.db` | Path to your `sefaria.db` |

The search API reads its settings from environment variables:

| Variable     | Default                       | Purpose                      |
| ------------ | ----------------------------- | ---------------------------- |
| `DB_PATH`    | `sefaria.db`                  | Path to the Sefaria database |
| `INDEX_PATH` | `sefaria.index`               | Path to the FAISS index      |
| `MODEL_NAME` | `sentence-transformers/LaBSE` | Embedding model              |

## Usage

Run each service in its own terminal from the `final/` directory.

**1. Start the semantic search API (port 8000):**

```bash
DB_PATH=/path/to/sefaria.db INDEX_PATH=/path/to/sefaria.index \
  uv run uvicorn api_v3:app --host 127.0.0.1 --port 8000
```

Check that it is up with `curl http://127.0.0.1:8000/health`. You can also query it directly from the command line:

```bash
uv run python api_v3.py --db /path/to/sefaria.db --index /path/to/sefaria.index \
  --query "In the beginning God created"
```

**2. Start the pipeline backend (port 8081):**

```bash
uv run uvicorn frontend:app --host 127.0.0.1 --port 8081
```

**3. Open the web page:**

```bash
python -m http.server 5500
```

Then go to `http://127.0.0.1:5500/index.html`, choose a `.txt` transcript, and click Analyze. The annotated transcript appears with footnotes linking to each source on Sefaria.

To run the pipeline without the web page, edit the sample transcript in `main()` in `final/main.py` and run `uv run python main.py`. The annotated HTML is printed to the terminal.

## Acknowledgments

All source texts come from [Sefaria](https://www.sefaria.org) and its public data export.
Thank you to SoferAI for access to transcripts to use for training purposes.
