import argparse
import os
import sqlite3
import pandas as pd
from fastai.text.all import *


def train_and_save_model(db_path, table_name, model_output_name="bio_classifier.pkl"):
    # 1. Validation checks
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    print(f"Connecting to database: {db_path}...")
    conn = sqlite3.connect(db_path)

    # 2. Extract data from SQLite
    # Explicitly casting the label as a string to ensure fastai treats it as a discrete category
    query = f"""
        SELECT text, CAST(label AS TEXT) as label 
        FROM {table_name} 
        WHERE text IS NOT NULL AND text != '' AND label IN ('0', '1', '2')
    """

    print(f"Executing query on table '{table_name}'...")
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        conn.close()
        raise ValueError(f"Failed to read from table '{table_name}'. Error: {e}")
    finally:
        conn.close()

    if len(df) == 0:
        print(
            "No valid data found matching the criteria (non-empty text and labels 0, 1, or 2). Exiting."
        )
        return

    print(f"Successfully loaded {len(df)} rows.")
    print("Class distribution:\n", df["label"].value_counts())

    # 3. Create fastai DataLoaders
    print("\nPreparing DataLoaders (tokenizing text)...")
    dls = TextDataLoaders.from_df(
        df,
        text_col="text",
        label_col="label",
        valid_pct=0.2,
        bs=64,  # Batch size; lower this to 32 or 16 if your GPU runs out of VRAM
    )

    # 4. Initialize and Train the Model
    print("\nInitializing text classifier learner (AWD_LSTM)...")
    # drop_mult=0.5 adds regularization to prevent overfitting on smaller/medium datasets
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

    print("Starting training (Fine-tuning)...")
    # 1 freeze epoch followed by 3 unfrozen epochs is a solid baseline for fastai text
    learn.fine_tune(4, base_lr=2e-2)

    # 5. Export the Trained Model
    # .export() saves everything needed for inference (architecture, vocab, weights) into a single file
    print(f"\nTraining complete. Saving model to '{model_output_name}'...")
    learn.export(model_output_name)
    print("Model saved successfully!")


if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a fastai text classifier from an SQLite table."
    )
    parser.add_argument(
        "--db", type=str, required=True, help="Path to the SQLite database file."
    )
    parser.add_argument(
        "--table",
        type=str,
        required=True,
        help="Name of the table containing the data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bio_classifier.pkl",
        help="Filename for the exported model.",
    )

    args = parser.parse_args()

    train_and_save_model(
        db_path=args.db, table_name=args.table, model_output_name=args.output
    )
