import argparse
import os
import pandas as pd

from fastai.text.all import *


def train_model(
    data_path,
    output_path="citation_classifier.pkl"
):
    # Reproducibility
    set_seed(42, reproducible=True)

    # Check dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find dataset: {data_path}"
        )

    print("Loading dataset...")

    df = pd.read_csv(data_path)

    # Validate columns
    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required_columns}. "
            f"Found: {set(df.columns)}"
        )

    # Clean data
    df = df.dropna(subset=["text", "label"])

    # Ensure labels are strings for fastai classification
    df["label"] = df["label"].astype(str)

    # Validate labels
    valid_labels = {"0", "1"}

    if not set(df["label"].unique()).issubset(valid_labels):
        raise ValueError(
            f"Labels must only be 0 or 1. "
            f"Found: {df['label'].unique()}"
        )

    print(f"Loaded {len(df)} examples")
    print("\nLabel distribution:")
    print(df["label"].value_counts())


    print("\nCreating DataLoaders...")

    dls = TextDataLoaders.from_df(
        df,
        text_col="text",
        label_col="label",
        valid_pct=0.2,
        bs=64
    )


    print("\nCreating classifier...")

    learn = text_classifier_learner(
        dls,
        AWD_LSTM,
        drop_mult=0.5,
        metrics=[
            accuracy,
            F1Score(),
            Precision(),
            Recall()
        ]
    )


    print("\nTraining...")

    learn.fine_tune(
        4,
        base_lr=2e-2
    )


    print("\nSaving model...")

    learn.export(output_path)

    print(
        f"Model saved to {output_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train citation detection classifier"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV training data"
    )

    parser.add_argument(
        "--output",
        default="citation_classifier.pkl",
        help="Output model file"
    )

    args = parser.parse_args()


    train_model(
        args.data,
        args.output
    )