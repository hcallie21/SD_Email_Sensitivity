"""Enron dataset evaluation script.

This module mirrors the structure of ``AI4Priv_DatasetEval.py`` and
``SPY_DatasetEval.py`` but operates on the Enron email spam/ham CSV file
located at ``model/Dataset_Eval/enron_spam_data/enron_spam_data.csv``.

The script performs the following steps:
1. Load the CSV using ``datasets.load_dataset``.
2. Transform each row into the unified format expected by the utility
   functions in ``Model_Access_Utils`` – a list of sentences, a list of
   boolean sensitivity flags, and a small ``metadata`` dictionary.
   For the Enron dataset we do not have fine‑grained privacy masks, so the
   ``sensitive`` flag is derived from the ``Spam/Ham`` column: all sentences
   are marked ``True`` for spam messages and ``False`` otherwise.  This is a
   simple proxy that enables the downstream evaluation pipeline to run.
3. Run the sentence‑aware model predictions.
4. Produce sentence‑level and email‑level classification reports.
5. Provide a small distribution analysis class that can be used to inspect
   the proportion of sensitive (spam) vs non‑sensitive (ham) emails.

The implementation purposefully follows the same coding style as the
existing evaluation scripts to keep the project consistent.
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from datasets import load_dataset, Dataset
from typing import List

# Import utility functions that handle sentence splitting, model loading and
# evaluation metrics.
# Import utilities from the same package directory (mirroring AI4Priv_DatasetEval).
from Model_Access_Utils import (
    split_into_sentences,
    sentence_aware_predict_dataset,
    classification_report_dataset,
    classification_report_email_level,
)

import pandas as pd


# ---------------------------------------------------------------------------
# Helper: transform the raw Enron CSV rows into the unified HF dataset format.
# ---------------------------------------------------------------------------
def transform_enron_dataset(hf_dataset: Dataset) -> Dataset:
    """Map Enron rows to the standard ``{id, text, sensitive, metadata}`` schema.

    The original CSV contains the columns ``Message ID``, ``Subject``, ``Message``,
    ``Spam/Ham`` and ``Date``.  ``Message`` holds the raw email body.  We split the
    body into sentences using :func:`split_into_sentences`.  Sensitivity is a
    coarse proxy: if the row is labelled ``spam`` we mark every sentence as
    sensitive, otherwise we mark them as non‑sensitive.
    """

    def transform_row(row):
        """Safely transform a raw CSV row.

        Some rows in the Enron CSV have a missing ``Message`` field which
        results in ``None``.  ``split_into_sentences`` expects a string, so we
        coerce ``None`` to an empty string before processing.
        """
        # ``row`` is a dict‑like object provided by ``datasets``.
        raw_text = row.get("Message", "")
        # Ensure we always pass a string to the sentence splitter.
        text = raw_text if isinstance(raw_text, str) else ""
        sentences: List[str] = split_into_sentences(text)

        # Derive a simple sensitivity flag from the Spam/Ham column.
        is_spam = str(row.get("Spam/Ham", "")).strip().lower() == "spam"
        sentence_sensitive = [is_spam] * len(sentences)

        return {
            "id": str(row.get("Message ID", "")),
            "text": sentences,
            "sensitive": sentence_sensitive,
            "metadata": {
                "subject": row.get("Subject", ""),
                "date": row.get("Date", ""),
                "spam_ham": row.get("Spam/Ham", ""),
            },
        }

    return hf_dataset.map(transform_row)


# ---------------------------------------------------------------------------
# Distribution analysis helper class.
# ---------------------------------------------------------------------------
class DistAnalysis:
    """Utility class for simple distribution analysis of the Enron dataset.

    The class expects a transformed dataset (the output of
    :func:`transform_enron_dataset`).  It provides a method to print the
    proportion of sensitive vs non‑sensitive emails and a method to retrieve the
    raw counts for further processing.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def compute_distribution(self):
        """Print the normalized distribution of the ``sensitive`` column.

        The ``sensitive`` column is a list of booleans per email.  An email is
        considered *sensitive* at the email level if **any** of its sentences
        are marked ``True``.  This mirrors the logic used in the other eval
        scripts.
        """
        # Convert list of bools to a single bool per email.
        email_sensitive = [any(s) for s in self.dataset["sensitive"]]
        total = len(email_sensitive)
        sensitive_count = sum(email_sensitive)
        non_sensitive = total - sensitive_count
        print("Enron dataset sensitive distribution (email level):")
        print(f"  Sensitive: {sensitive_count}/{total} ({sensitive_count/total:.2%})")
        print(f"  Non-sensitive: {non_sensitive}/{total} ({non_sensitive/total:.2%})")
        return {"sensitive": sensitive_count, "non_sensitive": non_sensitive}


# ---------------------------------------------------------------------------
# Main execution flow – analogous to the other evaluation scripts.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the raw CSV as a HuggingFace Dataset.
    raw = load_dataset(
        "csv",
        data_files={"enron": "model/Dataset_Eval/enron_spam_data/enron_spam_data.csv"},
        split="enron",
    )
    print("Raw Enron dataset loaded:", raw)

    # Transform to the unified schema.
    enron_dataset = transform_enron_dataset(raw)
    print("Transformed features:", enron_dataset.features)

    # Run predictions using the sentence‑aware model.
    y_pred = sentence_aware_predict_dataset(enron_dataset)

    # Produce classification reports.
    classification_report = classification_report_dataset(enron_dataset, y_pred)
    print(classification_report)
    email_level = classification_report_email_level(enron_dataset, y_pred)
    print(email_level)

    # Distribution analysis.
    dist = DistAnalysis(enron_dataset)
    dist.compute_distribution()
