from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from datasets import Value

path = "../Model_s"
from pathlib import Path

path = str(Path(__file__).resolve().parent.parent / "Model_s")
overall_threshold = 0.90
def load_sentence_Model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        print("Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    return tokenizer, model
def split_into_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s|\n+', text)
    return [s.strip() for s in sentences if s.strip()]

def predict(text, tokenizer, torch, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score = probs[0][1].item()
        prediction = 1 if score >= overall_threshold else 0
        return prediction, score

from datasets import Value, Sequence
def _df_format_check(df):
    if len(df) == 0:
        return False

    features = df.features

    # Case 1: Already correct format
    if (
        "id" in features and
        "text" in features and
        "sensitive" in features
    ):
        return True

    # Case 2: SPY raw format
    if (
        "tokens" in features and
        "ent_tags" in features and
        "trailing_whitespaces" in features
    ):
        return True

    return False
# def _df_format_check(df):
#     # Hugging Face Dataset is empty if len(df) == 0
#     if len(df) == 0:
#         return False
#
#     features = df.features
#     try:
#         # required columns
#         assert "id" in features
#         assert "source_text" in features
#         assert "text" in features
#         assert "sensitive" in features
#
#         # check types
#         assert features["id"] == Value("string")
#         assert features["source_text"] == Value("string") or features["source_text"] == Value("large_string")
#
#         # sentence-level lists
#         assert isinstance(features["text"], Sequence)
#         assert features["text"].feature == Value("string")
#
#         assert isinstance(features["sensitive"], Sequence)
#         assert features["sensitive"].feature == Value("bool")
#
#         return True
#     except AssertionError:
#         return False
#

import torch
from typing import Dict, List

def predict_dataset(df) -> Dict[str, Dict[str, List]]:
    if not _df_format_check(df):
        raise ValueError(f"Dataset not supported: {df}")

    df = _normalize_dataset(df)
    tokenizer, model = load_sentence_Model()
    model.eval()

    results: Dict[str, Dict[str, List]] = {}

    for row in df:
        row_id = row["id"]
        text = row["source_text"]

        sentences = split_into_sentences(text)

        sentence_preds: List[bool] = []
        sentence_scores: List[float] = []

        for sentence in sentences:
            prediction, score = predict(sentence, tokenizer, torch, model)
            sentence_preds.append(bool(prediction))
            sentence_scores.append(score)

        results[row_id] = {
            "predictions": sentence_preds,
            "scores": sentence_scores
        }

    return results

import torch
from typing import Dict, List

def sentence_aware_predict_dataset(df) -> Dict[str, Dict[str, List]]:
    if not _df_format_check(df):
        raise ValueError(f"Dataset not supported: {df}")

    df = _normalize_dataset(df)
    tokenizer, model = load_sentence_Model()
    model.eval()

    results: Dict[str, Dict[str, List]] = {}

    for row in df:
        row_id = row["id"]
        sentences: List[str] = row["text"]  # already split

        sentence_preds: List[bool] = []
        sentence_scores: List[float] = []

        if not sentences:
            results[row_id] = {
                "predictions": [],
                "scores": [],
                "overall_prediction": False,
                "overall_score": 0.0
            }
            continue

        # --- batch inference (IMPORTANT: much faster) ---
        inputs = tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # class 1 = sensitive
        scores = probs[:, 1].tolist()
        preds = [score >= overall_threshold for score in scores]

        sentence_preds.extend(preds)
        sentence_scores.extend(scores)

        # --- document-level aggregation ---
        overall_score = max(sentence_scores)
        overall_prediction = overall_score >= overall_threshold

        results[row_id] = {
            "predictions": sentence_preds,     # List[bool]
            "scores": sentence_scores,         # List[float]
            "overall_prediction": overall_prediction,
            "overall_score": overall_score
        }

    return results
import numpy as np
from sklearn.metrics import classification_report

def classification_report_dataset(df, pred_dict):
    y_true = []
    y_pred = []

    missing_ids = []
    length_mismatches = []

    for row in df:
        row_id = row["id"]

        if row_id not in pred_dict:
            missing_ids.append(row_id)
            continue

        gt_labels = row["sensitive"]              # List[bool]
        pred_labels = pred_dict[row_id]["predictions"]  # List[bool]

        if len(gt_labels) != len(pred_labels):
            length_mismatches.append((row_id, len(gt_labels), len(pred_labels)))
            continue

        # flatten
        y_true.extend(gt_labels)
        y_pred.extend(pred_labels)

    # --- diagnostics ---
    if missing_ids:
        print(f"[WARN] Missing predictions for {len(missing_ids)} ids")

    if length_mismatches:
        print(f"[WARN] Length mismatches in {len(length_mismatches)} rows")
        print("Example:", length_mismatches[:3])

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    print("\nClassification Report (Sentence-Level):")
    print(classification_report(y_true, y_pred))


def classification_report_email_level(df, pred_dict):
    y_true = []
    y_pred = []

    for row in df:
        row_id = row["id"]

        if row_id not in pred_dict:
            continue

        gt = any(row["sensitive"])
        pred = pred_dict[row_id]["overall_prediction"]

        y_true.append(int(gt))
        y_pred.append(int(pred))

    print("\nClassification Report (Email-Level):")
    print(classification_report(y_true, y_pred))

def _normalize_dataset(df):
    features = df.features

    # --- Case 1: already normalized ---
    if "text" in features and "sensitive" in features:
        return df

    # --- Case 2: SPY format ---
    if "tokens" in features:

        def reconstruct_text(tokens, trailing_ws):
            return "".join(
                tok + (" " if ws else "") for tok, ws in zip(tokens, trailing_ws)
            ).strip()

        def split_sentences(tokens, trailing_ws):
            sentences = []
            current = ""

            for tok, ws in zip(tokens, trailing_ws):
                current += tok
                if ws:
                    current += " "
                if tok in {".", "!", "?"}:
                    sentences.append(current.strip())
                    current = ""

            if current.strip():
                sentences.append(current.strip())

            return sentences

        def compute_sensitive(tokens, trailing_ws, ent_tags, sentences):
            # vectorized version
            token_lengths = np.array([len(t) for t in tokens])
            spaces = np.array(trailing_ws, dtype=int)

            offsets = np.cumsum(np.concatenate([[0], token_lengths + spaces]))[:-1]
            token_starts = offsets

            full_text = reconstruct_text(tokens, trailing_ws)

            sent_starts = []
            cursor = 0
            for s in sentences:
                start = full_text.find(s, cursor)
                sent_starts.append(start)
                cursor = start + len(s)

            sent_ends = np.array(sent_starts) + np.array([len(s) for s in sentences])

            token_to_sent = np.searchsorted(sent_ends, token_starts, side="right")

            is_entity = np.array(ent_tags) != "O"

            sent_counts = np.bincount(
                token_to_sent,
                weights=is_entity.astype(int),
                minlength=len(sentences)
            )

            return (sent_counts > 0).tolist()

        def transform_row(row):
            tokens = row["tokens"]
            ws = row["trailing_whitespaces"]
            ent_tags = row["ent_tags"]

            sentences = split_sentences(tokens, ws)
            sensitive = compute_sensitive(tokens, ws, ent_tags, sentences)

            return {
                "id": row.get("id", ""),
                "source_text": reconstruct_text(tokens, ws),
                "text": sentences,
                "sensitive": sensitive
            }

        return df.map(transform_row)

    raise ValueError("Unsupported dataset format")