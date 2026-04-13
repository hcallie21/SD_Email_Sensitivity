from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from datasets import Value

path = "../Model_s"
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
    # Hugging Face Dataset is empty if len(df) == 0
    if len(df) == 0:
        return False

    features = df.features
    try:
        # required columns
        assert "id" in features
        assert "source_text" in features
        assert "text" in features
        assert "sensitive" in features

        # check types
        assert features["id"] == Value("string")
        assert features["source_text"] == Value("string") or features["source_text"] == Value("large_string")

        # sentence-level lists
        assert isinstance(features["text"], Sequence)
        assert features["text"].feature == Value("string")

        assert isinstance(features["sensitive"], Sequence)
        assert features["sensitive"].feature == Value("bool")

        return True
    except AssertionError:
        return False

# def predict_dataset(df) -> Dict[str, List[bool, float]]:
#     correct_format:bool = _df_format_check(df)
#     if not correct_format:
#         print(f"Dataset not formatted correctly: {df}")
#
#     tokenizer, model = load_sentence_Model()
#     #loop through each row in json dataset
#     results:Dict[str, List[bool, float]] = {}
#     for row in df.itertuples():
#         sentences=split_into_sentences(row.source_text)
#         for each_sentence in sentences:
#             prediction, score = predict(each_sentence, tokenizer, model)
#             results[each_sentence] = [prediction, score]

import torch
from typing import Dict, List

def predict_dataset(df) -> Dict[str, Dict[str, List]]:
    correct_format: bool = _df_format_check(df)
    if not correct_format:
        raise ValueError(f"Dataset not formatted correctly: {df}")

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
    correct_format: bool = _df_format_check(df)
    if not correct_format:
        raise ValueError(f"Dataset not formatted correctly: {df}")

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