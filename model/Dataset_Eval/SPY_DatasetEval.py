from datasets import load_dataset
import numpy as np
import pandas as pd
from typing import List
from datasets import Dataset
from model.Dataset_Eval.Model_Access_Utils import (split_into_sentences, sentence_aware_predict_dataset,
                                classification_report_dataset, classification_report_email_level)

dataset = load_dataset(
    "parquet",
    data_files={
        "legal_questions": "model/Dataset_Eval/spy_dataset/spy_legal_questions.parquet",
        "medical_consultations": "model/Dataset_Eval/spy_dataset/spy_medical_consultations.parquet"
    }
)

medical = dataset["medical_consultations"]
legal_questions = dataset["legal_questions"]

#(4491, 4)
#{'tokens': List(Value('string')), 'ent_tags': List(Value('string')),
# 'trailing_whitespaces': List(Value('bool')), 'labels': List(Value('int64'))}




def transform_spy_dataset_fast(hf_dataset: Dataset):

    def reconstruct_text(tokens, trailing_ws):
        return "".join(
            tok + (" " if ws else "") for tok, ws in zip(tokens, trailing_ws)
        ).strip()

    def split_into_sentences_from_tokens(tokens, trailing_ws):
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

    def compute_sentence_sensitivity_vectorized(tokens, trailing_ws, ent_tags, sentences):
        n = len(tokens)

        # --- Build token start/end positions (vectorized) ---
        token_lengths = np.array([len(t) for t in tokens], dtype=np.int32)
        spaces = np.array(trailing_ws, dtype=np.int32)

        # cumulative positions
        offsets = np.cumsum(np.concatenate([[0], token_lengths + spaces]))[:-1]
        token_starts = offsets
        token_ends = offsets + token_lengths

        # --- Sentence spans ---
        full_text = reconstruct_text(tokens, trailing_ws)

        sent_starts = []
        cursor = 0
        for s in sentences:
            start = full_text.find(s, cursor)
            end = start + len(s)
            sent_starts.append(start)
            cursor = end

        sent_starts = np.array(sent_starts, dtype=np.int32)
        sent_ends = sent_starts + np.array([len(s) for s in sentences], dtype=np.int32)

        # --- Map each token → sentence index (vectorized) ---
        # For each token start, find which sentence it belongs to
        token_to_sent = np.searchsorted(sent_ends, token_starts, side="right")

        # --- Entity mask (vectorized BIO collapse) ---
        ent_tags_arr = np.array(ent_tags)
        is_entity_token = ent_tags_arr != "O"

        # --- Aggregate: sentence is sensitive if any token in it is entity ---
        # bincount with weights = entity mask
        sent_entity_counts = np.bincount(
            token_to_sent,
            weights=is_entity_token.astype(np.int32),
            minlength=len(sentences)
        )

        sentence_sensitive = sent_entity_counts > 0

        return sentence_sensitive.tolist()

    def transform_row(row):
        tokens = row.get("tokens", [])
        trailing_ws = row.get("trailing_whitespaces", [])
        ent_tags = row.get("ent_tags", [])
        labels = row.get("labels", [])

        sentences = split_into_sentences_from_tokens(tokens, trailing_ws)

        sentence_sensitive = compute_sentence_sensitivity_vectorized(
            tokens, trailing_ws, ent_tags, sentences
        )

        full_text = reconstruct_text(tokens, trailing_ws)

        return {
            "id": row.get("id", ""),
            "text": sentences,
            "sensitive": sentence_sensitive,
            "metadata": {
                "full_text": full_text,
                "tokens": tokens,
                "ent_tags": ent_tags,
                "labels": labels,
            }
        }

    return hf_dataset.map(transform_row)
import numpy as np


print("Mapping Medical Data")
spy_Medical = transform_spy_dataset_fast(medical)
print(spy_Medical.features)
print("Mapping Legal Data")
spy_Legal_Questions = transform_spy_dataset_fast(legal_questions)
print(spy_Legal_Questions.features)
#%%

print("Get sensitive vals ")
print("med", spy_Medical["sensitive"])
med_sensitive:List[bool] = spy_Medical["sensitive"]
email_sensitive_count = 0
email_not_sensitive = len(med_sensitive)
for message in med_sensitive:
    for sent in message:
        if sent:
            email_sensitive_count += 1
            email_not_sensitive -= 1
            break
print(email_not_sensitive, " emails not sensitive")
print(email_sensitive_count, " emails sensitive")
#%%

print("Predicting Medical Dataset")
y_pred_med = sentence_aware_predict_dataset(spy_Medical)

classification_report = classification_report_dataset(spy_Medical, y_pred_med)
email_level = classification_report_email_level(spy_Medical, y_pred_med)
print(classification_report)
print(email_level)
print("Predicting Legal Dataset")
y_pred_legal = sentence_aware_predict_dataset(spy_Legal_Questions)
classification_report_legal = classification_report_dataset(spy_Legal_Questions, y_pred_legal)
email_level_legal = classification_report_email_level(spy_Legal_Questions, y_pred_legal)
print(classification_report_legal)
print(email_level)
#%%
#Distribution of sensitive vs non
print("sensitive distribution of medical dataset")
# med_sensitive = spy_Medical["sensitive"]
print(spy_Medical["sensitive"].value_counts(normalize=True))

print("sensitive distribution of Legal dataset")
# leg_sensitive = spy_Legal_Questions["sensitive"]
print(spy_Legal_Questions["sensitive"].value_counts(normalize=True))
