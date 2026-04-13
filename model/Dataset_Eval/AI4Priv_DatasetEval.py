from datasets import load_dataset
from Model_Access_Utils import (split_into_sentences, sentence_aware_predict_dataset,
                                classification_report_dataset, classification_report_email_level)
import json

datasets = load_dataset("ai4privacy/pii-masking-300k")
print(datasets)
dataset = datasets["validation"]

# 2. Filter rows where language is 'english'
english_dataset = dataset.filter(lambda x: x["language"] == "English")
import json
from typing import List, Dict

def transform_hf_dataset(hf_dataset):

    def transform_row(row):

        def parse_json_field(val):
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return val
            return val

        text = row.get("source_text", "")
        sentences = split_into_sentences(text)

        privacy_mask = parse_json_field(row.get("privacy_mask")) or []
        span_labels = parse_json_field(row.get("span_labels")) or []
        mbert_bio_labels = parse_json_field(row.get("mbert_bio_labels")) or []

        # --- build sentence spans (char offsets) ---
        sentence_spans = []
        cursor = 0
        for sentence in sentences:
            start = text.find(sentence, cursor)
            end = start + len(sentence)
            sentence_spans.append((start, end))
            cursor = end

        # --- helper: check overlap ---
        def overlaps(span1, span2):
            return span1[0] < span2[1] and span2[0] < span1[1]

        # --- compute sentence-level sensitivity ---
        sentence_sensitive: List[bool] = []

        for sent_span in sentence_spans:
            is_sensitive = False
            for mask in privacy_mask:
                mask_span = (mask.get("start"), mask.get("end"))
                if overlaps(sent_span, mask_span):
                    is_sensitive = True
                    break
            sentence_sensitive.append(is_sensitive)

        return {
            "id": row.get("id", ""),
            "text": sentences,  # now list[str] instead of single string
            "sensitive": sentence_sensitive,  # list[bool]
            "metadata": {
                "target_text": row.get("target_text", ""),
                "privacy_mask": privacy_mask,
                "span_labels": span_labels,
                "mbert_bio_labels": mbert_bio_labels,
                "language": row.get("language"),
                "set": row.get("set")
            }
        }

    return hf_dataset.map(transform_row)

# dataset = dataset[:1000]
AI_4Priv_English_Validation_data = transform_hf_dataset(dataset)
print(AI_4Priv_English_Validation_data.features)

y_pred = sentence_aware_predict_dataset(AI_4Priv_English_Validation_data)
print(y_pred)
classification_report = classification_report_dataset(AI_4Priv_English_Validation_data, y_pred)
print(classification_report)
email_level = classification_report_email_level(AI_4Priv_English_Validation_data, y_pred)
print(email_level)

exit(0)