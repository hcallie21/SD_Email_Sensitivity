# nermodel.py
# Train TinyBERT for token-level NER on ner_dataset.conll
# Produces classification report
# Also includes a highlight function for inference

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

from seqeval.metrics import classification_report, precision_score, recall_score, f1_score


# ============================================================
# 1) LOAD CoNLL NER DATASET
# ============================================================

NER_PATH = "../ner_dataset.conll"  # same directory as script


def read_conll(path):
    sentences = []
    tags = []

    cur_tokens = []
    cur_tags = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip metadata lines (# id = ..., # text = ...)
            if line.startswith("#") or line == "":
                if cur_tokens:
                    sentences.append(cur_tokens)
                    tags.append(cur_tags)
                    cur_tokens = []
                    cur_tags = []
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue

            token, tag = parts
            cur_tokens.append(token)
            cur_tags.append(tag)

    # Final sentence
    if cur_tokens:
        sentences.append(cur_tokens)
        tags.append(cur_tags)

    return sentences, tags


print("Loading NER dataset...")
sentences, tags = read_conll(NER_PATH)
print(f"Loaded {len(sentences)} sentences from {NER_PATH}")


# ============================================================
# 2) LABEL MAPPING
# ============================================================

label_set = sorted({t for sent in tags for t in sent})
label2id = {label: i for i, label in enumerate(label_set)}
id2label = {i: label for label, i in label2id.items()}

print(f"Number of labels: {len(label_set)}")
print("Label set:", label_set)


# ============================================================
# 3) TRAIN/VAL SPLIT
# ============================================================

train_sentences, val_sentences, train_tags, val_tags = train_test_split(
    sentences, tags, test_size=0.2, random_state=42
)

print(f"Train sentences: {len(train_sentences)}")
print(f"Val sentences:   {len(val_sentences)}")


# ============================================================
# 4) TOKENIZE + ALIGN LABELS
# ============================================================

MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode_and_align(sentences, tags, max_length=128):
    input_ids, attn_masks, label_ids = [], [], []

    for words, word_tags in zip(sentences, tags):
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        word_ids = enc.word_ids()
        aligned = []

        for w_id in word_ids:
            if w_id is None:
                aligned.append(-100)
            else:
                aligned.append(label2id[word_tags[w_id]])

        input_ids.append(enc["input_ids"])
        attn_masks.append(enc["attention_mask"])
        label_ids.append(aligned)

    return (
        np.array(input_ids),
        np.array(attn_masks),
        np.array(label_ids),
    )


print("Encoding training data...")
X_train_ids, X_train_mask, y_train_ids = encode_and_align(train_sentences, train_tags)

print("Encoding validation data...")
X_val_ids, X_val_mask, y_val_ids = encode_and_align(val_sentences, val_tags)


# ============================================================
# 5) DATASET CLASS
# ============================================================

class NERDataset(Dataset):
    def __init__(self, input_ids, attn_mask, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attn_mask = torch.tensor(attn_mask, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "labels": self.labels[idx],
        }


train_dataset = NERDataset(X_train_ids, X_train_mask, y_train_ids)
val_dataset = NERDataset(X_val_ids, X_val_mask, y_val_ids)


# ============================================================
# 6) MODEL + TRAINER
# ============================================================

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_set),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorForTokenClassification(tokenizer)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true = []
    pred = []

    for p_seq, l_seq in zip(preds, labels):
        cur_true = []
        cur_pred = []
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            cur_true.append(id2label[l])
            cur_pred.append(id2label[p])
        true.append(cur_true)
        pred.append(cur_pred)

    return {
        "precision": precision_score(true, pred),
        "recall": recall_score(true, pred),
        "f1": f1_score(true, pred),
    }


training_args = TrainingArguments(
    output_dir="./ner_results",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_dir="./ner_logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ============================================================
# 7) TRAIN
# ============================================================

print("\n===== Training TinyBERT NER =====")
trainer.train()


# ============================================================
# 8) MODEL.PY-STYLE SUMMARY REPORT
# ============================================================

print("\nTinyBERT NER Metrics:")
metrics = trainer.evaluate()
for k, v in metrics.items():
    print(f"{k}: {v}")

print("\nDetailed NER Classification Report:")
pred_output = trainer.predict(val_dataset)
preds = np.argmax(pred_output.predictions, axis=-1)
labels = pred_output.label_ids

true_labels = []
true_preds = []

for p_seq, l_seq in zip(preds, labels):
    cur_true, cur_pred = [], []
    for p, l in zip(p_seq, l_seq):
        if l == -100:
            continue
        cur_true.append(id2label[l])
        cur_pred.append(id2label[p])
    true_labels.append(cur_true)
    true_preds.append(cur_pred)

print(classification_report(true_labels, true_preds))


# ============================================================
# 9) SAVE MODEL
# ============================================================

SAVE_DIR = "./ner_tinybert_model"
print(f"\nSaving TinyBERT NER model to {SAVE_DIR}")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)


# ============================================================
# 10) SIMPLE HIGHLIGHT INFERENCE
# ============================================================

def highlight(sentence):
    model.eval()
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(**enc).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])[1:-1]
    pred_ids = pred_ids[1:-1]

    labels = [id2label[i] for i in pred_ids]

    # Reconstruct merged words
    words = []
    wlabels = []
    cur_word = ""
    cur_lab = "O"

    for tok, lab in zip(tokens, labels):
        if tok.startswith("##"):
            cur_word += tok[2:]
            if lab != "O" and cur_lab == "O":
                cur_lab = lab
        else:
            if cur_word:
                words.append(cur_word)
                wlabels.append(cur_lab)
            cur_word = tok
            cur_lab = lab

    if cur_word:
        words.append(cur_word)
        wlabels.append(cur_lab)

    print("\n=== Highlighted Output ===")
    for w, lab in zip(words, wlabels):
        if lab != "O":
            print(f"[{w} ({lab})]", end=" ")
        else:
            print(w, end=" ")
    print("\n")


# Test run
test_sentence = "Your temporary password is r3JpIrYy(Z. Please reset it after your first login."
highlight(test_sentence)
