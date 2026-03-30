import os
import json
import math
import gc
import random
from datetime import datetime

import numpy as np 
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)

from seqeval.metrics import classification_report, precision_score, recall_score, f1_score


# ============================================================
# CONFIG
# ============================================================

NER_PATH = "../ner_dataset.conll"  # same assumption as the original script
OUTPUT_ROOT = "./ner_benchmark_results"
MAX_LENGTH = 128
RANDOM_SEED = 42
TEST_SIZE = 0.2
NUM_EPOCHS = 5  # bumped up from 3 to test whether extra epochs help
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 20

# Main lightweight comparison set.
# Edit this list freely if you want to add/remove models.
MODEL_CANDIDATES = {
    "TinyBERT": "huawei-noah/TinyBERT_General_4L_312D",
    "MobileBERT": "google/mobilebert-uncased",
    "DistilBERT": "distilbert-base-uncased",
    "MiniLM": "microsoft/MiniLM-L12-H384-uncased",
    # Optional extra:
    # "ALBERT": "albert-base-v2",
}


# ============================================================
# HELPERS
# ============================================================


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

    if cur_tokens:
        sentences.append(cur_tokens)
        tags.append(cur_tags)

    return sentences, tags


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


class MetricsComputer:
    def __init__(self, id2label):
        self.id2label = id2label

    def __call__(self, eval_pred):
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
                cur_true.append(self.id2label[l])
                cur_pred.append(self.id2label[p])
            true.append(cur_true)
            pred.append(cur_pred)

        return {
            "precision": precision_score(true, pred),
            "recall": recall_score(true, pred),
            "f1": f1_score(true, pred),
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



def maybe_fast_tokenizer(model_name):
    """Try fast tokenizer first because word_ids() requires it. Fall back if needed."""
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name)



def encode_and_align(tokenizer, sentences, tags, label2id, max_length=128):
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



def build_datasets(tokenizer, train_sentences, train_tags, val_sentences, val_tags, label2id):
    print("Encoding training data...")
    X_train_ids, X_train_mask, y_train_ids = encode_and_align(
        tokenizer, train_sentences, train_tags, label2id, max_length=MAX_LENGTH
    )

    print("Encoding validation data...")
    X_val_ids, X_val_mask, y_val_ids = encode_and_align(
        tokenizer, val_sentences, val_tags, label2id, max_length=MAX_LENGTH
    )

    train_dataset = NERDataset(X_train_ids, X_train_mask, y_train_ids)
    val_dataset = NERDataset(X_val_ids, X_val_mask, y_val_ids)
    return train_dataset, val_dataset



def get_detailed_report(trainer, val_dataset, id2label):
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

    report = classification_report(true_labels, true_preds)
    return report



def train_one_model(display_name, model_name, train_sentences, train_tags, val_sentences, val_tags,
                    label_set, label2id, id2label):
    print("\n" + "=" * 70)
    print(f"Starting benchmark run for {display_name}: {model_name}")
    print("=" * 70)

    model_slug = display_name.lower().replace(" ", "_")
    run_dir = os.path.join(OUTPUT_ROOT, model_slug)
    os.makedirs(run_dir, exist_ok=True)

    tokenizer = maybe_fast_tokenizer(model_name)
    train_dataset, val_dataset = build_datasets(
        tokenizer, train_sentences, train_tags, val_sentences, val_tags, label2id
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_set),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = MetricsComputer(id2label)

    training_args = TrainingArguments(
        output_dir=run_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir=os.path.join(run_dir, "logs"),
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
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

    trainer.train()
    metrics = trainer.evaluate()
    report = get_detailed_report(trainer, val_dataset, id2label)

    save_dir = os.path.join(run_dir, "best_model")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    param_count = count_parameters(model)
    approx_size_mb = param_count * 4 / (1024 ** 2)  # rough fp32 estimate

    result = {
        "display_name": display_name,
        "model_name": model_name,
        "epochs": NUM_EPOCHS,
        "eval_loss": float(metrics.get("eval_loss", math.nan)),
        "precision": float(metrics.get("eval_precision", math.nan)),
        "recall": float(metrics.get("eval_recall", math.nan)),
        "f1": float(metrics.get("eval_f1", math.nan)),
        "runtime_sec": float(metrics.get("eval_runtime", math.nan)),
        "samples_per_sec": float(metrics.get("eval_samples_per_second", math.nan)),
        "steps_per_sec": float(metrics.get("eval_steps_per_second", math.nan)),
        "parameter_count": int(param_count),
        "approx_size_mb": round(approx_size_mb, 2),
        "saved_model_dir": save_dir,
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{display_name} metrics:")
    for key, value in result.items():
        if key in {"display_name", "model_name", "saved_model_dir"}:
            continue
        print(f"  {key}: {value}")

    print("\nDetailed NER Classification Report:")
    print(report)

    # Free memory before next model.
    del trainer
    del model
    del tokenizer
    del train_dataset
    del val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result



def print_ranked_results(results):
    print("\n" + "#" * 70)
    print("FINAL RANKING (sorted by F1, highest first)")
    print("#" * 70)

    ranked = sorted(results, key=lambda x: x["f1"], reverse=True)

    header = (
        f"{'Rank':<6}{'Model':<15}{'F1':<10}{'Precision':<12}"
        f"{'Recall':<10}{'Size(MB)':<12}{'Params':<15}"
    )
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(ranked, start=1):
        print(
            f"{idx:<6}{row['display_name']:<15}{row['f1']:<10.4f}"
            f"{row['precision']:<12.4f}{row['recall']:<10.4f}"
            f"{row['approx_size_mb']:<12.2f}{row['parameter_count']:<15}"
        )

    summary_path = os.path.join(OUTPUT_ROOT, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)

    csv_path = os.path.join(OUTPUT_ROOT, "benchmark_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "rank,display_name,model_name,f1,precision,recall,eval_loss,epochs,"
            "runtime_sec,samples_per_sec,steps_per_sec,parameter_count,approx_size_mb,saved_model_dir\n"
        )
        for idx, row in enumerate(ranked, start=1):
            f.write(
                f"{idx},{row['display_name']},{row['model_name']},{row['f1']},{row['precision']},"
                f"{row['recall']},{row['eval_loss']},{row['epochs']},{row['runtime_sec']},"
                f"{row['samples_per_sec']},{row['steps_per_sec']},{row['parameter_count']},"
                f"{row['approx_size_mb']},{row['saved_model_dir']}\n"
            )

    print(f"\nSaved ranked summary to:\n  {summary_path}\n  {csv_path}")


# ============================================================
# MAIN
# ============================================================


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("Loading NER dataset...")
    sentences, tags = read_conll(NER_PATH)
    print(f"Loaded {len(sentences)} sentences from {NER_PATH}")

    label_set = sorted({t for sent in tags for t in sent})
    label2id = {label: i for i, label in enumerate(label_set)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Number of labels: {len(label_set)}")
    print("Label set:", label_set)

    train_sentences, val_sentences, train_tags, val_tags = train_test_split(
        sentences, tags, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print(f"Train sentences: {len(train_sentences)}")
    print(f"Val sentences:   {len(val_sentences)}")
    print(f"Epochs per model: {NUM_EPOCHS}")
    print("Models to benchmark:")
    for display_name, model_name in MODEL_CANDIDATES.items():
        print(f"  - {display_name}: {model_name}")

    set_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    results = []
    for display_name, model_name in MODEL_CANDIDATES.items():
        try:
            result = train_one_model(
                display_name,
                model_name,
                train_sentences,
                train_tags,
                val_sentences,
                val_tags,
                label_set,
                label2id,
                id2label,
            )
            results.append(result)
        except Exception as exc:
            print(f"\nERROR while running {display_name} ({model_name}): {exc}")

    if not results:
        raise RuntimeError("No model runs completed successfully.")

    print_ranked_results(results)


if __name__ == "__main__":
    main()
