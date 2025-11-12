from generator import *
import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

import yaml  # pip install pyyaml

PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z0-9_]+)\}")


# -----------------------------------------------------
#  LOAD ALL YAML TEMPLATES FROM A DIRECTORY
# -----------------------------------------------------
def load_all_templates_from_dir(dir_path: str | Path) -> List[Dict[str, Any]]:
    dir_path = Path(dir_path)
    templates = []

    for file in dir_path.glob("*.yaml"):
        text = file.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, list):
            raise ValueError(f"YAML file {file} must contain a list of templates")
        templates.extend(data)

    for file in dir_path.glob("*.yml"):
        text = file.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, list):
            raise ValueError(f"YAML file {file} must contain a list of templates")
        templates.extend(data)

    print(f"Loaded {len(templates)} total templates from directory: {dir_path}")
    return templates


# -----------------------------------------------------
#  PLACEHOLDER FILLING (same as before)
# -----------------------------------------------------
def fill_template_once(template: Dict[str, Any],
                       generators: Dict[str, callable]) -> Dict[str, Any]:
    filled = dict(template)
    placeholder_values: Dict[str, str] = {}

    def replace_placeholders(raw_text: str) -> str:
        def _replace(match: re.Match):
            key = match.group(1)
            if key in placeholder_values:
                return placeholder_values[key]
            gen_fn = generators.get(key)
            if gen_fn is None:
                return match.group(0)  # leave untouched
            value = gen_fn()
            placeholder_values[key] = value
            return value

        return PLACEHOLDER_PATTERN.sub(_replace, raw_text)

    original_text = template["text"]
    filled_text = replace_placeholders(original_text)
    filled["text"] = filled_text

    spans_out = []
    if "spans" in template and template["spans"]:
        search_start = 0
        for span in template["spans"]:
            span_copy = dict(span)
            span_text_filled = replace_placeholders(span["text"])
            span_copy["text"] = span_text_filled

            reasons = span.get("reason") or []
            if isinstance(reasons, list) and reasons:
                label = reasons[0]
            else:
                label = "UNKNOWN"
            span_copy["label"] = label

            idx = filled_text.find(span_text_filled, search_start)
            if idx == -1:
                span_copy["start"] = -1
                span_copy["end"] = -1
            else:
                span_copy["start"] = idx
                span_copy["end"] = idx + len(span_text_filled)
                search_start = span_copy["end"]

            spans_out.append(span_copy)

    filled["spans"] = spans_out
    return filled


# -----------------------------------------------------
#  REPEAT EACH TEMPLATE 5x WITH NEW RANDOM VALUES
# -----------------------------------------------------
def expand_templates(templates: List[Dict[str, Any]],
                     generators: Dict[str, callable],
                     repeats: int = 5) -> List[Dict[str, Any]]:
    expanded = []
    for tpl in templates:
        base_id = tpl.get("id", "NOID")
        for i in range(1, repeats + 1):
            filled = fill_template_once(tpl, generators)
            filled["id"] = f"{base_id}_{i}"
            expanded.append(filled)
    print(f"Expanded templates to {len(expanded)} total examples.")
    return expanded


# -----------------------------------------------------
#  TOKENIZATION & BIO LABELS (same as before)
# -----------------------------------------------------
def simple_tokenize_with_offsets(text: str):
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    return [(m.group(0), m.start(), m.end()) for m in pattern.finditer(text)]


def spans_to_bio_labels(text: str, spans: List[Dict[str, Any]], outside_label="O"):
    tokens = simple_tokenize_with_offsets(text)
    labels = [outside_label] * len(tokens)

    for span in spans:
        start, end = span["start"], span["end"]
        label = span["label"]
        if start < 0:
            continue

        first = True
        for i, (_, s, e) in enumerate(tokens):
            if not (e <= start or s >= end):
                prefix = "B" if first else "I"
                labels[i] = f"{prefix}-{label}"
                first = False

    return [(tok, lbl) for (tok, _, _), lbl in zip(tokens, labels)]


# -----------------------------------------------------
#  EXPORT: CoNLL WITH HEADERS
# -----------------------------------------------------
def export_conll_with_headers(examples: List[Dict[str, Any]],
                              out_path: str | Path,
                              outside_label="O"):
    out_path = Path(out_path)
    lines = []

    for ex in examples:
        ex_id = ex.get("id", "")
        text = ex["text"]
        sensitive = ex.get("sensitive", False)

        lines.append(f"# id = {ex_id}")
        lines.append(f"# text = {text}")
        lines.append(f"# sensitive = {str(bool(sensitive)).lower()}")
        lines.append("")

        for token, label in spans_to_bio_labels(text, ex["spans"], outside_label):
            lines.append(f"{token}\t{label}")

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Wrote CoNLL+headers → {out_path}")


# -----------------------------------------------------
#  EXPORT: Sentence-level JSONL
# -----------------------------------------------------
def export_sentence_jsonl(examples: List[Dict[str, Any]],
                          out_path: str | Path):
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "id": ex["id"],
                "text": ex["text"],
                "sensitive": bool(ex.get("sensitive", False))
            }, ensure_ascii=False) + "\n")

    print(f"✅ Wrote sentence-level JSONL → {out_path}")


# -----------------------------------------------------
#  MAIN PIPELINE
# -----------------------------------------------------
def build_datasets(
    template_dir: str | Path,
    conll_out: str | Path,
    sentence_jsonl_out: str | Path,
    generators: Dict[str, callable],
    repeats: int = 5,
):
    templates = load_all_templates_from_dir(template_dir)
    expanded_examples = expand_templates(templates, generators, repeats)

    export_conll_with_headers(expanded_examples, conll_out)
    export_sentence_jsonl(expanded_examples, sentence_jsonl_out)


# -----------------------------------------------------
#  RUN
# -----------------------------------------------------
if __name__ == "__main__":
    build_datasets(
        template_dir="Templates/",        # directory of .yaml files
        conll_out="ner_dataset.conll",   # output CoNLL
        sentence_jsonl_out="sentences.jsonl",  # output sentence-level JSONL
        generators=INDEX,
        repeats=10
    )
