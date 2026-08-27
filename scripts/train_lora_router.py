#!/usr/bin/env python3
"""Train a lightweight instruction -> app router.

This implementation uses a pure-Python multinomial Naive Bayes classifier so it
works in restricted environments without sklearn.

Inputs are JSONL records exported by scripts/build_lora_router_dataset.py.
Outputs:
- router.pkl: pickled model state
- router.json: metadata and training summary
- router.pt: optional torch-saved state when torch is available
"""

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight router classifier.")
    parser.add_argument("--train-file", type=Path, required=True, help="router_train.jsonl path.")
    parser.add_argument("--val-file", type=Path, default=None, help="router_val.jsonl path.")
    parser.add_argument("--output-dir", type=Path, default=Path("router_model"), help="Output directory.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing factor.")
    parser.add_argument("--min-token-count", type=int, default=1, help="Drop tokens below this global count.")
    parser.add_argument("--max-vocab-size", type=int, default=50000, help="Keep at most this many tokens.")
    return parser.parse_args()


def read_jsonl(path):
    if path is None:
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def tokenize(text):
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


class RouterState(object):
    def __init__(self, labels, vocab, label_priors, token_log_probs, unknown_log_prob, alpha, min_token_count, max_vocab_size):
        self.labels = labels
        self.vocab = vocab
        self.label_priors = label_priors
        self.token_log_probs = token_log_probs
        self.unknown_log_prob = unknown_log_prob
        self.alpha = alpha
        self.min_token_count = min_token_count
        self.max_vocab_size = max_vocab_size

    def to_dict(self):
        return {
            "labels": self.labels,
            "vocab": self.vocab,
            "label_priors": self.label_priors,
            "token_log_probs": self.token_log_probs,
            "unknown_log_prob": self.unknown_log_prob,
            "alpha": self.alpha,
            "min_token_count": self.min_token_count,
            "max_vocab_size": self.max_vocab_size,
        }


class NaiveBayesRouter(object):
    def __init__(self, alpha=1.0, min_token_count=1, max_vocab_size=50000):
        self.alpha = alpha
        self.min_token_count = min_token_count
        self.max_vocab_size = max_vocab_size
        self.state = None

    def fit(self, examples):
        if not examples:
            raise ValueError("Empty training set.")

        label_counts = Counter()
        global_token_counts = Counter()
        token_counts_by_label = defaultdict(Counter)

        for ex in examples:
            label = str(ex.get("label") or ex.get("app_name") or ex.get("run_id") or "").strip()
            text = str(ex.get("instruction") or "").strip()
            if not label or not text:
                continue
            label_counts[label] += 1
            tokens = tokenize(text)
            global_token_counts.update(tokens)
            token_counts_by_label[label].update(tokens)

        if not label_counts:
            raise ValueError("No valid labeled examples after filtering.")

        vocab_items = [(tok, cnt) for tok, cnt in global_token_counts.items() if cnt >= self.min_token_count]
        vocab_items.sort(key=lambda x: (-x[1], x[0]))
        vocab_items = vocab_items[: self.max_vocab_size]
        vocab = [tok for tok, _ in vocab_items]
        vocab_set = set(vocab)

        labels = sorted(label_counts)
        total_examples = float(sum(label_counts.values()))
        label_priors = {label: math.log(label_counts[label] / total_examples) for label in labels}

        token_log_probs = {}
        unknown_log_prob = {}
        vocab_size = max(len(vocab), 1)
        for label in labels:
            counts = token_counts_by_label[label]
            total_tokens = sum(counts[tok] for tok in vocab_set)
            denom = total_tokens + self.alpha * vocab_size
            token_log_probs[label] = {
                tok: math.log((counts.get(tok, 0) + self.alpha) / denom)
                for tok in vocab
            }
            unknown_log_prob[label] = math.log(self.alpha / denom)

        self.state = RouterState(
            labels=labels,
            vocab=vocab,
            label_priors=label_priors,
            token_log_probs=token_log_probs,
            unknown_log_prob=unknown_log_prob,
            alpha=self.alpha,
            min_token_count=self.min_token_count,
            max_vocab_size=self.max_vocab_size,
        )
        return self

    def _score(self, text):
        if self.state is None:
            raise RuntimeError("Model is not fitted.")
        tokens = tokenize(text)
        token_counts = Counter(tokens)
        scores = {}
        for label in self.state.labels:
            score = self.state.label_priors[label]
            token_probs = self.state.token_log_probs[label]
            unk = self.state.unknown_log_prob[label]
            for tok, cnt in token_counts.items():
                score += cnt * token_probs.get(tok, unk)
            scores[label] = score
        return scores

    def predict_one(self, text):
        scores = self._score(text)
        return max(scores, key=scores.get)

    def evaluate(self, examples):
        total = 0
        correct = 0
        per_label_total = Counter()
        per_label_correct = Counter()
        confusion = defaultdict(Counter)

        for ex in examples:
            label = str(ex.get("label") or ex.get("app_name") or ex.get("run_id") or "").strip()
            text = str(ex.get("instruction") or "").strip()
            if not label or not text:
                continue
            pred = self.predict_one(text)
            total += 1
            per_label_total[label] += 1
            confusion[label][pred] += 1
            if pred == label:
                correct += 1
                per_label_correct[label] += 1

        return {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
            "per_label_total": dict(per_label_total),
            "per_label_correct": dict(per_label_correct),
            "confusion": {label: dict(counter) for label, counter in confusion.items()},
        }

    def to_state(self):
        if self.state is None:
            raise RuntimeError("Model is not fitted.")
        return self.state.to_dict()

    @classmethod
    def from_state(cls, state):
        obj = cls(
            alpha=state.get("alpha", 1.0),
            min_token_count=state.get("min_token_count", 1),
            max_vocab_size=state.get("max_vocab_size", 50000),
        )
        obj.state = RouterState(
            labels=list(state["labels"]),
            vocab=list(state["vocab"]),
            label_priors=dict(state["label_priors"]),
            token_log_probs={k: dict(v) for k, v in state["token_log_probs"].items()},
            unknown_log_prob=dict(state["unknown_log_prob"]),
            alpha=state.get("alpha", 1.0),
            min_token_count=state.get("min_token_count", 1),
            max_vocab_size=state.get("max_vocab_size", 50000),
        )
        return obj


def maybe_save_torch_state(state, path):
    try:
        import torch  # type: ignore
    except Exception:
        return False
    torch.save(state, path)
    return True


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)

    router = NaiveBayesRouter(
        alpha=args.alpha,
        min_token_count=args.min_token_count,
        max_vocab_size=args.max_vocab_size,
    ).fit(train_rows)

    train_metrics = router.evaluate(train_rows)
    val_metrics = router.evaluate(val_rows) if val_rows else {}

    model_state = router.to_state()
    pkl_path = args.output_dir / "router.pkl"
    json_path = args.output_dir / "router.json"
    pt_path = args.output_dir / "router.pt"

    with pkl_path.open("wb") as f:
        pickle.dump(model_state, f)

    torch_saved = maybe_save_torch_state(model_state, pt_path)
    if not torch_saved and pt_path.exists():
        pt_path.unlink()

    summary = {
        "train_file": str(args.train_file),
        "val_file": str(args.val_file) if args.val_file else None,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "labels": router.state.labels if router.state else [],
        "vocab_size": len(router.state.vocab) if router.state else 0,
        "artifacts": {
            "pkl": str(pkl_path),
            "json": str(json_path),
            "pt": str(pt_path) if torch_saved else None,
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
