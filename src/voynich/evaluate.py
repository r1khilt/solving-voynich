"""Comparable model, n-gram and local-copy scores; test access is explicitly gated."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import sys

import torch

from .runtime import PageWindows, corpus_identity, digest, environment, evaluate_model, resolve_device, write_json
from .train import load_checkpoint


class NGram:
    """Hierarchical count smoothing, fitted only to training-page sequences."""
    def __init__(self, vocab_size, order=5, alpha=0.1, concentration=5.0):
        if order < 1 or alpha <= 0 or concentration <= 0:
            raise ValueError("Invalid n-gram configuration")
        self.vocab_size, self.order, self.alpha, self.concentration = vocab_size, order, alpha, concentration
        self.counts = [defaultdict(Counter) for _ in range(order)]

    def fit(self, data):
        for sequence in data.sequences.values():
            for position in range(1, len(sequence)):
                target = sequence[position]
                if target in data.ignored_ids:
                    continue
                for size in range(min(self.order - 1, position) + 1):
                    context = tuple(sequence[position-size:position]) if size else ()
                    self.counts[size][context][target] += 1
        return self

    def probability(self, prefix, target):
        counter = self.counts[0][()]
        probability = (counter[target] + self.alpha) / (counter.total() + self.alpha * self.vocab_size)
        for size in range(1, min(len(prefix), self.order - 1) + 1):
            counter = self.counts[size].get(tuple(prefix[-size:]))
            if counter:
                probability = (counter[target] + self.concentration * probability) / (counter.total() + self.concentration)
        return probability


def copy_probability(prefix, target, unigram, max_match=8, weight=0.8):
    """Longest suffix match with its continuation entirely before the target."""
    base = unigram.probability(prefix, target)
    for size in range(min(max_match, len(prefix) - 1), 0, -1):
        suffix = prefix[-size:]
        followers = Counter(prefix[j + size] for j in range(len(prefix) - size)
                            if prefix[j:j + size] == suffix)
        if followers:
            return weight * followers[target] / followers.total() + (1 - weight) * base
    return base


def score_baseline(data, probability):
    totals = defaultdict(lambda: [0.0, 0])
    for window in data.windows:
        context = window.sequence[window.start:window.start + window.length]
        for offset in range(window.length):
            target = window.sequence[window.start + offset + 1]
            if target in data.ignored_ids:
                continue
            p = probability(context[:offset + 1], target)
            if not 0 < p <= 1:
                raise ValueError("Invalid baseline probability")
            totals[window.page_id][0] -= math.log2(p)
            totals[window.page_id][1] += 1
    count = sum(value[1] for value in totals.values())
    if count == 0:
        raise ValueError("No scorable baseline targets")
    return {"bits_per_token": sum(value[0] for value in totals.values()) / count,
            "scored_tokens": count,
            "pages": {key: {"bits_per_token": value[0] / value[1], "scored_tokens": value[1]}
                      for key, value in totals.items() if value[1]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/zl3b")
    parser.add_argument("--preparation-manifest", help="Registered manifest of processed corpus checksums")
    parser.add_argument("--checkpoint")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.checkpoint and not args.baselines:
        parser.error("Provide --checkpoint and/or --baselines")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    identity = corpus_identity(args.data, args.preparation_manifest)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": list(sys.argv),
        "environment": environment(), "device": device,
        "split": args.split, "corpus_identity": identity, "models": {},
    }
    context = args.context
    if args.checkpoint:
        checkpoint_sha256 = digest(args.checkpoint)
        model, payload = load_checkpoint(args.checkpoint, device)
        if payload["corpus_identity"] != identity:
            raise ValueError("Checkpoint and corpus/tokenizer identity differ")
        context = model.config.context_length
    evaluated = PageWindows(args.data, args.split, context, allow_test=args.allow_test)
    if args.checkpoint:
        report["models"]["transformer"] = evaluate_model(model, evaluated, device)
        report["checkpoint"] = str(args.checkpoint)
        report["checkpoint_sha256"] = checkpoint_sha256
        if digest(args.checkpoint) != checkpoint_sha256:
            raise ValueError("Checkpoint changed during evaluation; repeat with immutable weights")
    if args.baselines:
        train_data = PageWindows(args.data, "train", context, evaluated.tokenizer)
        unigram = NGram(evaluated.tokenizer.vocab_size, order=1).fit(train_data)
        ngram = NGram(evaluated.tokenizer.vocab_size, order=5).fit(train_data)
        report["models"]["unigram"] = score_baseline(evaluated, unigram.probability)
        report["models"]["fivegram"] = score_baseline(evaluated, ngram.probability)
        report["models"]["local_copy_unigram"] = score_baseline(
            evaluated, lambda prefix, target: copy_probability(prefix, target, unigram))
        report["baseline_config"] = {"ngram_order": 5, "alpha": 0.1, "concentration": 5.0,
                                     "copy_max_match": 8, "copy_weight": 0.8}
    report["context_length"] = context
    report["metric_units"] = "EVA units with spaces/newlines/EOS; uncertainty/UNK excluded, same windows for all models"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, report)
    print(json.dumps({key: value["bits_per_token"] for key, value in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
