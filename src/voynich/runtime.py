"""Shared corpus windows, metrics, and run provenance; no windows cross pages."""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import torch
import torch.nn.functional as F

from .tokenizer import EVATokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def corpus_identity(data_dir, preparation_manifest=None):
    # Hash test bytes for provenance only; no test evaluation or selection here.
    root = Path(data_dir)
    actual = {name: digest(root / name) for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "tokenizer.json")}
    if preparation_manifest is None:
        local = root / "preparation.json"
        preparation_manifest = local if local.exists() else root.parent.parent / "manifests" / f"{root.name}_preparation.json"
    manifest_path = Path(preparation_manifest)
    if not manifest_path.is_file():
        raise ValueError("A registered preparation manifest is required before training/evaluation")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get("derived_sha256")
    if expected != actual:
        raise ValueError("Processed corpus/tokenizer differs from the registered preparation manifest; inspect drift")
    return actual


def environment():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    root = Path(__file__).resolve().parents[2]
    source_files = [*root.glob("src/voynich/*.py"), *root.glob("configs/*.json"),
                    *root.glob("data/manifests/*.json"), root / "pyproject.toml", root / "uv.lock"]
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")), "threads": torch.get_num_threads(),
        "source_sha256": {str(path.relative_to(root)): digest(path) for path in sorted(source_files) if path.is_file()},
    }


def resolve_device(requested="auto"):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable in this execution environment")
    return requested


@dataclass
class Window:
    page_id: str
    sequence: list[int]
    start: int
    length: int


class PageWindows:
    def __init__(self, data_dir, split, context_length, tokenizer=None, *, allow_test=False):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if split == "test" and not allow_test:
            raise ValueError("Test evaluation requires explicit allow_test=True after model selection is frozen")
        if context_length < 1:
            raise ValueError("context_length must be positive")
        self.context_length = context_length
        self.tokenizer = tokenizer or EVATokenizer.load(Path(data_dir) / "tokenizer.json")
        self.ignored_ids = set(self.tokenizer.uncertainty_ids) | {self.tokenizer.pad_id, self.tokenizer.unk_id}
        self.pages = [json.loads(line) for line in (Path(data_dir) / f"{split}.jsonl").read_text().splitlines() if line]
        if not self.pages:
            raise ValueError(f"Empty {split} split")
        self.sequences = {}
        self.windows = []
        for page in self.pages:
            if page["split"] != split:
                raise ValueError("Split label mismatch")
            key = page["page_id"]
            if key in self.sequences:
                raise ValueError(f"Duplicate page {key}")
            ids = self.tokenizer.encode(page["text"], add_bos=True, add_eos=True)
            self.sequences[key] = ids
            for start in range(0, len(ids) - 1, context_length):
                self.windows.append(Window(key, ids, start, min(context_length, len(ids) - 1 - start)))

    def batch(self, indices, device, horizons=(1,)):
        pad = self.tokenizer.pad_id
        width = self.context_length
        inputs = torch.full((len(indices), width), pad, dtype=torch.long)
        targets = {h: torch.full_like(inputs, -100) for h in horizons}
        page_ids = []
        for row, index in enumerate(indices):
            window = self.windows[int(index)]
            seq, start, count = window.sequence, window.start, window.length
            inputs[row, :count] = torch.tensor(seq[start:start + count])
            for h in horizons:
                values = seq[start + h:start + h + count]
                if values:
                    targets[h][row, :len(values)] = torch.tensor(values)
            page_ids.append(window.page_id)
        for h in targets:
            for ignored in self.ignored_ids:
                targets[h].masked_fill_(targets[h].eq(ignored), -100)
            # Only an actual input position can contribute a horizon objective.
            targets[h].masked_fill_(inputs.eq(pad), -100)
            targets[h] = targets[h].to(device)
        return inputs.to(device), targets, page_ids


def loss_sum(logits, targets):
    count = targets.ne(-100).sum()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100, reduction="sum")
    return loss, count


@torch.no_grad()
def evaluate_model(model, windows, device, batch_size=8):
    was_training = model.training
    model.eval()
    totals = defaultdict(lambda: [0.0, 0, 0])
    try:
        for start in range(0, len(windows.windows), batch_size):
            indices = range(start, min(start + batch_size, len(windows.windows)))
            x, targets, pages = windows.batch(indices, device)
            logits = model(x).logits
            losses = F.cross_entropy(logits.transpose(1, 2), targets[1], reduction="none", ignore_index=-100)
            correct = logits.argmax(-1).eq(targets[1])
            for i, page in enumerate(pages):
                valid = targets[1][i].ne(-100)
                totals[page][0] += float(losses[i].sum().cpu())
                totals[page][1] += int(valid.sum().cpu())
                totals[page][2] += int((correct[i] & valid).sum().cpu())
    finally:
        model.train(was_training)
    nll = sum(x[0] for x in totals.values())
    count = sum(x[1] for x in totals.values())
    if count == 0:
        raise ValueError("No scorable tokens")
    return {
        "nll_nats": nll / count, "bits_per_token": nll / count / math.log(2),
        "accuracy": sum(x[2] for x in totals.values()) / count, "scored_tokens": count,
        "metric_units": "EVA transcription units including definite spaces, newlines and EOS; uncertainty/UNK excluded",
        "context_policy": "non-overlapping page windows; reset context per window; every next-token target scored once",
        "pages": {key: {"bits_per_token": val[0] / val[1] / math.log(2), "scored_tokens": val[1]}
                  for key, val in totals.items() if val[1]},
    }
