"""Matched 10.7M-parameter raw/canonical symbol encoders for EXP-0032."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from voynich.exact_count_decode import exact_count_keep_masks
from voynich.latent_recovery import CIPHER_POOL, WORLD_C, build_vocab, copy_aware_features
from voynich.runtime import write_json


EXPERIMENT = "EXP-0032"
SEEDS = (320101, 320202)
VARIANTS = ("raw", "canonical")
STEPS = 3000
EVAL_INTERVAL = 250
MAX_TRAIN_SECONDS = 600
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
VOCAB = build_vocab()
POOL = frozenset(CIPHER_POOL)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_token_ids(text: str) -> list[int]:
    """First-occurrence cipher IDs; Latin cleartext IDs and space remain literal."""
    seen: dict[str, int] = {}
    out = []
    for ch in text:
        if ch in POOL:
            if ch not in seen:
                seen[ch] = len(seen)
            out.append(VOCAB[CIPHER_POOL[seen[ch]]])
        else:
            out.append(VOCAB.get(ch, VOCAB["<unk>"]))
    return out


def raw_token_ids(text: str) -> list[int]:
    return [VOCAB.get(ch, VOCAB["<unk>"]) for ch in text]


def encode_text(text: str, variant: str) -> list[int]:
    if variant == "raw":
        return raw_token_ids(text)
    if variant == "canonical":
        return canonical_token_ids(text)
    raise ValueError(f"unknown variant: {variant}")


class ContextMaskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim = 384
        self.emb = nn.Embedding(len(VOCAB), dim, padding_idx=0)
        self.position = nn.Embedding(128, dim)
        self.feat_proj = nn.Linear(4, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=6, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(dim)
        self.mask_head = nn.Linear(dim, 1)
        self.world_head = nn.Linear(dim, 4)

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != 128:
            raise ValueError("EXP-0032 requires fixed 128-character windows")
        positions = torch.arange(x.shape[1], device=x.device)
        hidden = self.emb(x) + self.position(positions)[None] + self.feat_proj(features)
        hidden = self.final_norm(self.context(hidden))
        return self.mask_head(hidden).squeeze(-1), self.world_head(hidden.mean(dim=1))


def arrays(rows: list[dict], variant: str, device: str) -> dict[str, torch.Tensor]:
    if any(len(row["text"]) != 128 or len(row["mask"]) != 128 for row in rows):
        raise ValueError("sample length drift")
    return {
        "x": torch.tensor([encode_text(row["text"], variant) for row in rows], dtype=torch.long, device=device),
        "features": torch.tensor(np.stack([copy_aware_features(row["text"]) for row in rows]), dtype=torch.float32, device=device),
        "mask": torch.tensor([row["mask"] for row in rows], dtype=torch.float32, device=device),
        "world": torch.tensor([row["world"] for row in rows], dtype=torch.long, device=device),
    }


@torch.no_grad()
def probabilities(model: ContextMaskModel, data: dict[str, torch.Tensor], indices: np.ndarray | list[int]) -> np.ndarray:
    model.eval()
    indices = np.asarray(indices, dtype=np.int64)
    out = []
    for start in range(0, len(indices), 64):
        idx = torch.tensor(indices[start : start + 64], dtype=torch.long, device=data["x"].device)
        logits, _ = model(data["x"][idx], data["features"][idx])
        out.append(logits.sigmoid().float().cpu().numpy())
    return np.concatenate(out, axis=0)


def null_f1(rows: list[dict], masks: list[np.ndarray]) -> float:
    tp = fp = fn = 0
    for row, pred in zip(rows, masks, strict=True):
        true = np.asarray(row["mask"])
        tp += int(((true == 0) & (pred == 0)).sum())
        fp += int(((true == 1) & (pred == 0)).sum())
        fn += int(((true == 0) & (pred == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def train(root: Path, variant: str, seed: int, device: str) -> dict:
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("unregistered variant or seed")
    if device not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; run the authorized local GPU job outside the sandbox")
    torch.set_num_threads(6)
    manifest_path = root / "data/manifests/exp0032_data.json"
    audit_path = root / "results/EXP-0032/data_audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["manifest_sha256"] != digest(manifest_path):
        raise ValueError("data source audit did not pass for this manifest")
    paths = {name: root / f"data/processed/exp0032/{name}.jsonl" for name in ("train", "validation")}
    for name, path in paths.items():
        if digest(path) != manifest["derived_sha256"][name]:
            raise ValueError(f"{name} data checksum changed")
    out_root = root / "outputs/EXP-0032"
    result_root = root / "results/EXP-0032"
    ckpt_path = out_root / f"{variant}_{seed}.pt"
    report_path = result_root / f"train_{variant}_{seed}.json"
    if ckpt_path.exists() or report_path.exists():
        raise FileExistsError("training output exists; refusing overwrite")
    training_rows = [json.loads(line) for line in paths["train"].read_text().splitlines()]
    validation_rows = [json.loads(line) for line in paths["validation"].read_text().splitlines()]
    train_data = arrays(training_rows, variant, device)
    copy_ids = np.asarray([i for i, row in enumerate(validation_rows) if row["world"] == WORLD_C and row["filler_family"] == "copy_mutate"], dtype=np.int64)
    if len(copy_ids) < 200:
        raise ValueError("too few copy validation rows")
    val_copy_rows = [validation_rows[int(i)] for i in copy_ids]
    val_data = arrays(val_copy_rows, variant, device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ContextMaskModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    c_ids = np.asarray([i for i, row in enumerate(training_rows) if row["world"] == WORLD_C])
    other_ids = np.asarray([i for i, row in enumerate(training_rows) if row["world"] != WORLD_C])
    if len(c_ids) < 6000 or len(other_ids) < 3000:
        raise ValueError("training world distribution drift")
    best_f1, best_step, best_state = -1.0, 0, None
    history = []
    start = time.monotonic()
    for step in range(1, STEPS + 1):
        selected = np.concatenate((rng.choice(c_ids, size=22, replace=True), rng.choice(other_ids, size=10, replace=True)))
        rng.shuffle(selected)
        idx = torch.tensor(selected, dtype=torch.long, device=device)
        logits, world_logits = model(train_data["x"][idx], train_data["features"][idx])
        truth = train_data["mask"][idx]
        weights = torch.where(truth < 0.5, 2.0, 1.0)
        loss_mask = F.binary_cross_entropy_with_logits(logits, truth, weight=weights)
        loss_world = F.cross_entropy(world_logits, train_data["world"][idx])
        loss = loss_mask + 0.10 * loss_world
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % EVAL_INTERVAL == 0:
            p = probabilities(model, val_data, list(range(len(val_copy_rows))))
            pred = exact_count_keep_masks([row for row in p])
            score = null_f1(val_copy_rows, pred)
            history.append({"step": step, "train_loss": float(loss.item()), "validation_copy_null_f1": score})
            if score > best_f1:
                best_f1, best_step = score, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            model.train()
    if best_state is None:
        raise AssertionError("no best checkpoint")
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.monotonic() - start
    out_root.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": EXPERIMENT,
        "variant": variant,
        "seed": seed,
        "model": best_state,
        "best_step": best_step,
        "validation_copy_null_f1": best_f1,
        "manifest_sha256": digest(manifest_path),
        "runner_sha256": digest(Path(__file__)),
    }, ckpt_path)
    report = {
        "experiment": EXPERIMENT,
        "variant": variant,
        "seed": seed,
        "device": device,
        "torch": str(torch.__version__),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "steps": STEPS,
        "best_step": best_step,
        "best_validation_copy_null_f1": best_f1,
        "validation_copy_n": len(val_copy_rows),
        "elapsed_seconds": elapsed,
        "history": history,
        "checkpoint_sha256": digest(ckpt_path),
        "manifest_sha256": digest(manifest_path),
        "runner_sha256": digest(Path(__file__)),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    def timed_out(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"{EXPERIMENT} training exceeded {MAX_TRAIN_SECONDS} seconds")
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(MAX_TRAIN_SECONDS)
    result = train(args.root, args.variant, args.seed, args.device)
    print(json.dumps({key: result[key] for key in ("variant", "seed", "parameter_count", "best_step", "best_validation_copy_null_f1", "elapsed_seconds", "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
