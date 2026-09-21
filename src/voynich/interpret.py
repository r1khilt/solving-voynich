"""Controlled causal interventions with absolute and normalized effects.

This diagnoses a model, not historical semantics. Context corruption is user-chosen;
the report preserves that choice and does not claim an automatically valid counterfactual.
"""

import argparse
import json
from numbers import Integral
from pathlib import Path
import re

import torch

from .runtime import digest, resolve_device, write_json
from .tokenizer import EVATokenizer
from .train import load_checkpoint


def patch_activation(donor, *, positions=None, head=None):
    """Replace [B,T,D] residuals or [B,T,H,D] head outputs with aligned donor values."""
    def patch(value):
        if value.ndim not in (3, 4):
            raise ValueError("Patch requires [B,T,D] or [B,T,H,D] activation data")
        if value.shape != donor.shape:
            raise ValueError("Donor and recipient shapes must match exactly")
        source = donor.to(device=value.device, dtype=value.dtype)
        result = value.clone()
        selected = slice(None) if positions is None else positions
        if head is None:
            result[:, selected] = source[:, selected]
        else:
            if value.ndim != 4 or not isinstance(head, Integral) or isinstance(head, bool) or not 0 <= head < value.shape[2]:
                raise ValueError("Head patch requires [B,T,H,D] data and a valid head")
            result[:, selected, head] = source[:, selected, head]
        return result
    return patch


def ablate_head(head):
    def ablate(value):
        if value.ndim != 4 or not isinstance(head, Integral) or isinstance(head, bool) or not 0 <= head < value.shape[2]:
            raise ValueError("Head ablation requires [B,T,H,D] data and a valid head")
        result = value.clone()
        result[:, :, head] = 0
        return result
    return ablate


def target_log_probability(logits, target_id):
    return float(logits[:, -1].log_softmax(-1)[:, target_id].mean().cpu())


def validate_report_site(site, head, n_heads):
    """Allow only known time-axis layouts; tensor rank cannot identify axis semantics."""
    vector_head_sites = {
        "attn.q_pre_norm", "attn.k_pre_norm", "attn.q_normalized", "attn.k_normalized",
        "attn.q", "attn.k", "attn.v", "attn.z", "attn.z_gated", "attn.result",
    }
    time_vector_sites = {
        "resid_pre", "resid_mid", "resid_post", "attn_input", "mlp_input", "attn.out",
        "attn.gate", "mlp.up", "mlp.gate", "mlp.act", "mlp.out",
    }
    match = re.fullmatch(r"blocks\.\d+\.(.+)", site)
    suffix = match[1] if match else None
    has_vector_heads = suffix in vector_head_sites
    if site not in {"embed", "final_norm", "logits"} and suffix not in time_vector_sites | vector_head_sites:
        raise ValueError(
            f"Unsupported activation layout for patch_report: {site}. "
            "Use model interventions directly for attention scores/patterns or other layouts."
        )
    if head is not None and (not has_vector_heads or not isinstance(head, Integral)
                             or isinstance(head, bool) or not 0 <= head < n_heads):
        raise ValueError("Head selection requires a vector-head activation site and a valid integer head")


@torch.no_grad()
def patch_report(model, clean, corrupted, target_id, site, *, positions=None, head=None, seed=42):
    """Patch aligned time-axis sites; scores/patterns require direct model interventions.

    Scalar attention gates can be patched across positions with head=None. Selecting
    one gate requires a custom intervention because its layout is [B,T,H], not [B,T,H,D].
    """
    if clean.ndim != 2 or corrupted.ndim != 2 or clean.shape[0] == 0:
        raise ValueError("Interpretation contexts must have nonempty [batch, time] shape")
    if clean.shape != corrupted.shape:
        raise ValueError("Use aligned clean/corrupted sequences of identical shape")
    if (clean.shape[1] < 2 or not isinstance(target_id, Integral) or isinstance(target_id, bool)
            or not 0 <= target_id < model.config.vocab_size):
        raise ValueError("Need a nonempty context and valid target")
    if clean.eq(model.config.pad_id).any() or corrupted.eq(model.config.pad_id).any():
        raise ValueError("Interpretation contexts must be unpadded")
    validate_report_site(site, head, model.config.n_heads)
    was_training = model.training
    model.eval()
    try:
        clean_output = model(clean, cache_names=[site])
        corrupt_output = model(corrupted, cache_names=[site])
        clean_score = target_log_probability(clean_output.logits, target_id)
        corrupt_score = target_log_probability(corrupt_output.logits, target_id)
        donor = clean_output.cache[site]
        identity = model(corrupted, interventions={site: patch_activation(
            corrupt_output.cache[site], positions=positions, head=head)})
        patched = model(corrupted, interventions={site: patch_activation(donor, positions=positions, head=head)})
        # Matched-length shuffled donor is a negative diagnostic, not a semantic counterfactual.
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(clean.shape[1] - 1, generator=generator).to(clean.device) + 1
        random_context = torch.cat((clean[:, :1], clean[:, permutation]), dim=1)
        random_donor = model(random_context, cache_names=[site]).cache[site]
        random_patch = model(corrupted, interventions={site: patch_activation(
            random_donor, positions=positions, head=head)})
        reverse = model(clean, interventions={site: patch_activation(
            corrupt_output.cache[site], positions=positions, head=head)})
        patched_score = target_log_probability(patched.logits, target_id)
        gap = clean_score - corrupt_score
        return {
            "site": site, "positions": positions if positions is not None else "all", "head": head,
            "metric": "log probability of specified next token (natural log)",
            "clean": clean_score, "corrupted": corrupt_score, "clean_corrupted_gap": gap,
            "patched": patched_score, "absolute_patch_effect": patched_score - corrupt_score,
            "normalized_recovery": (patched_score - corrupt_score) / gap if abs(gap) >= 1e-4 else None,
            "normalized_recovery_warning": "Undefined below |gap|=1e-4; unstable near zero and may exceed [0,1]",
            "identity_patch": target_log_probability(identity.logits, target_id),
            "identity_max_logit_difference": float((identity.logits - corrupt_output.logits).abs().max().cpu()),
            "shuffled_donor_patch": target_log_probability(random_patch.logits, target_id),
            "reverse_patch": target_log_probability(reverse.logits, target_id),
            "shuffled_donor_ids": random_context.cpu().tolist(), "seed": seed,
            "interpretation": "Single-pair diagnostic; no circuit discovery, semantic claim, or historical inference",
        }
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--clean", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--target", required=True, help="Exactly one transcription token")
    parser.add_argument("--site", default="blocks.0.attn.result")
    parser.add_argument("--head", type=int)
    parser.add_argument("--position", type=int, help="Omit to patch every position; -1 means final position")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else Path(args.checkpoint).parent / "tokenizer.json"
    if digest(tokenizer_path) != payload["corpus_identity"]["tokenizer.json"]:
        raise ValueError("Tokenizer does not match checkpoint provenance")
    tokenizer = EVATokenizer.load(tokenizer_path)
    def encode(text):
        return torch.tensor([tokenizer.encode(text, add_bos=True, add_eos=False)], device=device)
    target = tokenizer.encode(args.target, add_bos=False, add_eos=False)
    if len(target) != 1 or target[0] in set(tokenizer.uncertainty_ids) | {tokenizer.unk_id}:
        raise ValueError("Target must be exactly one known, unambiguous transcription token")
    report = patch_report(model, encode(args.clean), encode(args.corrupted), target[0], args.site,
                          positions=None if args.position is None else [args.position], head=args.head, seed=args.seed)
    report.update({"checkpoint": args.checkpoint, "checkpoint_sha256": digest(args.checkpoint),
                   "clean_text": args.clean, "corrupted_text": args.corrupted, "target": args.target,
                   "corpus_identity": payload["corpus_identity"]})
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
