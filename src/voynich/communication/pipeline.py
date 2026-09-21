"""Joint training data and independently checked neural explanation proposals."""

from dataclasses import asdict, dataclass
import hashlib
import math
import random

import torch

from .schema import Anchor, FAMILIES, GRAMMARS, MORPHOLOGIES, Observation, object_digest
from .worlds import WorldConfig, decode, generate_world, reencode


@dataclass(frozen=True)
class Episode:
    observation: Observation
    clean: torch.Tensor
    world: object


# Held-out grammar/morphology combinations, including fresh random encoding keys.
HELD_OUT = {"validation": ("VOS", "prefix"), "test": ("OVS", "suffix")}
SEED_BASES = {"train": 71_000_000, "validation": 81_000_000, "test": 91_000_000}


def make_episode(index, split, layout, *, anchor_count=0, seed=41021, family=None):
    if (
        split not in SEED_BASES
        or not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < 1_000_000
    ):
        raise ValueError("Invalid episode index or split")
    if not 0 <= anchor_count <= layout.max_anchors:
        raise ValueError("Anchor count exceeds layout")
    # Split domains cannot overlap even with neighboring run seeds.
    episode_seed = int.from_bytes(hashlib.sha256(f"{seed}:{split}:{index}".encode()).digest()[:8], "big")
    rng = random.Random(episode_seed)
    combinations = [(g, m) for g in GRAMMARS for m in MORPHOLOGIES if (g, m) not in HELD_OUT.values()]
    grammar, morphology = rng.choice(combinations) if split == "train" else HELD_OUT[split]
    selected_family = family or rng.choice(FAMILIES)
    cfg = WorldConfig(
        seed=episode_seed,
        family=selected_family,
        alphabet_size=layout.alphabet_size,
        entities=4,
        events=rng.randint(4, 9),
        null_rate=rng.choice((0.0, 0.1, 0.25)),
        homophones=rng.choice((1, 2)),
        grammar=grammar,
        morphology=morphology,
    )
    world = generate_world(cfg)
    if len(world.observation) > layout.max_observation:
        raise ValueError("Generated episode exceeds layout; increase capacity, never silently truncate")
    present = sorted({x for x in world.observation if world.inverse_key[x - 2] != 0})
    rng.shuffle(present)
    # Deliberately labeled synthetic oracle correspondence supervision, not independent historical evidence.
    observation_id = object_digest(list(world.observation))
    anchors = tuple(
        Anchor(x, world.inverse_key[x - 2], f"synthetic-oracle:{observation_id}")
        for x in present[:anchor_count]
    )
    obs = Observation(
        f"world-{object_digest(list(world.observation))[:20]}",
        world.observation,
        layout.alphabet_size,
        anchors,
        f"observation:{observation_id}",
        split,
    )
    target = layout.pack(
        world.inverse_key,
        world.keep,
        world.word_boundary,
        grammar=cfg.grammar,
        morphology=cfg.morphology,
        family=cfg.family,
    )
    target.masked_fill_(layout.tensors([obs])["fixed"][0].eq(0), 0)
    return Episode(obs, target, world)


def collate(episodes, layout, device="cpu"):
    inputs = layout.tensors([e.observation for e in episodes], device)
    inputs["clean"] = torch.stack([e.clean for e in episodes]).to(device)
    return inputs


def verify_candidate(observation, hypothesis, *, max_null_fraction=0.7, parse_semantics=True):
    """Check exact shared-channel coverage; grammar/semantics checks use no gold targets.

    The candidate language grammar is a declared synthetic hypothesis family, not a
    universal natural-language parser. Re-encoding alone is never treated as truth.
    """
    issues = []
    if not math.isfinite(max_null_fraction) or not 0 <= max_null_fraction < 1:
        raise ValueError("Invalid null bound")
    key = tuple(hypothesis["inverse_key"])
    keep = tuple(hypothesis["keep"])
    boundaries = tuple(hypothesis["boundaries"])
    if (
        len(key) != observation.alphabet_size
        or len(keep) != len(observation.symbols)
        or len(boundaries) != len(keep)
    ):
        raise ValueError("Malformed explanation dimensions")
    if any(k == 1 or not isinstance(k, int) or not 0 <= k < observation.alphabet_size + 2 for k in key):
        raise ValueError("Invalid key category")
    maximum_canonical = 8 if hypothesis["morphology"] == "none" else 10
    if parse_semantics and any(k > maximum_canonical for k in key):
        issues.append("key_outside_declared_inventory")
    expected_keep = tuple(key[x - 2] != 0 for x in observation.symbols)
    if keep != expected_keep:
        issues.append("keep_decisions_disagree_with_global_key")
    if any(boundary and not is_kept for boundary, is_kept in zip(boundaries, keep)):
        issues.append("word_boundary_on_null")
    null_fraction = 1 - sum(expected_keep) / len(expected_keep)
    if null_fraction > max_null_fraction:
        issues.append("null_budget_exceeded")
    if not any(expected_keep):
        issues.append("empty_explanation")
    for anchor in observation.anchors:
        if key[anchor.observed - 2] != anchor.canonical:
            issues.append("independent_anchor_violated")
    decoded = decode(observation.symbols, key)
    reconstructed = reencode(decoded.plaintext, key, decoded.alignment, symbols=observation.symbols)
    exact = tuple(reconstructed) == observation.symbols
    if not exact:
        issues.append("forward_reconstruction_failed")
    # Optional executor grammar is a separately specified, fully inspectable family.
    semantic_report = {"status": "not_checked", "limitation": "No independent world evidence supplied"}
    if parse_semantics:
        from . import worlds

        parser = worlds.parse_document
        if parse_semantics:
            compact_boundaries = tuple(b for b, k in zip(boundaries, expected_keep) if k)
            try:
                parsed = parser(
                    decoded.plaintext,
                    compact_boundaries,
                    entities=4,
                    grammar=hypothesis["grammar"],
                    morphology=hypothesis["morphology"],
                    family=hypothesis["family"],
                )
                semantic_report = {"status": "syntactic_parse", "parsed": [asdict(c) for c in parsed]}
                if hypothesis["family"] == "procedure":
                    witness = worlds.infer_procedure_trace(parsed, 4)
                    semantic_report.update(status="executable_procedure", witness=asdict(witness))
                elif hypothesis["family"] == "taxonomy":
                    graph = worlds.infer_taxonomy_graph(parsed, 4)
                    semantic_report.update(status="compatible_taxonomy", witness=[asdict(r) for r in graph])
                else:
                    semantic_report["status"] = "nonsemantic_surface_only"
            except ValueError as error:
                issues.append("declared_grammar_rejected")
                semantic_report = {"status": "rejected", "reason": str(error)}
    return {
        "valid": not issues,
        "issues": sorted(set(issues)),
        "exact_reencoding": exact,
        "null_fraction": null_fraction,
        "plaintext": list(decoded.plaintext),
        "alignment": list(decoded.alignment),
        "semantics": semantic_report,
        "description_units": sum(k != 0 for k in key) + sum(boundaries),
        "unresolved_key_symbols": list(hypothesis.get("unresolved_key_symbols", ())),
        "claim": "candidate executable explanation only; not a translation or decipherment",
    }


def project_consistency(tokens, observation, layout):
    """Project redundant local keep fields onto the sampled document-global channel.

    Changes are logged; this does not claim to sample an exact posterior.
    """
    values = tokens.detach().clone()
    parts = layout.unpack(values, observation)
    changes = []
    for i, symbol in enumerate(observation.symbols):
        keep = parts["inverse_key"][symbol - 2] != 0
        position = layout.keep_slice.start + i
        if int(values[position]) != 2 + int(keep):
            changes.append({"position": position, "before": int(values[position]), "after": 2 + int(keep)})
            values[position] = 2 + int(keep)
        position = layout.boundary_slice.start + i
        if not keep and int(values[position]) != 2:
            changes.append({"position": position, "before": int(values[position]), "after": 2})
            values[position] = 2
    retained = [i for i, x in enumerate(observation.symbols) if parts["inverse_key"][x - 2] != 0]
    if retained:
        position = layout.boundary_slice.start + retained[0]
        if int(values[position]) != 3:
            changes.append({"position": position, "before": int(values[position]), "after": 3})
            values[position] = 3
    return values, changes


@torch.no_grad()
def infer(
    model,
    observation,
    layout,
    *,
    candidates=8,
    steps=16,
    seed=0,
    max_null_fraction=0.7,
    parse_semantics=True,
    compiler_budget=20000,
    compiler_beam=32,
    allow_test=False,
):
    """Propose explanations; final holdouts require an explicit frozen-evaluation override."""
    from .denoiser import sample
    from .search import compiler_search

    if type(allow_test) is not bool:
        raise ValueError("allow_test must be a boolean")
    if observation.split == "test" and not allow_test:
        raise ValueError("Final holdout inference requires allow_test=True and a separately frozen evaluation")
    if type(candidates) is not int or not 1 <= candidates <= 1024:
        raise ValueError("Candidate budget must be1..1024")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if type(compiler_budget) is not int or not 0 <= compiler_budget <= 1_000_000:
        raise ValueError("Compiler budget must be0..1000000")
    device = next(model.parameters()).device
    inputs = layout.tensors([observation] * candidates, device)
    generator = torch.Generator().manual_seed(seed)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        proposals, trace = sample(
            model,
            inputs["condition"],
            length=layout.latent_length,
            steps=steps,
            generator=generator,
            latent_types=inputs["latent_types"],
            allowed=inputs["allowed"],
            fixed=inputs["fixed"],
            return_trace=True,
        )
        projected, repairs, origins = [], [], []
        for row in proposals:
            value, change = project_consistency(row, observation, layout)
            projected.append(value)
            repairs.append(change)
            origins.append("denoising")
        # All candidates share this condition; this is an explicit proposal prior,
        # not a normalized probability of an entire explanation.
        fixed = inputs["fixed"][:1]
        masks = torch.where(fixed >= 0, fixed, torch.ones_like(fixed))
        logits = model(
            masks,
            inputs["condition"][:1],
            torch.ones(1, device=device),
            latent_types=inputs["latent_types"][:1],
        )
        logits = logits.masked_fill(~inputs["allowed"][:1], -torch.inf)
        prior = torch.log_softmax(logits, -1)
        compiler_report = {"enabled": False, "expansions": 0, "max_expansions": compiler_budget}
        if compiler_budget and parse_semantics:
            globals_ = {
                name: prior[0, layout.global_start + i, 2 : 2 + len(values)].cpu().tolist()
                for i, (name, values) in enumerate(
                    (("grammar", GRAMMARS), ("morphology", MORPHOLOGIES), ("family", FAMILIES))
                )
            }
            searched = compiler_search(
                observation,
                key_log_probs=prior[0, layout.key_slice, 2:].cpu().tolist(),
                global_log_probs=globals_,
                beam_size=compiler_beam,
                max_expansions=compiler_budget,
                max_candidates=16,
                max_null_fraction=max_null_fraction,
            )
            compiler_report = {"enabled": True, **{k: v for k, v in searched.items() if k != "candidates"}}
            for candidate in searched["candidates"]:
                h = candidate["hypothesis"]
                value = layout.pack(
                    h["inverse_key"],
                    h["keep"],
                    h["boundaries"],
                    grammar=h["grammar"],
                    morphology=h["morphology"],
                    family=h["family"],
                ).to(device)
                value.masked_fill_(fixed[0].eq(0), 0)
                layout.unpack(value, observation)  # Enforce fixed anchors on this branch too.
                projected.append(value)
                repairs.append([])
                origins.append("compiler_guided")
        projected = torch.stack(projected)
        log_probs = prior.expand(len(projected), -1, -1).gather(-1, projected.unsqueeze(-1)).squeeze(-1)
        mutable = (fixed < 0).expand(len(projected), -1)
        scores = (log_probs.masked_fill(~mutable, 0).sum(-1) / mutable.sum(-1).clamp_min(1)).cpu().tolist()
        records, seen = [], set()
        for i, row in enumerate(projected):
            parts = layout.unpack(row, observation)
            fingerprint = object_digest(parts)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            report = verify_candidate(
                observation, parts, max_null_fraction=max_null_fraction, parse_semantics=parse_semantics
            )
            records.append(
                {
                    "id": fingerprint[:16],
                    "hypothesis": parts,
                    "verification": report,
                    "proposal_score": scores[i],
                    "projection_changes": repairs[i],
                    "source": origins[i],
                }
            )
        records.sort(key=lambda r: (not r["verification"]["valid"], -r["proposal_score"], r["id"]))
        return {
            "schema_version": 1,
            "observation_digest": observation.digest,
            "seed": seed,
            "requested_candidates": candidates,
            "denoising_steps": steps,
            "candidates": records,
            "valid_candidates": sum(r["verification"]["valid"] for r in records),
            "trace_steps": len(trace),
            "compiler_search": compiler_report,
            "score_kind": "factorized masked proposal score, not posterior probability",
            "status": "candidate_hypotheses"
            if any(r["verification"]["valid"] for r in records)
            else "abstain",
            "claim": "No historical meaning established",
        }
    finally:
        for module, training in modes:
            module.training = training


def episode_identity(episode):
    return {
        "observation_sha256": episode.observation.digest,
        "target_sha256": object_digest(episode.clean.tolist()),
        "world_config": asdict(episode.world.config),
    }
