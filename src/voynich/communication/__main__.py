"""Run the communication-system research workbench: python -m voynich.communication."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .schema import JointLayout, Observation, save_json


def _load(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate", help="Generate separate observations and gold audit records")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=41021)
    p.add_argument("--split", choices=["train", "validation"], default="validation")
    p.add_argument("--alphabet-size", type=int, default=24)
    p.add_argument("--max-observation", type=int, default=128)
    p.add_argument("--anchors", type=int, default=0)
    p = sub.add_parser("train", help="Train joint masked inference within fixed time/update bounds")
    p.add_argument("--config", default="configs/communication-local.json")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    p = sub.add_parser("train-actions", help="Train the persistent-entity action world model")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--max-seconds", type=float, default=180)
    p.add_argument("--seed", type=int, default=41022)
    p = sub.add_parser("infer", help="Propose and verify explanations of an observation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--observation", required=True)
    p.add_argument("--action-checkpoint", help="Optional learned action-prior diagnostic; not evidence")
    p.add_argument("--compiler-budget", type=int, default=20000)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument(
        "--channel-only", action="store_true", help="Disable the declared synthetic semantic grammar"
    )
    p = sub.add_parser("evaluate", help="Evaluate joint proposals on synthetic development holdout")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--anchors", type=int, default=0)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=41021)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument(
        "--untrained", action="store_true", help="Reinitialize architecture under checkpoint training seed"
    )
    p = sub.add_parser("export-manuscript", help="Read-only train/validation format adapter; no scoring")
    p.add_argument("--data", default="data/processed/zl3b")
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "validation"], default="validation")
    p.add_argument("--window", type=int, default=128)
    p = sub.add_parser("update-belief", help="Apply a supplied independent evidence record")
    p.add_argument("--belief", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser(
        "plan-evidence", help="Rank supplied evidence queries by exact finite information gain"
    )
    p.add_argument("--belief", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("intervene", help="Run model activation interventions with matched controls")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=41021)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p = sub.add_parser("ground", help="Align independently annotated anonymous relation graphs")
    p.add_argument("--candidate", required=True)
    p.add_argument("--observed", required=True)
    p.add_argument("--text-source-group", required=True)
    p.add_argument("--used-source-group", action="append", default=[])
    p.add_argument("--output", required=True)
    p = sub.add_parser(
        "extract-transitions", help="Compile supplied labeled transitions into an explicit program"
    )
    p.add_argument("--examples", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--on-conflict", choices=["raise", "report"], default="raise")
    args = parser.parse_args(argv)
    if args.command == "generate":
        from .pipeline import make_episode

        if not 0 < args.count <= 100_000:
            parser.error("count must be1..100000")
        layout = JointLayout(args.alphabet_size, args.max_observation, max(4, args.anchors))
        destination = Path(args.output)
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Output directory must be new")
        destination.mkdir(parents=True, exist_ok=True)
        observations, gold = destination / "observations.jsonl", destination / "gold.jsonl"
        with observations.open("w") as visible, gold.open("w") as hidden:
            for i in range(args.count):
                episode = make_episode(i, args.split, layout, anchor_count=args.anchors, seed=args.seed)
                visible.write(json.dumps(episode.observation.to_dict()) + "\n")
                hidden.write(
                    json.dumps(
                        {
                            "observation_id": episode.observation.document_id,
                            "targets": episode.clean.tolist(),
                            "world": episode.world.to_dict(),
                        }
                    )
                    + "\n"
                )
        result = {
            "schema_version": 1,
            "count": args.count,
            "split": args.split,
            "seed": args.seed,
            "layout": asdict(layout),
            "oracle_anchors": args.anchors,
            "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (observations, gold)},
        }
        save_json(destination / "manifest.json", result)
    elif args.command == "train":
        from .training import train

        result = train(
            args.config, args.run_dir, device=args.device, resume=args.resume, stop_after=args.stop_after
        )
    elif args.command == "train-actions":
        from .action_training import train_actions

        result = train_actions(
            args.run_dir, device=args.device, steps=args.steps, max_seconds=args.max_seconds, seed=args.seed
        )
    elif args.command in {"infer", "evaluate", "intervene"}:
        import torch
        from ..runtime import resolve_device
        from .training import load_checkpoint

        model, layout, checkpoint = load_checkpoint(args.checkpoint, resolve_device(args.device))
        if args.command == "infer":
            from .pipeline import infer

            raw = Path(args.observation).read_text()
            if args.observation.endswith(".jsonl"):
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
                if not 0 <= args.index < len(records):
                    raise ValueError("Observation index out of range")
                observation = Observation.from_dict(records[args.index])
            else:
                if args.index != 0:
                    raise ValueError("index only applies to JSONL")
                observation = Observation.from_dict(json.loads(raw))
            if observation.split == "test":
                raise ValueError("Final holdout inference requires a separately frozen evaluation")
            result = infer(
                model,
                observation,
                layout,
                candidates=args.candidates,
                steps=args.steps,
                seed=args.seed,
                parse_semantics=not args.channel_only,
                compiler_budget=args.compiler_budget,
            )
            if args.action_checkpoint:
                from .coupling import attach_action_prior

                result = attach_action_prior(
                    result, args.action_checkpoint, str(next(model.parameters()).device)
                )
        elif args.command == "evaluate":
            from .denoiser import ConditionalDenoiser
            from .evaluation import evaluate_joint

            if args.untrained:
                torch.manual_seed(checkpoint["training_config"]["seed"])
                model = ConditionalDenoiser(model.config).to(next(model.parameters()).device)
            result = evaluate_joint(
                model,
                layout,
                seed=args.seed,
                count=args.count,
                anchor_count=args.anchors,
                candidates=args.candidates,
                steps=args.steps,
            )
            result["model_condition"] = "untrained" if args.untrained else "trained"
        else:
            from .interpretation import causal_report
            from .pipeline import make_episode

            base = make_episode(args.index, "validation", layout, seed=args.seed)
            donor = make_episode(args.index + 1, "validation", layout, seed=args.seed)
            device = next(model.parameters()).device
            base_inputs, donor_inputs = (
                layout.tensors([base.observation], device),
                layout.tensors([donor.observation], device),
            )
            noisy = torch.where(
                base_inputs["fixed"] >= 0, base_inputs["fixed"], torch.ones_like(base_inputs["fixed"])
            )
            # Shared latent/padding support: interventions differ in external condition only.
            positions = [i for i in range(layout.alphabet_size) if int(base_inputs["fixed"][0, i]) < 0]
            result = causal_report(
                model,
                noisy,
                base_inputs["condition"],
                torch.ones(1, device=device),
                donor_condition=donor_inputs["condition"],
                positions=positions,
                latent_types=base_inputs["latent_types"],
                allowed=base_inputs["allowed"],
                seed=args.seed,
            )
            result["scope"] = "Changes in learned synthetic inference; no historical causal identification"
        save_json(args.output, result)
    elif args.command == "export-manuscript":
        from .adapters import export_manuscript

        result = export_manuscript(args.data, args.output, split=args.split, window=args.window)
    elif args.command == "update-belief":
        from .evidence import BeliefState, Evidence

        state = BeliefState.from_dict(_load(args.belief))
        result = state.update(Evidence.from_dict(_load(args.evidence))).to_dict()
        save_json(args.output, result)
    elif args.command == "ground":
        from .grounding import GroundingObservation, Relation, align_relations

        candidate = _load(args.candidate)
        observed = GroundingObservation.from_dict(_load(args.observed))
        result = align_relations(
            [Relation(**r) for r in candidate["relations"]],
            observed,
            candidate_nodes=candidate.get("nodes"),
            text_source_group=args.text_source_group,
            used_source_groups=args.used_source_group,
        )
        save_json(args.output, result)
    elif args.command == "extract-transitions":
        from .interpretation import extract_transition_program, evaluate_transition_program

        program = extract_transition_program(_load(args.examples), on_conflict=args.on_conflict)
        result = {
            "program": program.to_dict(),
            "evaluation": evaluate_transition_program(program, _load(args.validation)),
            "scope": "Extraction from supplied transition labels; not automatic semantic discovery",
        }
        save_json(args.output, result)
    else:
        from .evidence import BeliefState, Query, rank_queries

        state = BeliefState.from_dict(_load(args.belief))
        result = {
            "queries": rank_queries(state, [Query(**q) for q in _load(args.queries)]),
            "policy": "Recommendation only; no evidence acquired or resources spent",
        }
        save_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
