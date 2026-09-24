"""Generate fresh visible TEACH-0016 suite without model or checkpoints."""

import argparse
import json
from pathlib import Path

from voynich.workspace.teacher14_tasks import evaluation_suite, suite_manifest
from voynich.workspace.teacher16_tasks import generate_split, split_manifest


def prepare(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: output_dir / f"{split}.json"
             for split in ("discovery", "confirmation")}
    paths.update({str(seed): output_dir / f"teach14-{seed}.json"
                  for seed in (74111, 74117)})
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("No automatic TEACH-0016 suite overwrite")
    payloads = {split: split_manifest(generate_split(split), split)
                for split in ("discovery", "confirmation")}
    for seed, size in ((74111, 128), (74117, 2)):
        payloads[str(seed)] = suite_manifest(
            evaluation_suite(seed, size), seed=seed, group_count=size)
    for key, path in paths.items():
        path.write_text(json.dumps(payloads[key], sort_keys=True,
                                   separators=(",", ":")) + "\n")
    return {"suite_paths": {key: str(path) for key, path in paths.items()},
            "groups": {split: len(payloads[split]["groups"])
                       for split in ("discovery", "confirmation")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/TEACH-0016"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
