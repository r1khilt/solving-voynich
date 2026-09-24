"""Materialize the registered TEACH-0017 fresh visible suite only."""

import json
from pathlib import Path

from voynich.workspace.teacher17_tasks import generate_split, split_manifest


def prepare(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: output_dir / f"{split}.json"
             for split in ("discovery", "confirmation")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("No automatic TEACH-0017 suite redraw")
    sizes = {}
    for split, path in paths.items():
        manifest = split_manifest(generate_split(split), split)
        path.write_text(json.dumps(manifest, sort_keys=True,
                                   separators=(",", ":")) + "\n")
        sizes[split] = path.stat().st_size
    return {"suite": "TEACH-0017", "bytes": sizes}


if __name__ == "__main__":
    print(json.dumps(prepare(Path("outputs/TEACH-0017")), sort_keys=True))
