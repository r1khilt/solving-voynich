"""Materialize the two fixed fresh TEACH-0022 graph suites once."""

import json
from pathlib import Path

from voynich.workspace.teacher14_tasks import evaluation_suite, suite_manifest


SEEDS = (84411, 84511)
GROUPS = 128


def prepare(root: Path) -> dict:
    output = root / "outputs/TEACH-0022"
    if output.exists():
        raise FileExistsError("No automatic TEACH-0022 suite redraw")
    output.mkdir(parents=True)
    files = {}
    for seed in SEEDS:
        manifest = suite_manifest(evaluation_suite(seed, GROUPS),
                                  seed=seed, group_count=GROUPS)
        path = output / f"suite-{seed}.json"
        path.write_text(json.dumps(manifest, sort_keys=True, separators=(
            ",", ":")) + "\n")
        files[str(seed)] = {"path": str(path),
                            "episodes": sum(len(panel) for panel in manifest[
                                "panels"].values())}
    return files


if __name__ == "__main__":
    print(json.dumps(prepare(Path.cwd()), indent=2, sort_keys=True))
