"""No-generator structural/exposure audit for both TEACH-0022 suites."""

import hashlib
import json
from pathlib import Path

from scripts.teacher0014_suite_audit import audit_manifest


SEEDS = (84411, 84511)
PRIOR = ("outputs/TEACH-0016/teach14-74111.json",
         "results/TEACH-0014-v3/suite.json")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identities(manifest: dict) -> dict[str, set[str]]:
    return {name: {episode[name] for panel in manifest["panels"].values()
                   for episode in panel}
            for name in ("graph_id", "logical_id", "render_id")}


def audit(root: Path) -> dict:
    prior = {}
    for name in PRIOR:
        path = root / name
        manifest = json.loads(path.read_text())
        checked = audit_manifest(manifest)
        prior[name] = {"file_sha256": sha(path),
                       "canonical_sha256": checked["manifest_sha256"],
                       "identities": identities(manifest)}
    new = {}
    for seed in SEEDS:
        path = root / f"outputs/TEACH-0022/suite-{seed}.json"
        manifest = json.loads(path.read_text())
        checked = audit_manifest(manifest)
        if manifest["seed"] != seed or manifest["group_count"] != 128:
            raise ValueError("Fresh suite seed/group count differs")
        new[seed] = {"file_sha256": sha(path),
                     "canonical_sha256": checked["manifest_sha256"],
                     "identities": identities(manifest),
                     "episodes": sum(len(panel) for panel in manifest[
                         "panels"].values())}
    sources = {**prior, **new}
    for first_index, (first_name, first) in enumerate(sources.items()):
        for second_name, second in list(sources.items())[first_index + 1:]:
            for identity in ("graph_id", "logical_id", "render_id"):
                overlap = first["identities"][identity] & second[
                    "identities"][identity]
                if overlap:
                    raise ValueError(f"{identity} overlap: {first_name} / "
                                     f"{second_name}: {len(overlap)}")
    return {"audit": "pass", "seed_order": list(SEEDS),
            "prior": {name: {key: value for key, value in entry.items()
                             if key != "identities"}
                      for name, entry in prior.items()},
            "new": {str(seed): {key: value for key, value in entry.items()
                                if key != "identities"}
                    for seed, entry in new.items()},
            "exact_graph_logical_render_overlap": 0}


if __name__ == "__main__":
    root = Path.cwd()
    report = audit(root)
    path = root / "results/TEACH-0022/suite-audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("No automatic TEACH-0022 suite audit overwrite")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
