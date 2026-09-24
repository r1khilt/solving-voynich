"""Move EXP-0029's complete per-pair result to an ignored archive and emit a compact report."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/EXP-0029/results.json"
ROWS = ROOT / "results/EXP-0029/rows.json"


def loss(margin: float) -> float:
    return (max(0, -margin) + math.log1p(math.exp(-abs(margin)))) / math.log(2)


def main() -> None:
    if ROWS.exists():
        raise FileExistsError("Raw archive already exists; refusing overwrite")
    raw = RESULT.read_bytes()
    result = json.loads(raw)
    pairs = result["validation"]["pairs"]
    per_leaf = defaultdict(lambda: {"pairs": 0, "form_bits_sum": 0.0,
                                    "full_bits_sum": 0.0, "form_correct_sum": 0.0,
                                    "full_correct_sum": 0.0})
    for pair in pairs:
        item = per_leaf[pair["leaf"]]
        item["pairs"] += 1
        for name in ("form", "full"):
            margin = pair[f"{name}_margin"]
            item[f"{name}_bits_sum"] += loss(margin)
            item[f"{name}_correct_sum"] += (margin > 0) + 0.5 * (margin == 0)
    result["validation"].pop("pairs")
    result["validation"]["per_leaf"] = dict(sorted(per_leaf.items()))
    result["raw_archive_sha256"] = hashlib.sha256(raw).hexdigest()
    ROWS.write_bytes(raw)
    RESULT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"compact": str(RESULT.relative_to(ROOT)),
                      "raw_archive": str(ROWS.relative_to(ROOT)),
                      "raw_sha256": result["raw_archive_sha256"],
                      "compact_sha256": hashlib.sha256(RESULT.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
