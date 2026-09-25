"""Independent NetworkX replay of the frozen DINO development selection."""

import argparse
import json
from pathlib import Path

from scripts.herbal_control_0003_score_audit import audit


DINO_METHOD = "DINOv2-with-registers-base final CLS 224 direct resize L2 cosine"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("score", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.features, args.score, feature_method=DINO_METHOD,
        score_id="HERBAL-CONTROL-0003-DINO-development-selection",
        audit_id="HERBAL-CONTROL-0003-DINO-independent-development-score-audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"],
                      "primary_method": result["primary_method"]}))


if __name__ == "__main__":
    main()
