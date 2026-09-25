"""Independent NetworkX replay of the frozen DINO evaluation score."""

import argparse
import json
from pathlib import Path

from scripts.herbal_control_0003_eval_audit import audit


DINO_METHOD = "DINOv2-with-registers-base final CLS 224 direct resize L2 cosine"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("development_selection", type=Path)
    parser.add_argument("development_audit", type=Path)
    parser.add_argument("evaluation_score", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.features, args.development_selection, args.development_audit,
        args.evaluation_score, feature_method=DINO_METHOD,
        selection_id="HERBAL-CONTROL-0003-DINO-development-selection",
        development_audit_id="HERBAL-CONTROL-0003-DINO-independent-development-score-audit",
        score_id="HERBAL-CONTROL-0003-DINO-evaluation-score",
        audit_id="HERBAL-CONTROL-0003-DINO-independent-evaluation-audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "gates": result["gates"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
