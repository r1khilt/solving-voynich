"""Score frozen DINO evaluation features with audited DINO development settings."""

import argparse
import json
from pathlib import Path

from scripts.herbal_control_0003_score_eval import score


DINO_METHOD = "DINOv2-with-registers-base final CLS 224 direct resize L2 cosine"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("development_selection", type=Path)
    parser.add_argument("development_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = score(
        args.features, args.development_selection, args.development_audit,
        feature_method=DINO_METHOD,
        selection_id="HERBAL-CONTROL-0003-DINO-development-selection",
        audit_id="HERBAL-CONTROL-0003-DINO-independent-development-score-audit",
        result_id="HERBAL-CONTROL-0003-DINO-evaluation-score")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"],
                      "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
