"""Independent assignment replay of the folio-only development selection."""

import argparse
import json
from pathlib import Path

from voynich.herbal_folio import FEATURE_METHOD
from scripts.herbal_control_0003_score_audit import audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.features, args.selection, feature_method=FEATURE_METHOD,
        score_id="HERBAL-CONTROL-0003-FOLIO-development-selection",
        audit_id="HERBAL-CONTROL-0003-FOLIO-independent-development-score-audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"],
                      "primary_method": result["primary_method"]}, sort_keys=True))


if __name__ == "__main__":
    main()
