"""Select the folio-only challenger from development rows after the full freeze."""

import argparse
import json
from pathlib import Path

from voynich.herbal_folio import FEATURE_METHOD
from scripts.herbal_control_0003_score_dev import score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = score(args.features, feature_method=FEATURE_METHOD,
                   result_id="HERBAL-CONTROL-0003-FOLIO-development-selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"primary_method": result["primary_method"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
