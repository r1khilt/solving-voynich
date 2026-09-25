"""Score folio-only evaluation with frozen folio and primary-image development gates."""

import argparse
import json
from pathlib import Path

from voynich.herbal_folio import FEATURE_METHOD
from scripts.herbal_control_0003_folio_features import verify_evaluation_feature_gate
from scripts.herbal_control_0003_score_eval import score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("folio_development_features", type=Path)
    parser.add_argument("folio_feature_audit", type=Path)
    parser.add_argument("folio_selection", type=Path)
    parser.add_argument("folio_audit", type=Path)
    parser.add_argument("image_selection", type=Path)
    parser.add_argument("image_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify_evaluation_feature_gate(
        args.features, folio_features_path=args.folio_development_features,
        folio_feature_audit_path=args.folio_feature_audit,
        folio_selection_path=args.folio_selection,
        folio_audit_path=args.folio_audit,
        image_selection_path=args.image_selection,
        image_audit_path=args.image_audit)
    result = score(
        args.features, args.folio_selection, args.folio_audit,
        feature_method=FEATURE_METHOD,
        selection_id="HERBAL-CONTROL-0003-FOLIO-development-selection",
        audit_id="HERBAL-CONTROL-0003-FOLIO-independent-development-score-audit",
        result_id="HERBAL-CONTROL-0003-FOLIO-evaluation-score",
        feasibility_gate_name="folio_open_set_feasibility")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"],
                      "macro_ba": result["methods"][result["primary_method"]]["batch_macro_ba"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
