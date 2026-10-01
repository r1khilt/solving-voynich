"""One finite exact-search/artifact replay; no new model inference."""
from __future__ import annotations

import json
import signal
import time

from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_key_observability001 import BULK, OUT, PATHS, fixed_inputs, summary
from scripts.run_latin_source_model001 import artifact, limit_resources
from voynich.known_plaintext_keys import oracle_statistics, solve_known_plaintext


def audit():
    result = json.loads((OUT/"result.json").read_text())
    require_frozen(result["freeze"], PATHS)
    save_new(OUT/"audit-started.json", {"freeze": result["freeze"], "start_unix": time.time()})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        manifest, cases = fixed_inputs()
        trace = BULK/"cases.jsonl"
        if (artifact(trace) != result["records"] or manifest["validation_source"] != result["source"]
                or manifest["validation_episodes"] != result["episodes"]):
            raise ValueError("Completed inputs/trace binding differs")
        rows = [json.loads(s) for s in trace.read_text().splitlines()]
        if len(rows) != 64 or [r["case"] for r in rows] != list(range(64)):
            raise ValueError("Fixed complete ordered grid required")
        for row, (source, cipher, truth) in zip(rows, cases, strict=True):
            solved = solve_known_plaintext(source, cipher)
            replay = json.loads(json.dumps(solved))
            compatible = tuple(truth[r] if r in solved["used_rows"] else None for r in range(23))
            if (row["solver"] != replay or row["statistics"] != oracle_statistics(solved)
                    or row["gold_used_dictionary_found"] != (compatible in solved["solutions"])
                    or row["gold_key_is_solver_input"] is not False):
                raise ValueError("Search/count/gold scoring replay differs")
            # Alternate literal forward verifier; every returned dictionary
            # must emit both complete records, without supplied boundaries.
            for key in solved["solutions"]:
                for plain, observed in zip(source, cipher, strict=True):
                    emitted = []
                    for symbol in plain:
                        emitted.extend(key[symbol])
                    if tuple(emitted) != observed:
                        raise ValueError("Returned word-equation solution is invalid")
            if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                raise MemoryError("2GiB sampled audit host cap")
        if summary(rows) != result["summary"]:
            raise ValueError("Summary arithmetic differs")
        save_new(OUT/"audit.json", {"status": "PASS_fixed_search_replay_and_literal_verification",
            "cases": len(rows), "result": artifact(OUT/"result.json"), "records": artifact(trace),
            "resources": resource_report(wall, cpu), "independent_agent_review": False,
            "independent_count_algorithm": False, "model_inference": False})
        print("64-case exact-search/literal-forward/artifact audit PASS", flush=True)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    audit()
