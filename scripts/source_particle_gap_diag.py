"""Post-outcome registered known-reading mass and finite-bank gap diagnostic."""
import argparse
import gzip
import json
import math
import signal
import time

from scripts.run_source_particle_systems001 import (
    OUT as PARENT, PATHS as PARENT_PATHS, artifact, convert_finite, require_frozen,
    resource_report, save_new,
)
from scripts.run_latin_source_model001 import ROOT, limit_resources
from voynich.compact_suffix_adapter import CompactSuffixAdapter
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.joint_key_proposal import unit_pool
from voynich.known_plaintext_keys import solve_known_plaintext
from voynich.source_prefix_inverse import logadd, logsum

EXP = "SOURCE-PARTICLE-SYSTEMS-001-DIAG-A"
OUT, BULK = ROOT/"results"/EXP, ROOT/"outputs"/EXP
SOURCE = "results/LATIN-SOURCE-COMPACT-001/large.json"
PATHS = [*PARENT_PATHS, "src/voynich/known_plaintext_keys.py",
    "scripts/source_particle_gap_diag.py", "tests/test_source_particle_gap_diag.py",
    "docs/experiments/SOURCE-PARTICLE-SYSTEMS-001-DIAG-A.md",
    "results/SOURCE-PARTICLE-SYSTEMS-001/result.json", "results/SOURCE-PARTICLE-SYSTEMS-001/audit.json"]


def source_log_mass(texts, source, rho):
    terms = []
    for text in texts:
        terms.extend((math.log(rho), len(text)*math.log1p(-rho)))
        state = 0
        for row in text:
            terms.append(math.log(float(source.row(state)[row])))
            state = source.step(state, row)
    return math.fsum(terms)


def reading_mass(texts, cipher, source, *, glyphs=6, rho=1/225):
    texts = tuple(map(tuple, texts))
    solved = solve_known_plaintext(texts, cipher, rows=len(source.alphabet), glyphs=glyphs,
        node_cap=1_000_000, solution_cap=4096)
    log_prior = source_log_mass(texts, source, rho)
    single = log_prior-len(solved["used_rows"])*math.log(glyphs+glyphs**2)
    count = solved["solution_count_lower_bound"]
    marginal = single+math.log(count) if count else -math.inf
    return {"source_log_mass": log_prior, "one_used_key_log_mass": single,
        "reading_log_mass": marginal, "reading_mass_is_lower_bound": not solved["complete"],
        "solver": solved}


def joint_bank_coverage(log_gold, log_bank, disjoint):
    """Bound only source/used-key joint support, NOT the reading marginal."""
    if not disjoint:
        return {"joint_bank_log_coverage_upper": 0., "joint_tv_lower": 0.}
    denominator = logadd(log_gold, log_bank)
    return {"joint_bank_log_coverage_upper": log_bank-denominator,
        "joint_tv_lower": math.exp(log_gold-denominator)}


def run(freeze):
    require_frozen(freeze, PATHS)
    save_new(OUT/"started.json", {"freeze": freeze, "start_unix": time.time(), "exploratory_post_outcome": True})
    limit_resources(600, 500)
    wall, cpu = time.monotonic(), time.process_time()
    completed_counts = 0
    try:
        parent = json.loads((PARENT/"result.json").read_text())
        audit = json.loads((PARENT/"audit.json").read_text())
        if audit["result"] != artifact(PARENT/"result.json"):
            raise ValueError("Audited parent result changed")
        spec = parent["fixtures"]
        if artifact(ROOT/spec["path"]) != spec:
            raise ValueError("Parent fixture binding changed")
        fixtures = json.loads((ROOT/spec["path"]).read_text())["fixtures"]
        selected = json.loads((ROOT/SOURCE).read_text())
        if artifact(ROOT/selected["counts"]["path"]) != selected["counts"] or selected["counts"] != parent["source"]:
            raise ValueError("Unchanged statistical source required")
        source = CompactSuffixAdapter(CompactSuffixSource.load(ROOT/selected["counts"]["path"]),
            selected["selected"]["tau"], cache_size=2048)
        gold = []
        pool = unit_pool(6)
        for fixture in fixtures:
            if tuple(tuple(g for r in text for g in pool[fixture["generation_key"][r]])
                    for text in fixture["generation_source"]) != tuple(map(tuple, fixture["cipher"])):
                raise ValueError("Generating source/key must re-encode literal ciphertext")
            value = reading_mass(fixture["generation_source"], fixture["cipher"], source)
            completed_counts += 1
            used = value["solver"]["used_rows"]
            target = tuple(pool[fixture["generation_key"][r]] if r in used else None for r in range(23))
            if value["solver"]["complete"] and target not in value["solver"]["solutions"]:
                raise ValueError("Known generator's used key missed by complete source oracle")
            gold.append(value)
        comparisons, solver_trace = [], []
        for spec in parent["workloads"]:
            if artifact(ROOT/spec["path"]) != spec:
                raise ValueError("Compact population metadata changed")
            row = json.loads((ROOT/spec["path"]).read_text())
            prediction = row["prediction"]
            if artifact(ROOT/prediction["path"]) != prediction:
                raise ValueError("Parent prediction bank changed")
            bank = json.loads(gzip.decompress((ROOT/prediction["path"]).read_bytes()))
            fixture, g = fixtures[row["case"]], gold[row["case"]]
            mode = reading_mass(bank["decision"]["decision_records"], fixture["cipher"], source)
            completed_counts += 1
            best = max(bank["bank"], key=lambda b: b["literal_log_mass"])
            scalar = source_log_mass(best["source_records"], source, 1/225)
            scalar -= sum(k >= 0 for k in best["used_key"])*math.log(42)
            if abs(scalar-best["literal_log_mass"]) > 1e-9:
                raise ValueError("Independent lazy-source best-leaf score differs from dense score")
            used = set(g["solver"]["used_rows"])
            expected_key = tuple(fixture["generation_key"][r] if r in used else -1 for r in range(23))
            gold_present = any(b["source_records"] == fixture["generation_source"]
                and tuple(b["used_key"]) == expected_key for b in bank["bank"])
            log_bank = logsum(b["literal_log_mass"] for b in bank["bank"])
            if abs(log_bank-bank["decision"]["visited_literal_bank_log_mass"]) > 1e-9:
                raise ValueError("Unique literal bank mass differs")
            comparison = {"name": row["name"], "case": row["case"],
                "gold_used_key_leaf_in_bank": gold_present,
                "gold_reading_in_bank": any(b["source_records"] == fixture["generation_source"] for b in bank["bank"]),
                "gold_leaf_minus_best_visited_leaf_nats": g["one_used_key_log_mass"]-scalar,
                "gold_leaf_minus_log_evidence_estimate_nats": g["one_used_key_log_mass"]-row["summary"]["log_evidence_estimate"],
                "gold_leaf_minus_unique_bank_log_mass_nats": g["one_used_key_log_mass"]-log_bank,
                "mode_complete_key_count": mode["solver"]["complete"],
                "gold_reading_minus_mode_reading_nats": (g["reading_log_mass"]-mode["reading_log_mass"]
                    if g["solver"]["complete"] and mode["solver"]["complete"] else None),
                **joint_bank_coverage(g["one_used_key_log_mass"], log_bank, not gold_present)}
            comparisons.append(comparison)
            solver_trace.append({"name": row["name"], "mode": mode})
            if resource_report(wall, cpu)["peak_rss_bytes"] > 2*1024**3:
                raise MemoryError("2GiB sampled diagnostic host guard")
        if len(fixtures) != 4 or len(comparisons) != 32:
            raise ValueError("Fixed4gold/32mode diagnostic grid required")
        trace = save_new(BULK/"reading-counts.json.gz", convert_finite({"gold": gold, "modes": solver_trace}), compressed=True)
        save_new(OUT/"result.json", convert_finite({"freeze": freeze, "parent": artifact(PARENT/"result.json"),
            "parent_audit": artifact(PARENT/"audit.json"), "source": selected["counts"],
            "trace": trace, "comparisons": comparisons, "resources": resource_report(wall, cpu),
            "scope": "Exploratory known-answer mass diagnostic. Joint-support bounds do not bound reading-marginal TV or identify historical truth.",
            "paid_spend_usd": 0, "interval_arithmetic_certificate": False}))
    except Exception as exc:
        signal.alarm(0)
        save_new(OUT/"failure.json", {"freeze": freeze, "error": repr(exc),
            "completed_counts": completed_counts, "resources": resource_report(wall, cpu), "no_retry": True})
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    run(parser.parse_args().freeze)
