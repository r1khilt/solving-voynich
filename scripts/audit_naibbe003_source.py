"""Independent scalar audit of source-only NAIBBE-003 language models.

No production model, data builder, cipher decoder, or evaluator is imported.
Counts are ordinary string Counters and probabilities are evaluated recursively.
The models use raw counts at every order: absolute discounting here is NOT
Kneser--Ney, which would require lower-order continuation counts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np


FAMILIES = {"dirichlet", "absolute_discount", "baseline"}
BASE_PSEUDOCOUNT = .1
BASELINE_WEIGHTS = (.02, .08, .20, .70)
TAUS = (.25, 1., 4., 16., 64., 256.)
DISCOUNTS = (.25, .5, .75, .9)
ROOT = Path(__file__).resolve().parent.parent
ALPHABET = "abcdefghilmnopqrstuvxyz"
RANGES = {"train": [0, 220000], "dev": [220000, 268000], "check": [268000, 317326]}
SOURCE_PATHS = {"scripts/select_naibbe003_source.py", "src/voynich/recursive_character_lm.py",
                "src/voynich/naibbe_key_search.py", "docs/experiments/NAIBBE-003.md"}
CONFIGS = ([{"family": "legacy", "parameter": None}]
           + [{"family": "dirichlet", "parameter": value} for value in TAUS]
           + [{"family": "absolute_discount", "parameter": value} for value in DISCOUNTS])


class ScalarSourceLM:
    """Normalized character four-gram source model from independent counts.

    ``parameter`` is shared across orders two through four: concentration tau
    for Dirichlet interpolation, or discount d for absolute discounting.
    Both use add-.1 unigram probabilities on the complete supplied alphabet.
    The frozen baseline is retained solely for source-only comparison.
    """

    def __init__(self, text: str, alphabet: str, family: str, parameter: float | None = None):
        if not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError("A nonempty, unique character alphabet is required")
        if not text or not set(text) <= set(alphabet):
            raise ValueError("Nonempty training text must use only the supplied alphabet")
        if family not in FAMILIES:
            raise ValueError("Unknown source-model family")
        if family == "baseline":
            if parameter is not None:
                raise ValueError("Baseline has no tunable parameter")
        elif not isinstance(parameter, (int, float)) or not math.isfinite(parameter):
            raise ValueError("A finite smoothing parameter is required")
        elif family == "dirichlet" and parameter <= 0:
            raise ValueError("Dirichlet concentration must be positive")
        elif family == "absolute_discount" and not 0 < parameter < 1:
            raise ValueError("Absolute discount must be strictly between zero and one")
        self.alphabet = alphabet
        self._alphabet_set = set(alphabet)
        self.family = family
        self.parameter = parameter
        self.counts: dict[int, Counter[str]] = {}
        self.context_totals: dict[int, Counter[str]] = {}
        self.continuation_types: dict[int, Counter[str]] = {}
        self._cache: dict[tuple[str, str], float] = {}
        for order in range(1, 5):
            counts = Counter(text[index:index + order] for index in range(len(text) - order + 1))
            totals: Counter[str] = Counter()
            types: Counter[str] = Counter()
            for gram, count in counts.items():
                totals[gram[:-1]] += count
                types[gram[:-1]] += 1
            self.counts[order] = counts
            self.context_totals[order] = totals
            self.continuation_types[order] = types

    def probability(self, history: str, character: str) -> float:
        if len(character) != 1 or character not in self.alphabet:
            raise ValueError("Prediction requires a character in the supplied alphabet")
        if not set(history) <= self._alphabet_set:
            raise ValueError("History contains an out-of-alphabet character")
        history = history[-3:]
        index = history, character
        if index in self._cache:
            return self._cache[index]
        order = len(history) + 1
        size = len(self.alphabet)
        if self.family == "baseline":
            gram = history + character
            weighted = []
            for width in range(1, order + 1):
                suffix = gram[-width:]
                numerator = self.counts[width][suffix] + BASE_PSEUDOCOUNT
                denominator = self.context_totals[width][suffix[:-1]] + BASE_PSEUDOCOUNT * size
                weighted.append(BASELINE_WEIGHTS[width - 1] * numerator / denominator)
            probability = math.fsum(weighted) / math.fsum(BASELINE_WEIGHTS[:order])
        elif not history:
            probability = ((self.counts[1][character] + BASE_PSEUDOCOUNT)
                           / (self.context_totals[1][""] + BASE_PSEUDOCOUNT * size))
        else:
            lower = self.probability(history[1:], character)
            total = self.context_totals[order][history]
            count = self.counts[order][history + character]
            if total == 0:
                probability = lower
            elif self.family == "dirichlet":
                probability = (count + self.parameter * lower) / (total + self.parameter)
            else:
                removed_mass = self.parameter * self.continuation_types[order][history]
                probability = (max(count - self.parameter, 0.) + removed_mass * lower) / total
        self._cache[index] = probability
        return probability

    def distribution(self, history: str) -> dict[str, float]:
        return {character: self.probability(history, character) for character in self.alphabet}

    def score(self, text: str) -> float:
        """Log probability with fresh empty context on EVERY separate call."""
        return math.fsum(self.log_terms(text))

    def log_terms(self, text: str) -> list[float]:
        if not set(text) <= self._alphabet_set:
            raise ValueError("Evaluation text contains out-of-alphabet characters")
        return [math.log(self.probability(text[max(0, i - 3):i], character))
                for i, character in enumerate(text)]

    def nll_per_character(self, text: str) -> float:
        if not text:
            raise ValueError("Cannot average an empty evaluation sequence")
        return -self.score(text) / len(text)

    def normalization_audit(self) -> dict:
        maximum_error, minimum = 0., 1.
        histories = 0
        unseen = 0
        for length in range(4):
            for letters in itertools.product(self.alphabet, repeat=length):
                history = "".join(letters)
                row = self.distribution(history)
                if any(not math.isfinite(value) or value <= 0 for value in row.values()):
                    raise AssertionError("Source distribution is not strictly positive and finite")
                maximum_error = max(maximum_error, abs(math.fsum(row.values()) - 1.))
                minimum = min(minimum, min(row.values()))
                histories += 1
                if history and self.context_totals[length + 1][history] == 0:
                    unseen += 1
                    if self.family != "baseline" and row != self.distribution(history[1:]):
                        raise AssertionError("An unseen history did not exactly back off")
        if maximum_error > 1e-12:
            raise AssertionError("Source probability rows do not normalize")
        return {"histories": histories, "unseen_histories": unseen,
                "max_normalization_error": maximum_error, "minimum_probability": minimum}


def bootstrap_gain(gains: list[float]) -> tuple[list[dict], list[float]]:
    """Independent scalar sums; same registered random draws and quantile rule.

    Resample blocks uniformly, then divide the selected sums by their selected
    character counts. The incomplete final block is never padded or dropped.
    """
    if not gains or any(not math.isfinite(value) for value in gains):
        raise ValueError("Bootstrap needs finite nonempty character gains")
    blocks = []
    for start in range(0, len(gains), 1000):
        part = gains[start:start + 1000]
        blocks.append({"sum": math.fsum(part), "count": len(part)})
    sampled = np.random.default_rng(920301).integers(0, len(blocks), size=(2000, len(blocks)))
    values = []
    for indices in sampled:
        total = math.fsum(blocks[int(index)]["sum"] for index in indices)
        count = sum(blocks[int(index)]["count"] for index in indices)
        values.append(total / count)
    return blocks, np.quantile(values, [.025, .975], method="linear").tolist()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_number(saved, expected: float, label: str, tolerance: float = 1e-10) -> float:
    require(isinstance(saved, (int, float)) and math.isfinite(saved), f"Invalid number: {label}")
    difference = abs(saved - expected)
    require(difference <= tolerance, f"Independent source metric mismatch: {label}")
    return difference


def committed(path: Path, revision: str) -> None:
    relative = str(path.relative_to(ROOT))
    frozen = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
    require(frozen == path.read_bytes(), f"File differs from source freeze: {relative}")


def make_model(text: str, config: dict) -> ScalarSourceLM:
    family = "baseline" if config["family"] == "legacy" else config["family"]
    return ScalarSourceLM(text, ALPHABET, family, config["parameter"])


def audit_source_selection() -> dict:
    started = time.monotonic()
    path = ROOT / "results/NAIBBE-003/source_selection.json"
    require(path.is_file(), "Complete source selection before this audit")
    report = json.loads(path.read_text())
    require(report["experiment"] == "NAIBBE-003" and report["stage"] == "source_only_selection",
            "Unexpected source selection report")
    require(report["alphabet"] == ALPHABET and report["ranges"] == RANGES,
            "Source alphabet or split specification changed")
    require(set(report["languages"]) == {"latin", "english"}, "Unexpected source languages")
    revision = subprocess.check_output(["git", "rev-parse", report["source_commit"]],
                                       cwd=ROOT, text=True).strip()
    require(revision == report["source_commit"] and len(revision) == 40,
            "Use a full source-freeze commit identifier")
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=ROOT, check=True)
    require(set(report["source_hashes"]) == SOURCE_PATHS, "Unexpected source-code hash inventory")
    checked = {}
    for name, expected in report["source_hashes"].items():
        source_path = ROOT / name
        require(digest(source_path) == expected, f"Source-code hash mismatch: {name}")
        committed(source_path, revision)
        checked[name] = expected
    for name in ("scripts/audit_naibbe003_source.py", "tests/test_naibbe003_source_audit.py"):
        committed(ROOT / name, revision)
        checked[name] = digest(ROOT / name)
    manifest_path = ROOT / "data/manifests/naibbe003_data.json"
    require(digest(manifest_path) == report["manifest_sha256"], "Source manifest hash mismatch")
    committed(manifest_path, revision)
    checked[str(manifest_path.relative_to(ROOT))] = report["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    require(manifest["experiment"] == "NAIBBE-003", "Wrong source manifest experiment")

    # Deliberately follow only the two source-LM paths, never split inputs,
    # answer keys, Pliny source data, cipher predictions, or cipher results.
    expected_paths = {f"data/processed/naibbe001/{language}_lm.txt" for language in ("latin", "english")}
    require({source["path"] for source in manifest["lms"].values()} == expected_paths,
            "Auditor permits only the registered non-cipher source corpus paths")
    maximum_delta = 0.
    results = {}
    total_histories = 0
    for language in ("latin", "english"):
        saved = report["languages"][language]
        source = manifest["lms"][language]
        require(saved["source"] == source, "Saved source metadata mismatch")
        corpus_path = ROOT / source["path"]
        require(digest(corpus_path) == source["sha256"], "Source corpus hash mismatch")
        checked[source["path"]] = source["sha256"]
        text = corpus_path.read_text().strip()
        require(len(text) == source["characters"] == 317326, "Source corpus length mismatch")
        train = text[0:220000]
        dev = text[220000:268000]
        check = text[268000:317326]
        require([row["config"] for row in saved["candidates"]] == CONFIGS,
                "Source candidate inventory or order changed")
        replayed = []
        normalization = []
        for index, config in enumerate(CONFIGS):
            model = make_model(train, config)
            bits = model.nll_per_character(dev) / math.log(2.)
            maximum_delta = max(maximum_delta, assert_number(
                saved["candidates"][index]["dev_bits_per_character"], bits, f"{language}/dev/{index}"))
            replayed.append(bits)
            normalized = model.normalization_audit()
            normalization.append(normalized)
            total_histories += normalized["histories"]
        winner = min(range(len(replayed)), key=lambda index: (replayed[index], index))
        require(saved["selected_index"] == winner and saved["selected_config"] == CONFIGS[winner],
                "Source-only winner or stable tie rule mismatch")

        # The check block is scored for only the selected model and baseline.
        base = make_model(train, CONFIGS[0]).log_terms(check)
        selected = make_model(train, CONFIGS[winner]).log_terms(check)
        baseline_bits = -math.fsum(base) / len(base) / math.log(2.)
        selected_bits = -math.fsum(selected) / len(selected) / math.log(2.)
        gains = [(chosen - old) / math.log(2.) for chosen, old in zip(selected, base, strict=True)]
        gain = math.fsum(gains) / len(gains)
        blocks, interval = bootstrap_gain(gains)
        saved_check = saved["check"]
        require(saved_check["characters"] == len(check) == 49326, "Source check length mismatch")
        for name, expected in (("legacy_bits_per_character", baseline_bits),
                               ("selected_bits_per_character", selected_bits),
                               ("gain_bits_per_character", gain)):
            maximum_delta = max(maximum_delta, assert_number(saved_check[name], expected, f"{language}/{name}"))
        require(len(saved_check["gain_blocks"]) == len(blocks) == 50, "Bootstrap block count mismatch")
        for index, (block, expected) in enumerate(zip(saved_check["gain_blocks"], blocks, strict=True)):
            require(block["count"] == expected["count"], "Bootstrap block character count mismatch")
            maximum_delta = max(maximum_delta, assert_number(block["sum"], expected["sum"],
                                                             f"{language}/block/{index}", 1e-8))
        require(blocks[-1]["count"] == 326, "Final partial block was omitted or padded")
        require(len(saved_check["gain_ci95"]) == 2, "Invalid bootstrap interval")
        for index, expected in enumerate(interval):
            maximum_delta = max(maximum_delta, assert_number(saved_check["gain_ci95"][index], expected,
                                                             f"{language}/quantile/{index}"))
        criteria = {"new_family_selected": CONFIGS[winner]["family"] != "legacy",
                    "check_gain_at_least_005_bits": gain >= .005,
                    "block_bootstrap_lower_positive": interval[0] > 0}
        decision = "PASS" if all(criteria.values()) else "FAIL"
        require(saved["criteria"] == criteria and saved["decision"] == decision,
                "Source gate or decision mismatch")
        results[language] = {"selected_index": winner, "selected_config": CONFIGS[winner],
                             "dev_bits_per_character": replayed,
                             "baseline_check_bits_per_character": baseline_bits,
                             "selected_check_bits_per_character": selected_bits,
                             "gain_bits_per_character": gain, "gain_ci95": interval,
                             "gain_blocks": blocks, "normalization": normalization,
                             "criteria": criteria, "decision": decision}
    allowed = results["latin"]["decision"] == "PASS"
    require(report["cipher_phase_allowed"] == allowed, "Cipher-phase source gate mismatch")
    return {"experiment": "NAIBBE-003", "stage": "independent_source_only_audit", "audit_pass": True,
            "source_commit": revision, "source_selection_sha256": digest(path),
            "auditor_sha256": digest(Path(__file__).resolve()), "checked_hashes": checked,
            "languages": results, "cipher_phase_allowed": allowed,
            "candidate_models_replayed": 2 * len(CONFIGS), "normalized_histories": total_histories,
            "maximum_numeric_delta": maximum_delta, "cipher_files_read": False,
            "seconds": time.monotonic() - started,
            "scope": ["Independent raw string counts and scalar conditional probabilities",
                      "All eleven source-development candidates in each language and fixed winner selection",
                      "Every conditional row positive and normalized; recursive families back off exactly on unseen histories",
                      "Only winner and baseline source-check scores, paired character gains and weighted block bootstrap",
                      "Source-input hashes and local Git source/manifest/auditor consistency"],
            "not_established": ["No cipher evaluation, plaintext recovery, or historical mechanism audited here",
                                "Within-work source blocks do not establish cross-author generalization",
                                "Block bootstrap does not constitute independent-document uncertainty",
                                "Local Git checks do not prove publication timing or process isolation"]}


def self_test() -> dict:
    normalized_histories = 0
    for family, parameter in [("baseline", None), *(('dirichlet', tau) for tau in TAUS),
                              *(('absolute_discount', discount) for discount in DISCOUNTS)]:
        model = ScalarSourceLM("ababa", "abc", family, parameter)
        normalized_histories += model.normalization_audit()["histories"]
        if family != "baseline":
            for letter in model.alphabet:
                if model.probability("cc", letter) != model.probability("", letter):
                    raise AssertionError("Toy unseen history backoff mismatch")
        if model.score("") != 0.:
            raise AssertionError("Empty-string score mismatch")
    direct = ScalarSourceLM("ababa", "abc", "dirichlet", 4.)
    discounted = ScalarSourceLM("ababa", "abc", "absolute_discount", .5)
    base_b, base_c = 2.1 / 5.3, .1 / 5.3
    if not math.isclose(direct.probability("a", "c"), 4. * base_c / 6., abs_tol=1e-15):
        raise AssertionError("Toy Dirichlet formula mismatch")
    if not math.isclose(discounted.probability("a", "b"), 1.5 / 2. + .5 / 2. * base_b, abs_tol=1e-15):
        raise AssertionError("Toy absolute-discount formula mismatch")
    return {"self_test_pass": True, "normalized_toy_histories": normalized_histories,
            "models": 1 + len(TAUS) + len(DISCOUNTS), "source_files_read": False,
            "cipher_files_read": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--after-selection", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    output = ROOT / "results/NAIBBE-003/source_audit.json"
    require(not output.exists(), "Preserve the existing source audit")
    result = audit_source_selection()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("audit_pass", "candidate_models_replayed", "normalized_histories",
                       "maximum_numeric_delta", "cipher_phase_allowed", "cipher_files_read", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
