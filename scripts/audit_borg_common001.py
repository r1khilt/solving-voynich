"""ONE post-seal literal-score/accounting audit; no historical answer or fit.

The parser and corpus normalization are shared frozen inputs. Source recounts,
conditional arithmetic, full-string scoring, decode and control gates below
are separate implementations. Search trajectories are not independently replayed.
"""
from collections import Counter
import argparse
import hashlib
import json
import math
import resource
import signal
import time

from scripts import run_borg_common001 as run

WALL, CPU, HOST = 120, 100, 1024**3


def independent_source(text, alphabet):
    if not text or set(text) - set(alphabet):
        raise ValueError("Unsupported source")
    counts, totals = {}, {}
    for order in range(1, 5):
        counts[order] = Counter(text[i:i+order] for i in range(len(text)-order+1))
        totals[order] = Counter()
        for gram, count in counts[order].items():
            totals[order][gram[:-1]] += count

    def probability(context, letter):
        context = context[-3:]
        order = len(context)+1
        if order == 1:
            return (counts[1][letter]+.1)/(len(text)+.1*len(alphabet))
        parent = probability(context[1:], letter)
        return (counts[order][context+letter]+16*parent)/(totals[order][context]+16)

    cache = {}
    def log_probability(context, letter):
        key = context[-3:], letter
        if key not in cache:
            cache[key] = math.log(probability(*key))
        return cache[key]
    return log_probability


def literal_decode(records, key, observed=run.COMMON, alphabet=run.LATIN):
    if (type(key) is not list or len(key) != len(alphabet)
            or any(type(index) is not int for index in key)
            or set(key) != set(range(len(alphabet)))):
        raise ValueError("Invalid integer injection completion")
    translation = {symbol: alphabet[key[index]] for index, symbol in enumerate(observed)}
    return ["".join(translation[symbol] for symbol in record) for record in records]


def literal_score(records, probability):
    terms = []
    for text in records:
        if not text:
            raise ValueError("Empty literal record")
        for index, letter in enumerate(text):
            terms.append(probability(text[max(0,index-3):index], letter))
    return math.fsum(terms)


def independent_metrics(records, predictions, truth, key):
    if len(records) != len(predictions) or len(predictions) != len(truth["plain_transfer"]):
        raise ValueError("Changed transfer record count")
    errors, characters = 0, 0
    for crypt, prediction, answer in zip(records, predictions, truth["plain_transfer"], strict=True):
        if len(crypt) != len(prediction) or len(prediction) != len(answer):
            raise ValueError("Changed literal length")
        characters += len(answer)
        errors += sum(a != b for a, b in zip(prediction, answer, strict=True))
    used = {run.COMMON.index(symbol) for record in records for symbol in record}
    correct = sum(key[index] == truth["key"][index] for index in used)
    return {"errors": errors, "characters": characters, "used_rows": len(used),
            "correct_used_rows": correct,
            "pass": 100*errors <= characters and 100*correct >= 95*len(used)}


def close_number(observed, expected):
    if not (type(observed) in (int,float) and math.isfinite(observed)
            and math.isclose(observed, expected, rel_tol=0, abs_tol=1e-7)):
        raise ValueError("Independent literal score disagrees")


def expected_fit_inputs(inputs):
    answer = {name: (row["fit"], "latin", row["search_seed"])
              for name,row in inputs["controls"].items()}
    historical = [row["symbols"] for row in inputs["historical"]["fit"]]
    # Reproduce only the registered null input, not any fitted search/RNG path.
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(96767))
    shuffled = ["".join(rng.permutation(list(row))) for row in historical]
    answer.update({"borg-latin": (historical,"latin",run.HIST_SEEDS[0]),
                   "borg-english": (historical,"english",run.HIST_SEEDS[1]),
                   "borg-shuffle-latin": (shuffled,"latin",run.HIST_SEEDS[2])})
    return answer


def audit(freeze):
    started_path = run.OUT/"audit-started.json"
    result_path, seal_path = run.OUT/"result.json", run.OUT/"sealed_predictions.json"
    if not result_path.exists() or not seal_path.exists() or (run.OUT/"failure.json").exists():
        raise ValueError("Need complete original producer and no failure")
    run.save_new(started_path, {"freeze": freeze, "no_retry": True})
    wall, cpu = time.monotonic(), time.process_time()
    def timeout(*_):
        raise TimeoutError("Bounded audit time cap")
    old_alarm, old_prof = signal.getsignal(signal.SIGALRM), signal.getsignal(signal.SIGPROF)
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGPROF, timeout)
    signal.alarm(WALL)
    signal.setitimer(signal.ITIMER_PROF, CPU)
    def guard():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > HOST:
            raise MemoryError("Audit process RSS cap")
    try:
        result, seal = json.loads(result_path.read_text()), json.loads(seal_path.read_text())
        preparation = json.loads(run.PREP.read_text())
        start = json.loads((run.OUT/"started.json").read_text())
        for value in (result,seal,start):
            if value["freeze"] != freeze or value["experiment"] != run.EXP:
                raise ValueError("Changed freeze/namespace")
        if run.load_bound(result["seal"]) != seal:
            raise ValueError("Changed candidate seal")
        paths = (*run.PATHS,str(run.PREP.relative_to(run.ROOT)))
        if set(seal["source_hashes"]) != set(paths) or start["source_hashes"] != seal["source_hashes"]:
            raise ValueError("Incomplete frozen-input binding")
        import subprocess
        for path in paths:
            raw = (run.ROOT/path).read_bytes()
            if (hashlib.sha256(raw).hexdigest() != seal["source_hashes"][path]
                    or subprocess.check_output(["git","show",f"{freeze}:{path}"],cwd=run.ROOT) != raw):
                raise ValueError("Changed frozen bytes")
        if seal["preparation_sha256"] != run.digest(run.PREP.read_bytes()):
            raise ValueError("Changed input preparation")
        if not start["started_unix"] <= seal["seal_unix"]:
            raise ValueError("Seal timestamp precedes producer")
        if (seal["historical_answer_opened"] is not False or result["historical_answer_opened"] is not False
                or result["historical_recovery"] != "UNASSESSED_NO_EXTERNAL_ANSWER"):
            raise ValueError("Historical success is outside this audit")
        inputs = run.load_bound(preparation["inputs"], compressed=True)
        gold = run.load_bound(preparation["control_gold"], compressed=True)
        transfers = run.load_bound(result["private_transfers"], compressed=True)
        regenerated, regenerated_gold, metadata = run.prepare_payload()
        if (inputs != regenerated or gold != regenerated_gold
                or any(preparation[key] != value for key,value in metadata.items())):
            raise ValueError("Preparation/selection does not replay")
        # The regeneration shares the frozen offset parser/selector/generator.
        # Validate every retained raw piece directly and never bridge exclusions.
        blob = (run.ROOT/run.RAW).read_bytes()
        retained_pieces = 0
        for rows in inputs["historical"].values():
            for row in rows:
                decoded_pieces = []
                previous = None
                for piece in row["pieces"]:
                    begin,end = piece["start_byte"],piece["end_byte"]
                    if not 0 <= begin < end <= len(blob):
                        raise ValueError("Bad offset")
                    raw = blob[begin:end]
                    if run.digest(raw) != piece["raw_sha256"] or raw.decode() != piece["symbols"]:
                        raise ValueError("Retained raw piece changed")
                    if set(piece["symbols"]) - set(run.COMMON):
                        raise ValueError("Unsupported retained name")
                    if previous is not None:
                        gap = blob[previous:begin]
                        if not gap or not gap.decode().isspace() or b"\n" in gap or b"\r" in gap:
                            raise ValueError("Bridged an exclusion/line boundary")
                    previous = end
                    decoded_pieces.append(piece["symbols"])
                    retained_pieces += 1
                if "".join(decoded_pieces) != row["symbols"]:
                    raise ValueError("Changed record reconstruction")
        expected = expected_fit_inputs(inputs)
        if set(expected) != set(seal["fits"]) or len(expected) != 7 or len(gold) != 2:
            raise ValueError("Incomplete allocation")
        sources = run.source_inputs()
        if {name:row["entry"] for name,row in sources.items()} != preparation["source_files"]:
            raise ValueError("Changed source identity")
        models = {language:independent_source(row["text"],run.LATIN) for language,row in sources.items()}
        endpoint_checks, deltas = 0, []
        for name,(records,language,seed) in expected.items():
            guard()
            fit = seal["fits"][name]
            if (fit["language"] != language or fit["seed"] != seed or fit["restarts"] != run.RESTARTS
                    or fit["max_sweeps"] != run.SWEEPS or len(fit["endpoints"]) != run.RESTARTS
                    or fit["fit_records"] != len(records) or fit["fit_characters"] != sum(map(len,records))
                    or fit["input_sha256"] != run.digest(json.dumps(records).encode())):
                raise ValueError("Wrong fit allocation")
            for index,endpoint in enumerate(fit["endpoints"]):
                if endpoint["restart"] != index or not 1 <= endpoint["sweeps"] <= run.SWEEPS:
                    raise ValueError("Wrong restart inventory")
                actual = literal_score(literal_decode(records,endpoint["key"]),models[language])
                close_number(endpoint["score"],actual)
                deltas.append(abs(endpoint["score"]-actual))
                endpoint_checks += 1
            best = min(fit["endpoints"],key=lambda endpoint:(-endpoint["score"],tuple(endpoint["key"])))
            if fit["key"] != best["key"] or fit["score"] != best["score"]:
                raise ValueError("Wrong retained endpoint")
        if set(transfers) != {*inputs["controls"],"borg-latin","borg-english"}:
            raise ValueError("Incomplete transfer inventory")
        metrics = {}
        for name, transfer in transfers.items():
            guard()
            language = "english" if name == "borg-english" else "latin"
            records = (inputs["controls"][name]["transfer"] if name in inputs["controls"]
                       else [row["symbols"] for row in inputs["historical"]["transfer"]])
            key = seal["fits"][name]["key"]
            decoded = literal_decode(records,key)
            if transfer["decoded"] != decoded or transfer["characters"] != sum(map(len,records)):
                raise ValueError("Changed transfer decode/length")
            close_number(transfer["score"],literal_score(decoded,models[language]))
            if result["transfer_scores"][name] != {k:v for k,v in transfer.items() if k != "decoded"}:
                raise ValueError("Changed compact transfer summary")
            if name in gold:
                metrics[name] = independent_metrics(records,decoded,gold[name],key)
        if metrics != result["control_metrics"] or len(metrics) != 2:
            raise ValueError("Independent known-answer metrics disagree")
        verdict = "PASS" if all(row["pass"] for row in metrics.values()) else "FAIL"
        if result["control_competence"] != verdict:
            raise ValueError("Changed competence gate")
        for key,cap in (("cpu_seconds",run.CPU_CAP),("wall_seconds",run.WALL_CAP),
                        ("peak_rss_bytes",run.RSS_CAP),("paid_cost_usd",0)):
            if type(result[key]) not in (int,float) or not 0 <= result[key] <= cap:
                raise ValueError("Producer exceeded bounded allocation")
        guard()
        run.save_new(run.OUT/"audit.json", {"status":"PASS_literal_scores_selection_and_accounting",
            "experiment":run.EXP,"freeze":freeze,"result_sha256":run.digest(result_path.read_bytes()),
            "seal_sha256":run.digest(seal_path.read_bytes()),"frozen_inputs":len(paths),
            "restart_endpoint_literal_score_checks":endpoint_checks,"transfer_record_sets":len(transfers),
            "retained_raw_piece_checks":retained_pieces,"max_score_delta":max(deltas),
            "control_metrics":metrics,"historical_recovery":"UNASSESSED_NO_EXTERNAL_ANSWER",
            "optimizer_trajectory_rng_not_replayed":True,"shared_parser_selector_generator":True,
            "same_researcher_not_external_expert":True,"no_new_fit_or_historical_answer":True,
            "wall_seconds":time.monotonic()-wall,"cpu_seconds":time.process_time()-cpu,
            "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"paid_cost_usd":0})
        return "PASS_literal_scores_selection_and_accounting"
    except BaseException as error:
        run.save_new(run.OUT/"audit-failure.json", {"exception":type(error).__name__,"message":str(error),
            "no_retry":True,"wall_seconds":time.monotonic()-wall,"cpu_seconds":time.process_time()-cpu})
        raise
    finally:
        signal.alarm(0)
        signal.setitimer(signal.ITIMER_PROF,0)
        signal.signal(signal.SIGALRM,old_alarm)
        signal.signal(signal.SIGPROF,old_prof)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze",required=True)
    print(audit(parser.parse_args().freeze))
