"""Post-run artifact, finite-bank bound and sampled literal-support audit.

Does not fit keys, enumerate new readings or open answers. Numerical candidate
support is independently replayed on a fixed sample, not claimed exhaustive.
"""
import json
import math
import signal
import time

from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import load_archive, resource_report, save_new
from scripts.run_key_bank_read001 import MANIFEST, NAMES, OUT, PATHS, ROOT, admission
from scripts.run_latin_source_model001 import artifact, limit_resources


def logsum(values):
    values = list(values)
    maximum = max(values, default=-math.inf)
    return maximum if maximum == -math.inf else maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


def logvalue(value):
    return -math.inf if value is None else value


def close(actual, expected):
    if actual != expected and (not math.isfinite(actual) or not math.isfinite(expected)
                               or abs(actual - expected) > 1e-7):
        raise ValueError('Independent score or bound arithmetic mismatch')


def audit_reading(alphabet, records, bank, reading):
    active = [i for i, row in enumerate(bank['bank']) if row['log_weight'] is not None]
    if reading['active_bank_indices'] != active:
        raise ValueError('Positive key inventory mismatch')
    mixture = reading['mixture']
    lists = mixture['per_key']
    if len(lists) != len(active) or mixture['k'] != 8:
        raise ValueError('Incomplete per-key lists or changed k')
    raw = [bank['bank'][i]['log_weight'] for i in active]
    z = logsum(raw)
    weights = [w - z for w in raw]
    encoders = [dict(zip(alphabet, bank['bank'][i]['units'], strict=True)) for i in active]
    candidates, joint = {}, []
    tuples = 0
    for local, row in enumerate(lists):
        values = row['readings']
        scores = [v[1] for v in values]
        if len(values) > 8 or any(a + 1e-7 < b for a, b in zip(scores, scores[1:])):
            raise ValueError('Per-key ranking/order mismatch')
        if len({tuple(v[0]) for v in values}) != len(values):
            raise ValueError('Duplicate tuple in key list')
        next_score = logvalue(row['next_log_probability'])
        if (len(values) < 8 and next_score != -math.inf) or (scores and next_score > scores[-1] + 1e-7):
            raise ValueError('Invalid next-tuple upper bound')
        if scores and logsum(scores + [next_score]) > logvalue(row['log_likelihood']) + 1e-7:
            raise ValueError('Enumerated mass exceeds exact per-key evidence')
        for texts, score in values:
            texts = tuple(texts)
            if texts in candidates:
                close(score, candidates[texts])
            candidates[texts] = score
        if scores:
            joint.append((raw[local] + scores[0], -local, tuple(values[0][0])))
        tuples += len(values)
    if len(candidates) != mixture['candidate_tuples'] or reading['checks']['tuples'] != tuples:
        raise ValueError('Candidate/count mismatch')
    if reading['checks']['keys'] != len(active):
        raise ValueError('Checked key count mismatch')
    upper = logsum(w + logvalue(row['next_log_probability']) for w, row in zip(weights, lists))
    evidence = logsum(w + logvalue(row['log_likelihood']) for w, row in zip(weights, lists))
    close(logvalue(mixture['unseen_log_probability_upper_bound']), upper)
    close(logvalue(mixture['log_likelihood']), evidence)
    if joint:
        score, negative, texts = max(joint)
        if tuple(reading['joint']['plaintexts']) != texts or reading['joint']['key_index'] != active[-negative]:
            raise ValueError('Joint key/text optimum mismatch')
        close(reading['joint']['log_probability'], score)
    elif reading['joint'] is not None:
        raise ValueError('Unsupported bank has a joint reading')
    texts = mixture['plaintexts']
    winner = None if texts is None else tuple(texts)
    best = logvalue(mixture['joint_log_probability'])
    tolerance = 1e-8 * max(1, sum(map(len, records)))
    separated = winner is not None and (upper == -math.inf or best > upper + tolerance)
    if separated != mixture['floating_bound_separated']:
        raise ValueError('Separation flag mismatch')
    ordered = sorted(candidates)
    sample = {ordered[i * (len(ordered) - 1) // 63] for i in range(64)} if ordered else set()
    if winner is not None:
        if winner not in candidates:
            raise ValueError('Winner not present in candidate union')
        sample.add(winner)
    if joint:
        sample.add(max(joint)[2])
    support_checks = 0
    for text_tuple in sorted(sample):
        supported = [i for i, encoder in enumerate(encoders)
                     if all(''.join(encoder[c] for c in text) == record
                            for text, record in zip(text_tuple, records, strict=True))]
        support_checks += len(encoders)
        score = candidates[text_tuple] + logsum(weights[i] for i in supported)
        if score > best + 1e-7:
            raise ValueError('Sampled candidate beats declared winner')
        if text_tuple == winner:
            if list(mixture['compatible_key_indices']) != supported:
                raise ValueError('Independent literal winner support mismatch')
            close(score, best)
    if winner is None and (candidates or best != -math.inf):
        raise ValueError('Missing winner despite supported candidates')
    return {'active_keys': len(active), 'listed_tuples': tuples, 'unique_candidates': len(candidates),
            'sampled_candidate_tuples': len(sample), 'literal_candidate_key_checks': support_checks,
            'floating_bound_separated': separated, 'all_candidate_supports_independently_replayed': False}


def main():
    campaign = json.loads((OUT / 'campaign.json').read_text())
    admitted, _ = admission(campaign['freeze'])
    require_frozen(campaign['freeze'], PATHS)
    if [r['case'] for r in campaign['results']] != list(NAMES):
        raise ValueError('Campaign identity/order mismatch')
    limit_resources(600, 600)
    wall, cpu = time.monotonic(), time.process_time()
    manifest = json.loads((ROOT / MANIFEST).read_text())
    cases = {}
    for process in campaign['results']:
        name = process['case']
        result_path = OUT / f'{name}.json'
        if process['returncode'] != 0:
            cases[name] = {'status': 'process_failure_retained', 'process': process}
            continue
        result = json.loads(result_path.read_text())
        if result['freeze'] != campaign['freeze'] or result['case'] != name:
            raise ValueError('Result identity mismatch')
        points = json.loads((OUT / f'{name}-points.json').read_text())
        if points['readings'] != result['points'] or result['transfer'] != manifest['cases'][name]['artifacts']['transfer']:
            raise ValueError('Prediction point/input binding mismatch')
        load_archive(points['readings'])
        reading = load_archive(result['readings'])
        report = {'result': artifact(result_path), 'status': result['status']}
        if admitted[name] is None:
            if (result['status'] != 'fit_bank_unavailable' or reading['mixture'] is not None
                    or reading['joint'] is not None or result['bank'] is not None):
                raise ValueError('Unavailable bank incorrectly represented')
        else:
            if result['bank'] != admitted[name]['bank'] or reading['status'] != 'complete':
                raise ValueError('Bank/status mismatch')
            observed = checked_artifact(result['transfer'])
            bank = load_archive(admitted[name]['bank'])
            report.update(audit_reading(observed['context']['source_alphabet'], observed['records'], bank, reading))
            progress_path = ROOT / f'outputs/KEY-BANK-READ-001/{name}-progress.jsonl'
            progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
            expected = [{'active_index': i, 'bank_index': original, **{k: values[k] for k in ('nodes', 'edges', 'expanded')}}
                        for i, (original, values) in enumerate(zip(reading['active_bank_indices'], reading['mixture']['per_key'], strict=True))]
            if progress != expected:
                raise ValueError('Progress record mismatch')
            report['progress'] = artifact(progress_path)
        cases[name] = report
    signal.alarm(0)
    save_new(OUT / 'audit.json', {'status': 'PASS', 'campaign': artifact(OUT / 'campaign.json'),
        'auditor': artifact(ROOT / 'scripts/audit_key_bank_read001.py'), 'cases': cases,
        'no_gold_opened': True, 'resources': resource_report(wall, cpu)})


if __name__ == '__main__':
    main()
