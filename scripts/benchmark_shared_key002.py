"""Fixed k-best/bound feasibility probe on the first benchmark's artificial inputs."""
import argparse
import gc
import json
import math
import resource
import signal
import time
from dataclasses import asdict, is_dataclass

from scripts.benchmark_shared_key001 import PATHS as PREVIOUS_PATHS
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import resource_report, save_new
from scripts.run_latin_source_model001 import ROOT, artifact, limit_resources
from voynich.compact_suffix_adapter import CompactSuffixAdapter
from voynich.compact_suffix_source import CompactSuffixSource
from voynich.fresh_reader_panel import encode_known
from voynich.kbest_suffix import bounded_mixture
from voynich.local_key_bank import one_move_bank
from voynich.sparse_suffix_source import decode

EXP = 'SHARED-KEY-BENCH-002'
OUT, BULK = ROOT / 'results' / EXP, ROOT / 'outputs' / EXP
INPUT = 'results/SHARED-KEY-BENCH-001/inputs.json'
PATHS = sorted(set(PREVIOUS_PATHS + [INPUT, 'scripts/benchmark_shared_key002.py',
    'src/voynich/kbest_suffix.py', 'tests/test_kbest_suffix.py',
    'docs/experiments/SHARED-KEY-BENCH-002.md',
    'docs/research/bounded-key-list-mixture-2026-09-30.md']))


def serializable(value):
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {key: serializable(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if value != -math.inf:
            raise ValueError('Unexpected nonfinite result')
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    require_frozen(args.freeze, PATHS)
    save_new(OUT / 'started.json', {'freeze': args.freeze, 'start_unix': time.time()})
    limit_resources(650, 650)
    wall, cpu = time.monotonic(), time.process_time()
    selected = json.loads((ROOT / 'results/LATIN-SOURCE-COMPACT-001/large.json').read_text())
    inputs = json.loads((ROOT / INPUT).read_text())
    if artifact(ROOT / selected['counts']['path']) != inputs['source'] or inputs['source'] != selected['counts']:
        raise ValueError('Artificial workload source changed')
    source = CompactSuffixAdapter(CompactSuffixSource.load(ROOT / selected['counts']['path']), selected['selected']['tau'])
    bank = one_move_bank(inputs['parent'], 'ABCDEF')
    if len(bank) != inputs['bank_size'] or len(bank) != 1197:
        raise ValueError('Artificial bank changed')
    results = {}
    BULK.mkdir(parents=True, exist_ok=True)
    for name, records in [('source_generated', inputs['records'][4:]), ('glyph_shuffle', inputs['shuffled'][4:])]:
        began, clock = time.monotonic(), time.process_time()
        checks = {'keys': 0, 'tuples': 0, 'forward_records': 0, 'max_delta': 0.}
        signal.alarm(300)
        try:
            with (BULK / f'{name}-progress.jsonl').open('x') as stream:
                def progress(index, values):
                    for texts, score in values.readings:
                        if [encode_known(t, source.alphabet, bank[index]) for t in texts] != records:
                            raise ValueError('Listed tuple fails independent literal encoding')
                        replay = len(records) * math.log(1 / 225)
                        for text in texts:
                            state = source.state('')
                            replay += len(text) * math.log1p(-1 / 225)
                            for char in text:
                                letter = source.alphabet.index(char)
                                replay += math.log(source.probabilities[state, letter])
                                state = source.step(state, letter)
                        checks['max_delta'] = max(checks['max_delta'], abs(replay - score))
                    if index % 32 == 0 or index == len(bank) - 1:
                        reference = [decode(source, bank[index], row, 1 / 225, max_nodes=500_000) for row in records]
                        total = math.fsum(r.log_likelihood for r in reference)
                        if math.isfinite(total) != math.isfinite(values.log_likelihood):
                            raise ValueError('Independent forward support differs')
                        if math.isfinite(total):
                            checks['max_delta'] = max(checks['max_delta'], abs(total - values.log_likelihood))
                        checks['forward_records'] += len(records)
                    checks['keys'] += 1
                    checks['tuples'] += len(values.readings)
                    if checks['max_delta'] > 1e-7:
                        raise ValueError('Independent path/evidence check failed')
                    stream.write(json.dumps({'key': index, 'nodes': values.nodes, 'edges': values.edges,
                                             'expanded': values.expanded, 'readings': len(values.readings)}) + '\n')
                    if index % 64 == 0:
                        stream.flush()
                        print(json.dumps({'case': name, 'keys': index + 1}), flush=True)
                        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 4 * 1024**3:
                            raise MemoryError('Sampled 4GiB cap')
                value = bounded_mixture(source, bank, records, 1 / 225, k=8, progress=progress,
                                        max_nodes=500_000, max_edges=2_000_000, max_expanded=50_000)
            archive = save_new(BULK / f'{name}.json.gz', serializable(value), compressed=True)
            result = {key: v for key, v in serializable(value).items() if key != 'per_key'}
            result.update(status='complete', archive=archive)
        except (RuntimeError, TimeoutError, MemoryError) as error:
            result = {'status': type(error).__name__, 'error': str(error)}
        finally:
            signal.alarm(0)
        result.update(checks=checks, wall_seconds=time.monotonic() - began,
                      cpu_seconds=time.process_time() - clock, resources=resource_report(wall, cpu))
        save_new(OUT / f'{name}.json', result)
        results[name] = result
        gc.collect()
    save_new(OUT / 'result.json', {'freeze': args.freeze, 'inputs': artifact(ROOT / INPUT),
                                 'results': results, 'resources': resource_report(wall, cpu),
                                 'status': 'same_artificial_workloads_not_recovery', 'paid_cost': 0})


if __name__ == '__main__':
    main()
