"""ONE full original-source conditional replay, alternate family/map and RNG."""
import argparse
from fractions import Fraction as F
import itertools
import json
import signal
import time

import numpy as np
import torch

from scripts import run_window_reading_admit001 as run
from scripts.audit_reading_label_landscape001 import independent_point
from scripts.audit_reading_regrowth_recovery001 import linear_literal
from tests.test_window_conditionals import lattice_quantile
from voynich.reading_regrowth import SourceRegrowth
from voynich.source_action_proposal import ReadingEnvironment, ReadingState
from voynich.regrowth_trace import path_receipt


def direct_family(env, base, record, start, width):
    """Enumerate row tuples unconditionally; reconstruct all bindings from intervals."""
    tokens = []
    for r, text in enumerate(base.texts):
        offset, spans = 0, []
        for row in text:
            end = offset+len(env.pool[base.key[row]])
            assert env.records[r][offset:end] == env.pool[base.key[row]]
            spans.append((offset, end, row))
            offset = end
        assert offset == len(env.records[r])
        tokens.append(spans)
    boundaries = {0, *(end for _, end, _ in tokens[record])}
    if start not in boundaries or start+width not in boundaries:
        return {}
    left = [t for t in tokens[record] if t[1] <= start]
    right = [t for t in tokens[record] if t[0] >= start+width]
    answer = {}
    combinations = [(r,) for r in range(env.rows)]
    if width == 2:
        combinations += list(itertools.product(range(env.rows), repeat=2))
    for replacement in combinations:
        length = width if len(replacement) == 1 else 1
        local = [(start+i*length, start+(i+1)*length, row) for i, row in enumerate(replacement)]
        all_spans = tokens[:record]+[left+local+right]+tokens[record+1:]
        key, texts, legal = [-1]*env.rows, [], True
        for r, spans in enumerate(all_spans):
            text = []
            for first, last, row in spans:
                code = env.pool.index(env.records[r][first:last])
                if key[row] not in (-1, code):
                    legal = False
                    break
                key[row] = code
                text.append(row)
            if not legal:
                break
            texts.append(tuple(text))
        if legal:
            answer[replacement] = ReadingState(tuple(key), base.offsets, tuple(texts))
    return answer


def audit_table(source, sampler, base, table, seed, *, progress=lambda: None):
    env = sampler.env
    windows = [(r, start, width) for r, obs in enumerate(env.records) for start in range(len(obs))
               for width in (1, 2) if start+width <= len(obs)]
    assert table['window'] == list(windows[table['rank']]) and table['seed'] == seed
    family = direct_family(env, base.state, *table['window'])
    assert table['valid_endpoints'] == bool(family)
    before, after, values, weights = None, None, [], []
    selected, blocks, retained = None, 0, base
    old = F(*independent_point(source, env, base.state, progress))
    for replacement, state in family.items():
        mass = F(*independent_point(source, env, state, progress))
        ratio = mass/old
        weights.append(ratio)
        values.append((replacement, state, [hex(ratio.numerator), hex(ratio.denominator)]))
    assert [t['replacement'] for t in table['candidates']] == [list(x[0]) for x in values]
    for candidate, (_, state, ratio) in zip(table['candidates'], values, strict=True):
        assert candidate['ratio_hex'] == ratio
        assert candidate['source_length_delta'] == sum(map(len, state.texts))-len(base.actions)
        assert candidate['visited_delta'] == sum(k >= 0 for k in state.key)-sum(k >= 0 for k in base.state.key)
        # Independently rebuild contexts and count the exact matching point.
        record, start, width = table['window']
        prefix, consumed = [], 0
        for row in base.state.texts[record]:
            if consumed >= start:
                break
            prefix.append(row)
            consumed += len(env.pool[base.state.key[row]])
        old_context = new_context = source.state('')
        for row in prefix:
            old_context = int(source.transitions[old_context, row])
        new_context = old_context
        original = base.state.texts[record]
        pos = len(prefix)
        size, inner_end = 0, pos
        while size < width:
            row = original[inner_end]
            old_context = int(source.transitions[old_context, row])
            size += len(env.pool[base.state.key[row]])
            inner_end += 1
        for row in candidate['replacement']:
            new_context = int(source.transitions[new_context, row])
        steps = 0
        for row in original[inner_end:]:
            if old_context == new_context:
                break
            old_context = int(source.transitions[old_context, row])
            new_context = int(source.transitions[new_context, row])
            steps += 1
        assert candidate['suffix_steps'] == steps
        assert candidate['omitted_suffix_steps'] == len(original)-inner_end-steps
        assert candidate['contexts_matched'] == (old_context == new_context)
    if family:
        rng = np.random.default_rng(seed)
        before = rng.bit_generator.state
        selected, blocks = lattice_quantile(tuple(weights), rng, sampler.config.maximum_acceptance_blocks)
        after = rng.bit_generator.state
        state = values[selected][1]
        actual = linear_literal(env, table['retained'])
        assert actual == state
        # Legacy reference policy shared; new conditional probabilities/RNG above are separate.
        retained = sampler.path(forced_actions=tuple(table['retained']['actions']), progress=progress)
        assert path_receipt(retained) == table['retained']
    assert table['selected_index'] == selected and table['raw64_blocks'] == blocks
    assert table['rng_before'] == before and table['rng_after'] == after
    assert table['retained'] == path_receipt(retained)
    assert table['changed'] == (retained.state != base.state)
    assert table['wall_seconds'] >= 0 and table['cpu_seconds'] >= 0
    return {'valid_windows': int(bool(family)), 'candidates': len(values), 'changed_draws': int(table['changed'])}


def audit(freeze):
    run.training.old.require_frozen(freeze, run.PATHS)
    previous = run.require_prior()
    run.training.save_new(run.OUT/'audit-started.json', {'freeze': freeze, 'no_retry': True})
    run.training.limit_resources(run.WALL, run.CPU)
    wall, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    totals = dict.fromkeys(('valid_windows', 'candidates', 'changed_draws'), 0)
    try:
        def guard():
            r = run.training.resource_report(wall, cpu)
            if r['wall_seconds'] > run.WALL or r['cpu_seconds'] > run.CPU or r['peak_rss_bytes'] > run.HOST:
                raise MemoryError('Registered conditional audit cap; no retry')
            return r

        result = json.loads((run.OUT/'result.json').read_text())
        assert result['status'] == 'PASS_original_source_conditional_tables_and_cost' and result['freeze'] == freeze
        assert result['inputs'] == [run.training.artifact(run.ROOT/p) for p in run.PATHS]
        frames = [c for c in previous['cells'] if c['phase'] in run.PHASES]
        assert [(c['case'], c['phase']) for c in result['cells']] == [(i, p) for i in range(run.CASES) for p in run.PHASES]
        source, _ = run.prior.load_source()
        assert run.prior.array_identity(source) == result['source_arrays'] == previous['source_arrays']
        private = 0
        for frame_index, (cell, frame) in enumerate(zip(result['cells'], frames, strict=True)):
            saved = run.prior.load_archive(cell['archive'])
            original = run.prior.load_archive(frame['archive'])
            assert saved['records'] == original['records'] and saved['input_archive'] == frame['archive']
            assert (saved['case'], saved['kind'], saved['phase']) == (frame['case'], frame['kind'], frame['phase'])
            assert cell['kind'] == saved['kind'] and saved['no_chain_or_gold_selection'] is True
            env = ReadingEnvironment(saved['records'], rows=23, glyphs=6)
            sampler = SourceRegrowth(source, env)
            base = sampler.path(forced_actions=tuple(original['base']['actions']), progress=guard)
            assert path_receipt(base) == saved['base']
            count = sum(2*len(r)-1 for r in env.records)
            ranks = tuple(sorted({j*count//run.WINDOWS for j in range(run.WINDOWS)}))
            assert [t['rank'] for t in saved['tables']] == list(ranks)
            frame_totals = dict.fromkeys(totals, 0)
            for j, table in enumerate(saved['tables']):
                values = audit_table(source, sampler, base, table, run.SEED+frame_index*run.WINDOWS+j, progress=guard)
                for key in totals:
                    totals[key] += values[key]
                    frame_totals[key] += values[key]
            assert cell['windows'] == len(saved['tables'])
            assert all(cell[k] == frame_totals[k] for k in frame_totals)
            assert cell['window_cpu_seconds'] == sum(t['cpu_seconds'] for t in saved['tables'])
            assert cell['valid_two_glyph_windows'] == sum(t['valid_endpoints'] and t['window'][2] == 2 for t in saved['tables'])
            assert cell['length_changing_candidates'] == sum(c['source_length_delta'] != 0 for t in saved['tables'] for c in t['candidates'])
            private += cell['archive']['bytes']
            guard()
            print(json.dumps({'audited_frames': frame_index+1, 'allocated_frames': len(frames)}), flush=True)
        assert totals['valid_windows'] == result['valid_windows'] and totals['candidates'] == result['candidates']
        assert sum(c['windows'] for c in result['cells']) <= run.CASES*len(run.PHASES)*run.WINDOWS
        assert totals['candidates'] <= run.CASES*len(run.PHASES)*run.WINDOWS*552
        mean = sum(c['window_cpu_seconds'] for c in result['cells'])/max(1, totals['valid_windows'])
        assert mean == result['mean_window_cpu_per_valid_table_seconds']
        assert result['gates'] == {'some_valid_windows': totals['valid_windows'] > 0,
            'some_valid_two_glyph_windows': sum(c['valid_two_glyph_windows'] for c in result['cells']) > 0,
            'some_source_length_changing_candidates': sum(c['length_changing_candidates'] for c in result['cells']) > 0,
            'mean_window_cpu_per_valid_table_under_two_seconds': mean <= 2}
        assert all(result['gates'].values())
        assert private == result['private_archive_bytes'] <= run.PRIVATE
        resources = result['resources']
        assert resources['wall_seconds'] <= run.WALL and resources['cpu_seconds'] <= run.CPU
        assert resources['peak_rss_bytes'] <= run.HOST and resources['paid_spend_usd'] == 0
        assert all(result[k] is True for k in ('no_gold_metrics_neural_training_or_optimizer',
            'no_recovery_mixing_or_historical_claim', 'cost_includes_invalid_window_work_but_excludes_frame_initialization_io'))
        run.training.old.require_frozen(freeze, run.PATHS)
        assert run.prior.array_identity(source) == result['source_arrays']
        run.training.save_new(run.OUT/'audit.json', {'status': 'PASS_original_source_all_conditional_targets_and_rng',
            'freeze': freeze, 'result': run.training.artifact(run.OUT/'result.json'), 'totals': totals,
            'resources': guard(), 'independent_expert_review': False,
            'legacy_source_backend_and_reference_counts_shared': True,
            'new_family_map_full_target_and_categorical_threshold_replay': True,
            'no_chain_extension_gold_metrics_or_neural_calls': True})
        return totals
    except Exception as error:
        run.training.save_new(run.OUT/'audit-failure.json', {'error': repr(error), 'no_retry': True,
            'totals': totals, 'resources': run.training.resource_report(wall, cpu)})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    print(audit(parser.parse_args().freeze))
