"""Full sixteen-subset original-model attention-write factorial on fresh lookups."""

import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .campaign import answer_correct, canonical_digest, digest, load_config, write_json
from .causal_campaign import condition_seed
from .geometry import matched_random_delta
from .path3_backend import Path3Workspace
from .path6_campaign import changed_positions
from .path10_tasks import path10_tasks


UPSTREAM = 23
WINDOW = (28, 29, 30, 31)
ALL_CUT = tuple(range(24, 36))
MASKS = tuple(range(16))
FULL = 15
SINGLES = (1, 2, 4, 8)
SEED = 510111
CAP_SECONDS = 1800
CAP_BYTES = 45_000_000_000
TOLERANCE = .002
SOURCE_PATHS = (
    'src/voynich/workspace/path10_tasks.py',
    'src/voynich/workspace/path10_campaign.py',
    'src/voynich/workspace/path3_backend.py',
    'src/voynich/workspace/path6_tasks.py',
    'src/voynich/workspace/path6_campaign.py',
    'src/voynich/workspace/path8_tasks.py',
    'src/voynich/workspace/route_backend.py',
    'src/voynich/workspace/campaign.py',
    'src/voynich/workspace/causal_campaign.py',
    'src/voynich/workspace/geometry.py',
    'scripts/path0010_analyze.py',
    'configs/jspace0001.json',
    'docs/experiments/PATH-0010.md',
)


def mask_layers(mask):
    if type(mask) is not int or mask not in MASKS:
        raise ValueError('Invalid four-block subset mask')
    return tuple(block for index, block in enumerate(WINDOW) if mask & (1 << index))


def mobius_terms(margins):
    """Anchored finite differences of R(S)=m(empty)-m(S), not functional ANOVA."""
    if set(margins) != set(MASKS) or not all(np.isfinite(value) for value in margins.values()):
        raise ValueError('Complete finite margin grid required')
    effects = {mask: margins[0]-margins[mask] for mask in MASKS}
    terms = {}
    for mask in MASKS[1:]:
        terms[mask] = sum((-1)**((mask.bit_count()-part.bit_count()))*effects[part]
                          for part in MASKS if part & mask == part)
    if not np.isclose(sum(terms.values()), effects[FULL], rtol=0, atol=1e-8):
        raise AssertionError('Möbius reconstruction failed')
    return effects, terms


def _check_logits(actual, expected, label, controls):
    error = float(np.max(np.abs(actual-expected)))
    controls[label] = max(controls.get(label, 0.0), error)
    if error >= TOLERANCE or int(actual.argmax()) != int(expected.argmax()):
        write_json(Path('results/PATH-0010/failure.json'), {
            'status': 'uninformative_numerical', 'reason': label,
            'max_abs_logit_error': error, 'argmax_match': int(actual.argmax())
            == int(expected.argmax()), 'controls': controls})
        raise ValueError(f'PATH-0010 numerical control {label} failed: {error}')


def _row(task, condition, mask, phrase, tokens, logits, source, donor, eligible, copies):
    source_first = source['tokens'][0] if source['tokens'] else None
    donor_first = donor['tokens'][0] if donor['tokens'] else None
    return {
        'task_id': task['id'], 'bundle': task['bundle'], 'template': task['template'],
        'family': task['family'], 'condition': condition, 'mask': mask,
        'layers': list(mask_layers(mask)) if mask is not None else None,
        'text': phrase, 'tokens': tokens, 'eligible': task['id'] in eligible,
        'baseline_copy': task['id'] in copies,
        'desired_answer': answer_correct(phrase, [task['donor_answer']])
        if task['family'] == 'binding' else False,
        'source_answer': answer_correct(phrase, [task['answer']]),
        'donor_first_token': bool(tokens) and donor_first is not None and tokens[0] == donor_first,
        'copy_preserved': answer_correct(phrase, [task['answer']])
        if task['family'] == 'copy' else False,
        'donor_margin': (float(logits[donor_first]-logits[source_first])
                         if source_first is not None and donor_first is not None
                         and source_first != donor_first else None),
    }


def _bootstrap(rows, success):
    lookup = {(r['task_id'], r['condition'], r['mask']): r for r in rows}
    members = {bundle: sorted(key for key in success if int(key.split('/')[1]) == bundle)
               for bundle in range(8)}
    d_by_bundle = {}
    for bundle, keys in members.items():
        d_by_bundle[bundle] = float(np.mean([
            min(lookup[key, 'subset', mask]['donor_margin']
                -lookup[key, 'subset', FULL]['donor_margin'] for mask in SINGLES)
            for key in keys]))
    rng = np.random.default_rng(SEED)
    d_samples, rate_samples = [], []
    for _ in range(10_000):
        picks = rng.integers(0, 8, size=8)
        d_samples.append(float(np.mean([d_by_bundle[int(bundle)] for bundle in picks])))
        picked_keys = [key for bundle in picks for key in members[int(bundle)]]
        full_rate = np.mean([not lookup[key, 'subset', FULL]['donor_first_token']
                             for key in picked_keys])
        best_single = max(np.mean([not lookup[key, 'subset', mask]['donor_first_token']
                                    for key in picked_keys]) for mask in SINGLES)
        rate_samples.append(float(full_rate-best_single))
    return {
        'resamples': 10_000,
        'equal_bundle_mean_d_95': [float(np.quantile(d_samples, .025)),
                                   float(np.quantile(d_samples, .975))],
        'full_minus_best_single_rate_95': [float(np.quantile(rate_samples, .025)),
                                            float(np.quantile(rate_samples, .975))],
    }


def summarize(rows, eligible, copies):
    lookup = {(r['task_id'], r['condition'], r['mask']): r for r in rows}
    success = {key for key in eligible if lookup[key, 'subset', 0]['desired_answer']}
    if not success:
        return {'eligible': len(eligible), 'source_copy': len(copies),
                'upstream_success': 0, 'bundle_coverage': 0}
    subset = {}
    for mask in MASKS:
        relevant = [lookup[key, 'subset', mask] for key in success]
        subset[str(mask)] = {
            'layers': list(mask_layers(mask)), 'denominator': len(success),
            'donor_first_removed': sum(not r['donor_first_token'] for r in relevant),
            'source_full': sum(r['source_answer'] for r in relevant),
            'donor_full': sum(r['desired_answer'] for r in relevant),
            'mean_donor_margin': float(np.mean([r['donor_margin'] for r in relevant])),
            'copy_preserved': sum(lookup[key, 'subset', mask]['copy_preserved'] for key in copies),
        }
    other = {}
    for name in ('random_full', 'all_cut'):
        relevant = [lookup[key, name, None] for key in success]
        other[name] = {
            'denominator': len(success),
            'donor_first_removed': sum(not r['donor_first_token'] for r in relevant),
            'source_full': sum(r['source_answer'] for r in relevant),
            'donor_full': sum(r['desired_answer'] for r in relevant),
            'mean_donor_margin': float(np.mean([r['donor_margin'] for r in relevant])),
            'copy_preserved': sum(lookup[key, name, None]['copy_preserved'] for key in copies),
        }
    d_by_bundle = {}
    effects_by_bundle = {mask: [] for mask in MASKS[1:]}
    terms_by_bundle = {mask: [] for mask in MASKS[1:]}
    for bundle in range(8):
        keys = sorted(key for key in success if int(key.split('/')[1]) == bundle)
        if not keys:
            d_by_bundle[str(bundle)] = None
            continue
        differences = []
        effect_rows, term_rows = [], []
        for key in keys:
            margins = {mask: lookup[key, 'subset', mask]['donor_margin'] for mask in MASKS}
            effects, terms = mobius_terms(margins)
            effect_rows.append(effects)
            term_rows.append(terms)
            differences.append(min(margins[mask]-margins[FULL] for mask in SINGLES))
        d_by_bundle[str(bundle)] = float(np.mean(differences))
        for mask in MASKS[1:]:
            effects_by_bundle[mask].append(float(np.mean([r[mask] for r in effect_rows])))
            terms_by_bundle[mask].append(float(np.mean([r[mask] for r in term_rows])))
    covered = sum(value is not None for value in d_by_bundle.values())
    equal_d = (float(np.mean([value for value in d_by_bundle.values() if value is not None]))
               if covered == 8 else None)
    bootstrap = _bootstrap(rows, success) if covered == 8 else None
    return {
        'eligible': len(eligible), 'source_copy': len(copies),
        'upstream_success': len(success), 'bundle_coverage': covered,
        'subset': subset, 'other': other, 'd_by_bundle': d_by_bundle,
        'equal_bundle_mean_d': equal_d,
        'positive_d_bundles': sum(value is not None and value > 0 for value in d_by_bundle.values()),
        'equal_bundle_mean_effects': {str(mask): float(np.mean(values))
                                      for mask, values in effects_by_bundle.items()},
        'equal_bundle_mean_mobius': {str(mask): float(np.mean(values))
                                     for mask, values in terms_by_bundle.items()},
        'full_minus_sum_singles': (float(np.mean(effects_by_bundle[FULL]))
                                   - sum(float(np.mean(effects_by_bundle[mask])) for mask in SINGLES)),
        'bootstrap': bootstrap,
    }


def run():
    output, result = Path('outputs/PATH-0010'), Path('results/PATH-0010')
    if (result/'inputs.json').exists() or (output/'observations.jsonl').exists():
        raise FileExistsError('No automatic PATH-0010 rerun')
    source_status = subprocess.check_output(
        ['git', 'status', '--porcelain', '--', *SOURCE_PATHS], text=True).splitlines()
    if source_status:
        raise RuntimeError('PATH-0010 source and registration files must be committed before launch: '
                           + '; '.join(source_status))
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    (result/'failure.json').unlink(missing_ok=True)
    started = time.monotonic()
    config = load_config('configs/jspace0001.json')
    model = Path3Workspace(config['model_snapshot'], dense_transport=True)
    model.mx.set_memory_limit(min(config['memory_bytes_cap'], CAP_BYTES))
    controls = {}
    rows = []

    def budget():
        if time.monotonic()-started >= CAP_SECONDS or model.mx.get_peak_memory() > CAP_BYTES:
            write_json(result/'incomplete.json', {'reason': 'resource_cap',
                                                   'elapsed_seconds': time.monotonic()-started,
                                                   'completed_rows': len(rows)})
            raise RuntimeError('PATH-0010 resource cap')

    def finish(status, reason=None, **extra):
        decision = {'status': status, 'reason': reason, 'controls': controls,
                    'elapsed_seconds': time.monotonic()-started,
                    'peak_mlx_bytes': model.mx.get_peak_memory(), **extra}
        write_json(result/'decision.json', decision)
        return decision

    tasks = path10_tasks()
    prompts = {r[key] for r in tasks for key in ('prompt', 'donor_prompt')}
    ids = {prompt: model.encode(prompt, chat=True) for prompt in prompts}
    changed = {r['id']: changed_positions(ids[r['prompt']], ids[r['donor_prompt']])
               for r in tasks}
    write_json(output/'rendered-inputs.json', {canonical_digest(prompt): token_ids
                                               for prompt, token_ids in ids.items()})
    sources = [Path(path) for path in SOURCE_PATHS]
    write_json(result/'inputs.json', {
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_worktree_status': source_status,
        'source_sha256': {str(path): digest(path) for path in sources},
        'tasks_sha256': canonical_digest(tasks),
        'rendered_inputs_sha256': digest(output/'rendered-inputs.json'),
        'changed_positions_sha256': canonical_digest(changed),
        'model_input_sha256': digest('results/JSPACE-0001/inputs.json'),
        'seed': SEED, 'upstream': UPSTREAM, 'window': WINDOW,
        'all_cut': ALL_CUT, 'masks': MASKS,
        'cap_seconds': CAP_SECONDS, 'memory_bytes_cap': CAP_BYTES,
        'tolerance': TOLERANCE,
    })

    baseline_dir = output/'baselines'
    baseline_dir.mkdir(exist_ok=True)
    baselines = {}

    def baseline(prompt):
        if prompt not in baselines:
            budget()
            token_ids = ids[prompt]
            phrase, tokens, fields, logits = model.generate_field(token_ids,
                                                                  capture_layers=(UPSTREAM,))
            clean_logits, writes = model.clean_final_writes(token_ids, start=24)
            _check_logits(clean_logits, logits, 'clean_write_capture', controls)
            local = baseline_dir/(canonical_digest(prompt)+'.npz')
            np.savez_compressed(local, logits=logits, field=fields[UPSTREAM],
                                **{f'write_{layer}': value for layer, value in writes.items()})
            baselines[prompt] = {'text': phrase, 'tokens': tokens, 'logits': logits,
                                 'field': fields[UPSTREAM], 'writes': writes,
                                 'arrays_sha256': digest(local)}
        return baselines[prompt]

    public = []
    for task in tasks:
        source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
        public.append({
            'id': task['id'], 'bundle': task['bundle'], 'template': task['template'],
            'family': task['family'], 'source_text': source['text'], 'donor_text': donor['text'],
            'source_correct': answer_correct(source['text'], [task['answer']]),
            'donor_correct': answer_correct(donor['text'], [task['donor_answer']]),
            'first_token_differs': bool(source['tokens']) and bool(donor['tokens'])
            and source['tokens'][0] != donor['tokens'][0],
            'source_arrays_sha256': source['arrays_sha256'],
            'donor_arrays_sha256': donor['arrays_sha256'],
        })
    write_json(result/'baselines.json', public)
    eligible = {r['id'] for r in public if r['family'] == 'binding' and r['source_correct']
                and r['donor_correct'] and r['first_token_differs']}
    copies = {r['id'] for r in public if r['family'] == 'copy' and r['source_correct']}
    write_json(result/'capability.json', {'eligible': len(eligible), 'source_copy': len(copies)})
    if len(eligible) < 24 or len(copies) < 14:
        finish('inconclusive', 'baseline_competence', eligible=len(eligible), source_copy=len(copies))
        return

    with (output/'observations.jsonl').open('w') as stream:
        for task in tasks:
            budget()
            source, donor = baseline(task['prompt']), baseline(task['donor_prompt'])
            token_ids = ids[task['prompt']]
            field = np.zeros_like(source['field'])
            field[changed[task['id']]] = (donor['field']-source['field'])[changed[task['id']]]
            whole_logits, _, _ = model.prefill_field(
                token_ids, fields={UPSTREAM: donor['field']-source['field']})
            _check_logits(whole_logits, donor['logits'], 'whole_donor', controls)
            identity_text, identity_tokens, identity_logits, _ = model.generate_cut(
                token_ids, upstream_layer=UPSTREAM, upstream_field=np.zeros_like(field),
                clean_writes=source['writes'])
            _check_logits(identity_logits, source['logits'], 'zero_edit_identity', controls)
            if identity_text != source['text'] or identity_tokens != source['tokens']:
                raise ValueError('Zero-edit full-generation identity failed')
            upstream_text, upstream_tokens, upstream_logits, edited_writes = model.generate_cut(
                token_ids, upstream_layer=UPSTREAM, upstream_field=field,
                clean_writes=source['writes'], capture_writes=True)
            first = _row(task, 'subset', 0, upstream_text, upstream_tokens, upstream_logits,
                         source, donor, eligible, copies)
            rows.append(first)
            stream.write(json.dumps(first, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            for mask in MASKS[1:]:
                budget()
                phrase, tokens, logits, _ = model.generate_cut(
                    token_ids, upstream_layer=UPSTREAM, upstream_field=field,
                    clean_writes=source['writes'], cut_layers=mask_layers(mask))
                row = _row(task, 'subset', mask, phrase, tokens, logits,
                           source, donor, eligible, copies)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
            random_deltas = {}
            for layer in WINDOW:
                target = source['writes'][layer]-edited_writes[layer]
                random_deltas[layer] = matched_random_delta(
                    target, seed=condition_seed(SEED, task['id'], layer, 'random-window'))
                mismatch = abs(float(np.linalg.norm(random_deltas[layer]))
                               - float(np.linalg.norm(target)))
                controls['random_norm_error'] = max(controls.get('random_norm_error', 0.0),
                                                    mismatch)
                if mismatch >= TOLERANCE:
                    write_json(result/'failure.json', {
                        'status': 'uninformative_numerical',
                        'reason': 'random_norm_error', 'max_abs_norm_error': mismatch,
                        'controls': controls, 'completed_rows': len(rows)})
                    raise ValueError('Random full-window norm mismatch')
            for condition, cut_layers, additive in (
                ('random_full', WINDOW, random_deltas), ('all_cut', ALL_CUT, None)):
                budget()
                phrase, tokens, logits, _ = model.generate_cut(
                    token_ids, upstream_layer=UPSTREAM, upstream_field=field,
                    clean_writes=source['writes'], cut_layers=cut_layers,
                    random_deltas=additive)
                if condition == 'all_cut':
                    _check_logits(logits, source['logits'], 'all_cut_source', controls)
                row = _row(task, condition, None, phrase, tokens, logits,
                           source, donor, eligible, copies)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
    write_json(result/'rows.json', rows)
    summary = summarize(rows, eligible, copies)
    write_json(result/'summary.json', summary)
    success = summary['upstream_success']
    upstream_rate = success/len(eligible)
    upstream_copy = summary['subset']['0']['copy_preserved']/len(copies) if success else 0.0
    full_copy = summary['subset']['15']['copy_preserved']/len(copies) if success else 0.0
    if upstream_rate < .75 or summary['bundle_coverage'] < 8 or upstream_copy < .9:
        finish('uninformative_upstream', 'upstream_or_bundle_coverage',
               upstream_rate=upstream_rate, bundle_coverage=summary['bundle_coverage'])
        return
    full_rate = summary['subset']['15']['donor_first_removed']/success
    random_rate = summary['other']['random_full']['donor_first_removed']/success
    single_rates = {str(mask): summary['subset'][str(mask)]['donor_first_removed']/success
                    for mask in SINGLES}
    best_single_rate = max(single_rates.values())
    positive_control = full_rate >= .35
    supported = (positive_control and full_rate-random_rate >= .2
                 and full_rate-best_single_rate >= .2 and full_copy >= .9
                 and summary['equal_bundle_mean_d'] >= 2.0
                 and summary['positive_d_bundles'] >= 6)
    status = ('exploratory_supported' if supported else
              'uninformative_positive_control' if not positive_control else 'exploratory_failed')
    finish(status, None, supported=supported, upstream_rate=upstream_rate,
           full_rate=full_rate, random_rate=random_rate,
           single_rates=single_rates, best_single_rate=best_single_rate,
           full_minus_best_single_rate=full_rate-best_single_rate,
           equal_bundle_mean_d=summary['equal_bundle_mean_d'],
           positive_d_bundles=summary['positive_d_bundles'],
           scope='Supplied English direct lookup only; no historical decipherment')


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        failure = Path('results/PATH-0010/failure.json')
        failure.parent.mkdir(parents=True, exist_ok=True)
        if not failure.exists():
            write_json(failure, {'status': 'incomplete_or_uninformative',
                                 'error_type': type(error).__name__, 'message': str(error)})
        raise
