"""EXP-0010: matched-exposure context lengths and locus-boundary input ablation.

All helpers belong to this experiment. No manuscript final-test text is scored.
"""

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import NGram
from .context_analysis import paired_summary
from .model import ModelConfig, VoynichTransformer
from .runtime import PageWindows, corpus_identity, digest, environment, loss_sum, resolve_device, write_json


def lc_config(size, context, vocab=112, pad=0):
    sizes = {'main': (192, 4, 512), 'compact': (128, 2, 352), 'smoke': (16, 1, 32)}
    width, layers, hidden = sizes[size]
    return ModelConfig(vocab_size=vocab, pad_id=pad, d_model=width, n_layers=layers, n_heads=4,
                       d_ff=hidden, context_length=context, dropout=.1)


def lc_sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def lc_memory(device):
    report = {'process_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
              * (1 if sys.platform == 'darwin' else 1024)}
    if device == 'mps':
        report.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                      mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    return report


def lc_optimizer(model):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': .1},
                             {'params': no_decay, 'weight_decay': 0.}], lr=.0006, betas=(.9, .95))


def lc_update(model, optimizer, inputs, targets, *, context, micro_tokens=2048):
    """One optimizer update; summed microbatch losses / one common target count."""
    if inputs.shape != targets.shape or inputs.shape[1] % context:
        raise ValueError('Equal-shaped source blocks must divide into context length')
    count = int(targets.ne(-100).sum())
    if not count:
        raise ValueError('No scorable training targets')
    x = inputs.reshape(-1, context)
    y = targets.reshape(-1, context)
    micro_rows = max(1, micro_tokens // context)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.
    executed_rows = 0
    for start in range(0, len(x), micro_rows):
        # Wholly padded rows contribute nothing and can be omitted exactly.
        keep = y[start:start+micro_rows].ne(-100).any(-1)
        if not bool(keep.any()):
            continue
        batch_x, batch_y = x[start:start+micro_rows][keep], y[start:start+micro_rows][keep]
        value, _ = loss_sum(model(batch_x).logits, batch_y)
        (value / count).backward()
        total_loss += float(value.detach().cpu())
        executed_rows += len(batch_x)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return {'loss_nats': total_loss / count, 'scored_tokens': count, 'gradient_norm': float(norm.cpu()),
            'executed_input_positions': executed_rows * context,
            'attention_pair_slots': executed_rows * context * context}


def lc_benchmark(output, device='mps', warmup=5, updates=10, threads=2):
    """Only artificial uniform token IDs; counts full 4x2048 optimizer updates."""
    torch.set_num_threads(threads)
    device = resolve_device(device)
    records = []
    for size, context in [('main', 256), ('main', 512), ('main', 1024), ('main', 2048),
                          ('compact', 256), ('compact', 2048)]:
        torch.manual_seed(10101)
        model = VoynichTransformer(lc_config(size, context)).to(device).train()
        optimizer = lc_optimizer(model)
        generator = torch.Generator().manual_seed(10102)
        values = torch.randint(10, 112, (4, 2049), generator=generator).to(device)
        inputs, targets = values[:, :-1], values[:, 1:]
        for _ in range(warmup):
            lc_update(model, optimizer, inputs, targets, context=context)
        lc_sync(device)
        started = time.monotonic()
        for _ in range(updates):
            lc_update(model, optimizer, inputs, targets, context=context)
        lc_sync(device)
        elapsed = time.monotonic()-started
        item = {'size': size, 'context': context, 'parameters': model.parameter_count,
                'global_source_blocks': 4, 'global_positions_per_update': 8192,
                'microbatch_token_cap': 2048, 'warmup_updates': warmup, 'timed_updates': updates,
                'seconds_per_optimizer_update': elapsed / updates,
                'input_positions_per_second': updates * 8192 / elapsed, **lc_memory(device)}
        records.append(item)
        print(json.dumps(item), flush=True)
        del model, optimizer, values, inputs, targets
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    result = {'experiment': 'EXP-0010', 'environment': environment(), 'device': device,
              'artificial_inputs_only': True, 'test_evaluated': False, 'measurements': records,
              'memory_note': 'RSS peak is process lifetime; MPS allocation APIs are sampled, not peaks.',
              'forecast_training_seconds_21_runs_1200_updates': 1200 * 3 *
              (sum(r['seconds_per_optimizer_update'] for r in records) +
               next(r['seconds_per_optimizer_update'] for r in records
                    if r['size'] == 'main' and r['context'] == 2048))}
    write_json(output, result)
    return result


class LCBlocks:
    """Fixed same-page 2048-position blocks, shared by every context condition."""

    def __init__(self, data_dir, block_width=2048):
        self.data = PageWindows(data_dir, 'train', block_width)
        self.width = block_width
        self.inputs, targets, _ = self.data.batch(range(len(self.data.windows)), 'cpu')
        self.targets = targets[1]
        self.coordinates = [{'page_id': w.page_id, 'start': w.start, 'length': w.length}
                            for w in self.data.windows]

    def batch(self, indices, representation='original'):
        x, y = self.inputs[indices].clone(), self.targets[indices].clone()
        if representation == 'merge_locus_boundary':
            x[x.eq(self.data.tokenizer.line_id)] = self.data.tokenizer.space_id
        elif representation != 'original':
            raise ValueError('Unknown representation')
        return x, y


def lc_targets(validation, per_page=16, long_per_page=32, seed=10103):
    """Common target coordinates chosen without consulting losses or outputs."""
    rng = np.random.default_rng(seed)
    records = {}
    for page in validation.pages:
        seq = validation.sequences[page['page_id']]
        eligible = [p for p in range(64, len(seq)-1) if seq[p] not in validation.ignored_ids]
        long = [p for p in eligible if p >= 2048]
        for label, pool, cap in [('primary', eligible, per_page), ('full2048', long, long_per_page)]:
            selected = sorted(rng.choice(pool, min(len(pool), cap), replace=False).tolist())
            for position in selected:
                key = (page['page_id'], position)
                if key not in records:
                    target = seq[position]
                    records[key] = {'page_id': key[0], 'leaf_id': page['leaf_id'], 'position': position,
                                    'target': target, 'groups': [],
                                    'target_kind': 'space' if target == validation.tokenizer.space_id else
                                    'locus_boundary' if target == validation.tokenizer.line_id else 'glyph'}
                records[key]['groups'].append(label)
    if not records:
        raise ValueError('No eligible validation targets')
    return list(records.values())


def lc_coverage(records):
    report = {}
    for group in ['primary', 'full2048']:
        selected = [record for record in records if group in record['groups']]
        report[group] = {'targets': len(selected), 'page_count': len({r['page_id'] for r in selected}),
                         'leaf_count': len({r['leaf_id'] for r in selected}), 'contexts': {}}
        for context in [256, 512, 1024, 2048]:
            full = [r for r in selected if r['position'] >= context]
            report[group]['contexts'][str(context)] = {
                'targets_with_full_prefix': len(full), 'page_count_with_full_prefix': len({r['page_id'] for r in full}),
                'leaf_count_with_full_prefix': len({r['leaf_id'] for r in full}),
                'mean_prefix_units': float(np.mean([min(r['position'], context) for r in selected])) if selected else None}
    return report


def lc_prefix(validation, record, context, representation, corruption=None):
    position = record['position']
    values = validation.sequences[record['page_id']][max(0, position-context):position].copy()
    if not values or len(values) > context:
        raise ValueError('Invalid prefix')
    if representation == 'merge_locus_boundary':
        values = [validation.tokenizer.space_id if value == validation.tokenizer.line_id else value
                  for value in values]
    elif representation != 'original':
        raise ValueError('Unknown representation')
    if corruption == 'remote_shuffle_keep256' and len(values) > 256:
        # Per-coordinate seed is invariant to condition ordering and batch size.
        seed = int(hashlib.sha256(f"10104:{record['page_id']}:{position}".encode()).hexdigest()[:16], 16)
        values[:-256] = np.random.default_rng(seed).permutation(values[:-256]).tolist()
    elif corruption not in {None, 'remote_shuffle_keep256'}:
        raise ValueError('Unknown corruption')
    return values


@torch.no_grad()
def lc_score(model, validation, records, device, context, representation, batch_size=4, corruption=None):
    training = model.training
    model.eval()
    losses = np.zeros(len(records), dtype=float)
    lengths = np.zeros(len(records), dtype=int)
    # Right padding leaves selected earlier query logits causally unchanged.
    ordering = sorted(range(len(records)), key=lambda i: min(records[i]['position'], context))
    try:
        for start in range(0, len(ordering), batch_size):
            indices = ordering[start:start+batch_size]
            prefixes = [lc_prefix(validation, records[i], context, representation, corruption) for i in indices]
            width = max(map(len, prefixes))
            inputs = torch.full((len(indices), width), validation.tokenizer.pad_id, dtype=torch.long)
            for row, prefix in enumerate(prefixes):
                inputs[row, :len(prefix)] = torch.tensor(prefix)
            logits = model(inputs.to(device)).logits
            rows = torch.arange(len(indices), device=device)
            positions = torch.tensor([len(p)-1 for p in prefixes], device=device)
            targets = torch.tensor([records[i]['target'] for i in indices], device=device)
            values = F.cross_entropy(logits[rows, positions], targets, reduction='none') / math.log(2)
            losses[indices] = values.cpu().numpy()
            lengths[indices] = list(map(len, prefixes))
    finally:
        model.train(training)
    return {'losses_bits': losses.tolist(), 'prefix_lengths': lengths.tolist(),
            'scores': lc_group_scores(losses, records)}


def lc_group_scores(values, records):
    values = np.asarray(values)
    result = {}
    for group in ['primary', 'full2048']:
        ids = [i for i, r in enumerate(records) if group in r['groups']]
        if not ids:
            continue
        result[group] = {'bits_per_token': float(values[ids].mean()), 'scored_tokens': len(ids),
                         'pages': sorted({records[i]['page_id'] for i in ids}),
                         'leaves': sorted({records[i]['leaf_id'] for i in ids})}
        for kind in ['glyph', 'space', 'locus_boundary']:
            subset = [i for i in ids if records[i]['target_kind'] == kind]
            result[group][kind] = {'bits_per_token': float(values[subset].mean()), 'count': len(subset)} if subset else None
    return result


def lc_baselines(train, validation, records):
    model = NGram(train.tokenizer.vocab_size, order=5).fit(train)
    prior = np.array([model.probability([], i) for i in range(train.tokenizer.vocab_size)])
    losses = {key: [] for key in ['fivegram', 'unigram', 'fivegram_histogram_mix0.5',
                                 'fivegram_recency64_mix0.5', 'fivegram_recency256_mix0.5',
                                 'fivegram_recency1024_mix0.5']}
    for record in records:
        prefix = lc_prefix(validation, record, 2048, 'original')
        target = record['target']
        probability = model.probability(prefix, target)
        losses['fivegram'].append(-math.log2(probability))
        losses['unigram'].append(-math.log2(prior[target]))
        for tau, label in [(None, 'histogram'), (64, 'recency64'), (256, 'recency256'), (1024, 'recency1024')]:
            weights = np.ones(len(prefix)) if tau is None else np.exp(-np.arange(len(prefix)-1, -1, -1) / tau)
            counts = np.bincount(prefix, weights=weights, minlength=len(prior))
            local = (counts[target] + 32 * prior[target]) / (weights.sum() + 32)
            losses[f'fivegram_{label}_mix0.5'].append(-math.log2(.5 * probability + .5 * local))
    return {'losses_bits': losses, 'scores': {name: lc_group_scores(v, records) for name, v in losses.items()},
            'fit_split': 'train', 'selection': 'all fixed before scores; no coefficient optimization'}


def lc_plan(config):
    conditions = [('main', context, 'original') for context in [256, 512, 1024, 2048]]
    conditions += [('compact', context, 'original') for context in [256, 2048]]
    conditions += [('main', 2048, 'merge_locus_boundary')]
    # Rotation reduces systematic correlation between run order and context length.
    plan = []
    for j, seed in enumerate(config['seeds']):
        order = conditions[j:] + conditions[:j]
        for size, context, representation in order:
            plan.append({'size': size, 'context': context, 'representation': representation, 'seed': seed,
                         'name': f'{size}-c{context}-{representation}-seed{seed}'})
    return plan


def lc_weight_digest(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def lc_checkpoint(path, model, optimizer, step, config, condition, identity, sampler, tokens):
    state = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
             'model_config': model.config.to_dict(), 'optimizer': copy.deepcopy(optimizer.state_dict()),
             'step': step, 'training_config': config, 'condition': condition, 'corpus_identity': identity,
             'sampler_rng': sampler.get_state(), 'torch_rng': torch.get_rng_state(),
             'scored_training_tokens': tokens}
    if next(model.parameters()).device.type == 'mps':
        state['mps_rng'] = torch.mps.get_rng_state()
    temp = Path(path).with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def lc_train_one(condition, config, blocks, validation, records, identity, output, device, deadline, stop):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(condition['seed'])
    sampler = torch.Generator().manual_seed(condition['seed']+100)
    cfg = lc_config(condition['size'], condition['context'], blocks.data.tokenizer.vocab_size,
                    blocks.data.tokenizer.pad_id)
    model = VoynichTransformer(cfg).to(device)
    initial_digest = lc_weight_digest(model)
    optimizer = lc_optimizer(model)
    started = time.monotonic()
    manifest = {'condition': condition, 'config': config, 'model_config': cfg.to_dict(),
                'parameter_count': model.parameter_count, 'initial_weights_sha256': initial_digest,
                'corpus_identity': identity, 'environment': environment(), 'device': device,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'test_evaluated': False}
    write_json(output/'manifest.json', manifest)
    history = []
    counts = {'scored_training_tokens': 0, 'executed_input_positions': 0, 'attention_pair_slots': 0}
    sampled = hashlib.sha256()
    best, best_step, step = float('inf'), None, 0
    reason = 'step_budget'
    latest = lc_score(model, validation, records, device, cfg.context_length, condition['representation'])
    history.append({'step': 0, 'scores': latest['scores'], 'elapsed_seconds': time.monotonic()-started,
                    **counts, **lc_memory(device)})
    lc_checkpoint(output/'initial.pt', model, optimizer, 0, config, condition, identity, sampler, 0)
    for step in range(1, config['steps']+1):
        if stop['requested'] or time.monotonic() >= deadline:
            reason = 'signal' if stop['requested'] else 'track_time_budget'
            step -= 1
            break
        model.train()
        indices = torch.randint(len(blocks.inputs), (config['global_source_blocks'],), generator=sampler).tolist()
        sampled.update(np.asarray(indices, dtype='<i8').tobytes())
        x, y = blocks.batch(indices, condition['representation'])
        warmup = config['warmup_steps']
        progress = (step-warmup) / max(1, config['steps']-warmup)
        factor = step / warmup if step <= warmup else .1+.9*.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:
            group['lr'] = config['learning_rate'] * factor
        measured = lc_update(model, optimizer, x.to(device), y.to(device), context=cfg.context_length,
                             micro_tokens=config['microbatch_tokens'])
        counts['scored_training_tokens'] += measured['scored_tokens']
        for key in ['executed_input_positions', 'attention_pair_slots']:
            counts[key] += measured[key]
        if step % config['eval_interval'] == 0 or step == config['steps']:
            lc_sync(device)
            latest = lc_score(model, validation, records, device, cfg.context_length, condition['representation'])
            item = {'step': step, 'scores': latest['scores'], 'elapsed_seconds': time.monotonic()-started,
                    **counts, **lc_memory(device)}
            history.append(item)
            if latest['scores']['primary']['bits_per_token'] < best:
                best, best_step = latest['scores']['primary']['bits_per_token'], step
                lc_checkpoint(output/'best.pt', model, optimizer, step, config, condition, identity, sampler,
                              counts['scored_training_tokens'])
            lc_checkpoint(output/'last.pt', model, optimizer, step, config, condition, identity, sampler,
                          counts['scored_training_tokens'])
            write_json(output/'history.json', history)
            print(json.dumps({'experiment': 'EXP-0010', 'run': condition['name'], **item}), flush=True)
        if lc_memory(device)['process_max_rss_bytes'] > config['max_process_rss_gib'] * 2**30:
            reason = 'process_rss_budget'
            break
    lc_checkpoint(output/'last.pt', model, optimizer, step, config, condition, identity, sampler,
                  counts['scored_training_tokens'])
    if step != history[-1]['step']:
        latest = lc_score(model, validation, records, device, cfg.context_length, condition['representation'])
    final = {'original': latest}
    if cfg.context_length > 256 and reason == 'step_budget':
        final['cap256'] = lc_score(model, validation, records, device, 256, condition['representation'])
        final['remote_shuffle_keep256'] = lc_score(model, validation, records, device, cfg.context_length,
                                                   condition['representation'], corruption='remote_shuffle_keep256')
    summary = {'condition': condition, 'completed_steps': step, 'stop_reason': reason, 'best_step': best_step,
               'best_primary_validation_bits': None if best_step is None else best, 'final': final,
               'elapsed_seconds': time.monotonic()-started, **counts, 'parameter_count': model.parameter_count,
               'sampled_block_indices_sha256': sampled.hexdigest(), 'initial_weights_sha256': initial_digest,
               'last_checkpoint_sha256': digest(output/'last.pt'), 'test_evaluated': False, **lc_memory(device)}
    write_json(output/'history.json', history)
    write_json(output/'summary.json', summary)
    del model, optimizer
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    return summary


def lc_run(config_path, output_root, device='mps', data_dir='data/processed/zl3b'):
    config = json.loads(Path(config_path).read_text())
    torch.set_num_threads(config['threads'])
    device = resolve_device(device)
    env = environment()
    if env['git_dirty']:
        raise ValueError('Commit and publish source/registration before manuscript fitting')
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists() or (output/'runs').exists():
        raise ValueError('Use a fresh output directory; no implicit resume or overwrites')
    identity = corpus_identity(data_dir)
    blocks = LCBlocks(data_dir, config['source_block_width'])
    validation = PageWindows(data_dir, 'validation', 2048)
    records = lc_targets(validation, config['primary_targets_per_page'], config['long_targets_per_page'])
    plan = lc_plan(config)
    started = time.monotonic()
    deadline = started + config['max_wall_seconds']
    stop = {'requested': False}
    old_handlers = {}
    for sig in [signal.SIGTERM, signal.SIGINT]:
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda signum, frame: stop.update(requested=True))
    write_json(output/'manifest.json', {'experiment': 'EXP-0010', 'config': config, 'environment': env,
               'command': sys.argv, 'device': device, 'corpus_identity': identity, 'plan': plan,
               'config_sha256': digest(config_path), 'targets': records, 'train_blocks': blocks.coordinates,
               'target_coverage': lc_coverage(records),
               'training_scorable_positions_per_block': blocks.targets.ne(-100).sum(-1).tolist(),
               'started_utc': datetime.now(timezone.utc).isoformat(), 'test_evaluated': False,
               'allocator_environment': {key: os.environ.get(key) for key in
                                        ['PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO']}})
    summaries = []
    try:
        write_json(output/'baselines.json', lc_baselines(blocks.data, validation, records))
        for condition in plan:
            if time.monotonic() >= deadline or stop['requested']:
                break
            summary = lc_train_one(condition, config, blocks, validation, records, identity,
                                   output/'runs'/condition['name'], device, deadline, stop)
            summaries.append(summary)
            write_json(output/'progress.json', {'runs': summaries, 'elapsed_seconds': time.monotonic()-started})
            if summary['stop_reason'] != 'step_budget':
                break
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    end_env = environment()
    complete = {'experiment': 'EXP-0010', 'runs_completed': len(summaries), 'runs_planned': len(plan),
                'all_runs_finished': len(summaries) == len(plan) and
                all(s['stop_reason'] == 'step_budget' for s in summaries),
                'elapsed_seconds': time.monotonic()-started, 'test_evaluated': False,
                'source_unchanged': env['source_sha256'] == end_env['source_sha256'],
                'source_git_commit_unchanged': env['git_commit'] == end_env['git_commit'],
                'total_scored_training_tokens': sum(s['scored_training_tokens'] for s in summaries),
                'total_updates': sum(s['completed_steps'] for s in summaries), **lc_memory(device)}
    write_json(output/'completion.json', complete)
    return complete


def lc_archive(source, destination):
    """Audit completed local run data and export compact scores, never weights/text."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Archive destination must be new')
    manifest = json.loads((source/'manifest.json').read_text())
    completion = json.loads((source/'completion.json').read_text())
    if manifest['environment']['git_dirty'] or manifest['test_evaluated'] or completion['test_evaluated']:
        raise ValueError('Training provenance or holdout guard failed')
    if not completion['source_unchanged'] or not completion['source_git_commit_unchanged']:
        raise ValueError('Source changed during training')
    runs = {}
    for directory in sorted((source/'runs').glob('*')):
        if not (directory/'summary.json').exists():
            continue
        summary = json.loads((directory/'summary.json').read_text())
        run_manifest = json.loads((directory/'manifest.json').read_text())
        if summary['last_checkpoint_sha256'] != digest(directory/'last.pt'):
            raise ValueError('Checkpoint digest mismatch')
        if run_manifest['environment']['git_dirty'] or run_manifest['environment']['git_commit'] != manifest['environment']['git_commit']:
            raise ValueError('Source drift or dirty training run')
        if run_manifest['corpus_identity'] != manifest['corpus_identity'] or summary['test_evaluated']:
            raise ValueError('Corpus/holdout mismatch')
        runs[directory.name] = summary
    records = manifest['targets']
    comparisons = {}
    main_cases = [('main', c, 'original') for c in [512, 1024, 2048]]
    main_cases += [('compact', 2048, 'original'), ('main', 2048, 'merge_locus_boundary')]
    for size, context, representation in main_cases:
        label = f'{size}-c{context}-{representation}'
        paired = []
        for seed in manifest['config']['seeds']:
            treatment = runs.get(f'{label}-seed{seed}')
            reference_context = 2048 if representation != 'original' else 256
            reference = runs.get(f'{size}-c{reference_context}-original-seed{seed}')
            if not treatment or not reference or treatment['stop_reason'] != 'step_budget' or reference['stop_reason'] != 'step_budget':
                continue
            if treatment['scored_training_tokens'] != reference['scored_training_tokens'] or treatment['sampled_block_indices_sha256'] != reference['sampled_block_indices_sha256']:
                raise ValueError('Matched-exposure control failed')
            if treatment['initial_weights_sha256'] != reference['initial_weights_sha256']:
                raise ValueError('Same-capacity initialization differed')
            a = np.asarray(reference['final']['original']['losses_bits'])
            b = np.asarray(treatment['final']['original']['losses_bits'])
            item = {'seed': seed, 'gain_bits_reference_minus_treatment': {}}
            for group in ['primary', 'full2048']:
                indices = [i for i, r in enumerate(records) if group in r['groups']]
                if indices:
                    measured = paired_summary((a-b)[indices], [records[i] for i in indices], seed=10105)
                    if group == 'full2048':
                        measured.pop('equal_leaf_bootstrap_95', None)
                        measured['uncertainty_note'] = 'Only two leaf groups; no interval-based generalization claim.'
                    item['gain_bits_reference_minus_treatment'][group] = measured
            if context > 256:
                cap = np.asarray(treatment['final']['cap256']['losses_bits'])
                indices = [i for i, r in enumerate(records) if 'primary' in r['groups']]
                item['within_treatment_gain_full_vs_cap256'] = float((cap-b)[indices].mean())
            paired.append(item)
        means = [p['gain_bits_reference_minus_treatment']['primary']['sample_mean'] for p in paired]
        comparisons[label] = {'per_seed': paired, 'mean_gain_primary_bits': float(np.mean(means)) if means else None,
                              'registered_context_signal': representation == 'original' and len(paired) == 3 and
                              all(value > 0 for value in means) and float(np.mean(means)) >= .02 and
                              all(p['within_treatment_gain_full_vs_cap256'] > 0 for p in paired)}
    for filename in ['manifest.json', 'completion.json', 'baselines.json']:
        write_json(destination/filename, json.loads((source/filename).read_text()))
    for name, summary in runs.items():
        write_json(destination/'runs'/name/'summary.json', summary)
        write_json(destination/'runs'/name/'history.json', json.loads((source/'runs'/name/'history.json').read_text()))
        write_json(destination/'runs'/name/'manifest.json', json.loads((source/'runs'/name/'manifest.json').read_text()))
    report = {'experiment': 'EXP-0010', 'completion': completion, 'comparisons': comparisons,
              'manifest_sha256': digest(source/'manifest.json'), 'test_evaluated': False,
              'limitations': ['Development validation, repeatedly exposed in prior work; not a final test.',
                              'Full2048 coverage has only four pages from two leaf groups.',
                              'Exposure matched; computation differs and concurrent wall time is noisy.',
                              'Locus separators are transcription record boundaries, not uniformly physical lines.',
                              'No translations or historical production mechanism inferred.']}
    write_json(destination/'summary.json', report)
    return report


def lc_main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    benchmark = sub.add_parser('benchmark')
    benchmark.add_argument('--output', default='outputs/EXP-0010/benchmark.json')
    benchmark.add_argument('--device', default='mps')
    benchmark.add_argument('--warmup', type=int, default=5)
    benchmark.add_argument('--updates', type=int, default=10)
    benchmark.add_argument('--threads', type=int, default=2)
    run = sub.add_parser('run')
    run.add_argument('--config', default='configs/exp0010.json')
    run.add_argument('--output-root', default='outputs/EXP-0010/campaign')
    run.add_argument('--data', default='data/processed/zl3b')
    run.add_argument('--device', default='mps')
    archive = sub.add_parser('archive')
    archive.add_argument('--source', default='outputs/EXP-0010/campaign')
    archive.add_argument('--destination', default='results/EXP-0010')
    args = parser.parse_args()
    if args.action == 'benchmark':
        lc_benchmark(args.output, args.device, args.warmup, args.updates, args.threads)
    elif args.action == 'run':
        result = lc_run(args.config, args.output_root, args.device, args.data)
        print(json.dumps(result, indent=2))
        if not result['all_runs_finished'] or not result['source_unchanged'] or not result['source_git_commit_unchanged']:
            raise SystemExit(2)
    elif args.action == 'archive':
        print(json.dumps(lc_archive(args.source, args.destination), indent=2))


if __name__ == '__main__':
    lc_main()
