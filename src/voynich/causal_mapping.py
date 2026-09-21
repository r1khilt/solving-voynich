"""EXP-0009: prefix-only causal maps with independently selected future-effect tests."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .runtime import PageWindows, corpus_identity, digest, environment, resolve_device, write_json
from .synthetic import ALPHABET, SIGNAL_PROBABILITY, filter_sequence, generate_page
from .tokenizer import EVATokenizer
from .train import load_checkpoint


WIDTH = 128
HORIZONS = (1, 2, 4, 8)
REGIONS = ('final', 'recent8', 'remote120', 'all128')


@dataclass(frozen=True)
class MappingSite:
    layer: int
    kind: str
    region: str
    head: int | None = None

    @property
    def hook(self):
        suffix = {'head': 'attn.result', 'mlp': 'mlp.out', 'residual': 'resid_post'}[self.kind]
        return f'blocks.{self.layer}.{suffix}'

    @property
    def name(self):
        head = f'{self.head}' if self.head is not None else ''
        return f'L{self.layer}-{self.kind}{head}-{self.region}'

    @property
    def positions(self):
        bounds = {'final': (127, 128), 'recent8': (120, 128), 'remote120': (0, 120), 'all128': (0, 128)}
        return slice(*bounds[self.region])


def mapping_sites(model):
    return [MappingSite(layer, kind, region, head)
            for layer in range(model.config.n_layers)
            for kind in ('head', 'mlp', 'residual')
            for head in (range(model.config.n_heads) if kind == 'head' else (None,))
            for region in REGIONS]


def mapping_patch(donor, site, *, mode='donor', seed=0, basis=None):
    """Patch only prefix coordinates; never patch a future query position."""
    def apply(value):
        result = value.clone()
        index = (slice(None), site.positions, site.head) if site.kind == 'head' else (slice(None), site.positions)
        base, wanted = value[index], donor[index]
        delta = wanted-base
        if mode == 'zero':
            changed = torch.zeros_like(base)
        elif mode == 'random':
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(base.shape, generator=generator, dtype=base.dtype).to(base.device)
            if basis is not None:
                noise = (noise@basis)@basis.T
            delta = noise * delta.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            changed = base+delta
        elif mode == 'readout':
            if basis is None:
                raise ValueError('Readout projection requires a basis')
            changed = base+(delta@basis)@basis.T
        elif mode == 'donor':
            changed = wanted
        else:
            raise ValueError('Unknown intervention mode')
        result[index] = changed
        return result
    return apply


def continuation_from_belief(rng, posterior, length=7):
    state = int(rng.choice(4, p=posterior))
    chars = []
    for _ in range(length):
        if rng.random() < SIGNAL_PROBABILITY:
            state = (state+1) % 4
            symbol = 2*state+int(rng.integers(2))
        else:
            symbol = int(rng.integers(8))
        chars.append(ALPHABET[symbol])
    return ''.join(chars)


def synthetic_pairs(texts, count, seed):
    """Pair by generator beliefs, not neural activations, and use a shared continuation."""
    rng = np.random.default_rng(seed)
    continuation_rng = np.random.default_rng(seed+1000000)
    posterior = np.stack([filter_sequence(text)[0][-1] for text in texts])
    labels, confidence = posterior.argmax(1), posterior.max(1)
    last = np.array([text[-1] for text in texts])
    rows = []
    for recipient in range(len(texts)):
        if confidence[recipient] < .6:
            continue
        eligible = (last == last[recipient]) & (confidence >= .6)
        correct = np.flatnonzero(eligible & (labels != labels[recipient]))
        wrong = np.flatnonzero(eligible & (labels == labels[recipient]) & (np.arange(len(texts)) != recipient))
        if len(correct) == 0 or len(wrong) == 0:
            continue
        donor = int(rng.choice(correct))
        wrong_id = int(wrong[np.abs(posterior[wrong]-posterior[recipient]).sum(1).argmin()])
        suffix = continuation_from_belief(continuation_rng, posterior[donor])
        desired = filter_sequence(texts[donor]+suffix)[2]
        rows.append({'recipient': texts[recipient], 'donor': texts[donor], 'wrong': texts[wrong_id],
                     'suffix': suffix, 'oracle': desired[np.array(HORIZONS)+WIDTH-2].tolist(),
                     'pool_indexes': [recipient, donor, wrong_id], 'group': str(recipient),
                     'posterior_l1_wrong_vs_recipient': float(np.abs(posterior[wrong_id]-posterior[recipient]).sum())})
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f'Insufficient qualifying synthetic pairs: {len(rows)} / {count}')
    return rows


def prepare_synthetic(root):
    old = []
    for path in Path('data/processed/synthetic_cycle_null').glob('*.jsonl'):
        old.extend(json.loads(line)['text'] for line in path.read_text().splitlines())
    for path in Path('data/processed/alignment_v1').glob('*.jsonl'):
        old.extend(json.loads(line)['text'] for line in path.read_text().splitlines())
    old_windows = {text[i:i+WIDTH] for text in old for i in range(max(0, len(text)-WIDTH+1))}
    seen = set()
    splits = {}
    for split, count, pool_count, seed in [('discovery', 512, 4096, 9201), ('confirmation', 1024, 8192, 9202)]:
        rng = np.random.default_rng(seed)
        texts = [generate_page(rng, WIDTH, 'cycle_null')[0] for _ in range(pool_count)]
        if len(set(texts)) != len(texts) or any(text in seen or text in old_windows for text in texts):
            raise ValueError('Fresh synthetic prefix overlap detected')
        seen.update(texts)
        rows = synthetic_pairs(texts, count, seed+10)
        path = root/f'synthetic-{split}.json'
        write_json(path, rows)
        splits[split] = {'path': str(path), 'sha256': digest(path), 'pairs': count, 'pool_count': pool_count,
                         'seed': seed, 'unique_donors': len({row['pool_indexes'][1] for row in rows})}
    return splits


def prepare_manuscript(root):
    data = PageWindows('data/processed/zl3b', 'validation', 256)
    train = PageWindows('data/processed/zl3b', 'train', 256)
    rng = np.random.default_rng(9220)
    leaves = sorted({page['leaf_id'] for page in data.pages})
    shuffled_leaves = rng.permutation(leaves).tolist()
    discovery = set(shuffled_leaves[:len(leaves)//2])
    donors = {}
    for page in train.pages:
        seq = train.sequences[page['page_id']]
        for end in range(WIDTH+1, len(seq)-1, 16):
            donors.setdefault(seq[end-1], []).append({'prefix': seq[end-WIDTH:end],
                                                     'page_id': page['page_id'], 'end_position': end})
    records = {'discovery': [], 'confirmation': []}
    ignored = data.ignored_ids | {data.tokenizer.eos_id, data.tokenizer.bos_id}
    for page in data.pages:
        seq = data.sequences[page['page_id']]
        eligible = [p for p in range(WIDTH+1, len(seq)-max(HORIZONS))
                    if all(seq[p+h-1] not in ignored for h in HORIZONS) and seq[p-1] in donors]
        chosen = sorted(rng.choice(eligible, min(64, len(eligible)), replace=False).tolist())
        split = 'discovery' if page['leaf_id'] in discovery else 'confirmation'
        for p in chosen:
            clean = seq[p-WIDTH:p]
            corrupt = clean.copy()
            corrupt[:-16] = rng.permutation(corrupt[:-16]).tolist()
            candidates = donors[clean[-1]]
            wrong = candidates[int(rng.integers(len(candidates)))]
            records[split].append({'recipient': corrupt, 'donor': clean, 'wrong': wrong['prefix'],
                                   'suffix': seq[p:p+7], 'targets': [seq[p+h-1] for h in HORIZONS],
                                   'page_id': page['page_id'], 'position': p, 'group': page['leaf_id'],
                                   'wrong_donor_page_id': wrong['page_id'],
                                   'wrong_donor_end_position': wrong['end_position']})
    result = {}
    for split, rows in records.items():
        if not rows:
            raise ValueError('Empty manuscript partition')
        path = root/f'manuscript-{split}.json'
        write_json(path, rows)
        result[split] = {'path': str(path), 'sha256': digest(path), 'pairs': len(rows),
                         'leaf_ids': sorted({row['group'] for row in rows})}
    return result


def readout_basis(model, letter_ids=None):
    weight = model.unembedding.weight.detach().cpu()*model.final_norm.weight.detach().cpu()[None]
    if letter_ids is not None:
        weight = weight[letter_ids].reshape(4, 2, -1).mean(1)
    weight = weight-weight.mean(0)
    return torch.linalg.svd(weight, full_matrices=False).Vh[:3].T.to(next(model.parameters()).device)


def head_basis(model, site):
    if site.kind != 'head':
        return None
    width = model.config.d_model//model.config.n_heads
    weight = model.blocks[site.layer].attn.out.weight[:, site.head*width:(site.head+1)*width]
    return torch.linalg.qr(weight.detach().cpu(), mode='reduced').Q.to(weight.device)


def make_inputs(rows, tokenizer, device):
    result = {}
    for kind in ('recipient', 'donor', 'wrong'):
        values = []
        for row in rows:
            prefix, suffix = row[kind], row['suffix']
            if isinstance(prefix, str):
                values.append(tokenizer.encode(prefix+suffix, add_bos=False, add_eos=False))
            else:
                values.append(prefix+suffix)
        result[kind] = torch.tensor(values, dtype=torch.long, device=device)
    return result


def prediction_scores(logits, rows, letters, donor_logp):
    selected = logits[:, [WIDTH+h-2 for h in HORIZONS]]
    if letters is not None:
        selected = selected[:, :, letters]
    logp = selected.log_softmax(-1)
    donor_kl = (donor_logp.exp()*(donor_logp-logp)).sum(-1)/math.log(2)
    if letters is not None:
        desired = torch.tensor([row['oracle'] for row in rows], dtype=logp.dtype, device=logp.device)
        primary = (desired*(desired.clamp_min(1e-12).log()-logp)).sum(-1)/math.log(2)
    else:
        target = torch.tensor([row['targets'] for row in rows], device=logp.device)
        primary = -logp.gather(-1, target[..., None]).squeeze(-1)/math.log(2)
    return torch.stack((primary, donor_kl), -1).cpu().numpy()


def summarize_scores(values, rows):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite causal result')
    groups = np.array([row['group'] for row in rows])
    unique = sorted(set(groups))
    grouped = np.stack([values[groups == group].mean(0) for group in unique])
    return {'mean': values.mean(0).tolist(), 'equal_group_mean': grouped.mean(0).tolist(),
            'group_count': len(unique), 'examples': len(rows),
            'group_means': dict(zip(unique, grouped.tolist(), strict=True)) if len(unique) <= 20 else None}


@torch.no_grad()
def evaluate_mapping(model, rows, tokenizer, sites, device, *, selected=None, deadline=float('inf')):
    synthetic = 'oracle' in rows[0]
    letters = [tokenizer.token_to_id[c] for c in ALPHABET] if synthetic else None
    hooks = sorted({site.hook for site in sites} | {'embed'})
    last_site = MappingSite(model.config.n_layers-1, 'residual', 'final')
    hooks = sorted(set(hooks) | {last_site.hook})
    basis = readout_basis(model, letters)
    selected_basis = head_basis(model, selected) if selected is not None else None
    values = {}
    controls = {'identity_max_logit_error': 0., 'embedding_restore_max_logit_error': 0.,
                'full_final_prefix_restore_max_logit_error': 0.,
                'late_prefix_future_max_logit_error': 0.}
    for start in range(0, len(rows), 32):
        if time.monotonic() > deadline:
            raise TimeoutError('EXP-0009 reached its registered runtime cap')
        batch = rows[start:start+32]
        x = make_inputs(batch, tokenizer, device)
        clean = model(x['donor'], cache_names=hooks)
        base = model(x['recipient'], cache_names=hooks)
        wrong = model(x['wrong'], cache_names=hooks) if selected is not None else None
        donor_logits = clean.logits[:, [WIDTH+h-2 for h in HORIZONS]]
        donor_logp = (donor_logits[:, :, letters] if letters is not None else donor_logits).log_softmax(-1)
        def record(label, logits):
            values.setdefault(label, []).append(prediction_scores(logits, batch, letters, donor_logp))
        record('recipient', base.logits)
        record('donor', clean.logits)
        for site in sites:
            patched = model(x['recipient'], interventions={site.hook: mapping_patch(clean.cache[site.hook], site)})
            record(site.name+'/donor', patched.logits)
        if selected is not None:
            for mode in ('wrong', 'random0', 'random1', 'random2', 'random3', 'zero'):
                noise = mode.startswith('random')
                donor = wrong.cache[selected.hook] if mode == 'wrong' else clean.cache[selected.hook]
                patch = mapping_patch(donor, selected, mode='random' if noise else 'zero' if mode == 'zero' else 'donor',
                                      seed=9240+start+(int(mode[-1])*100000 if noise else 0), basis=selected_basis)
                # Zeroing measures necessity on clean donor input, not corruption restoration.
                inputs = x['donor'] if mode == 'zero' else x['recipient']
                out = model(inputs, interventions={selected.hook: patch})
                record(selected.name+'/'+mode, out.logits)
        out = model(x['recipient'], interventions={last_site.hook: mapping_patch(clean.cache[last_site.hook], last_site,
                                                                               mode='readout', basis=basis)})
        record('late_readout', out.logits)
        future_error = float((out.logits[:, WIDTH:]-base.logits[:, WIDTH:]).abs().max().cpu())
        controls['late_prefix_future_max_logit_error'] = max(controls['late_prefix_future_max_logit_error'], future_error)
        if start == 0:
            first = sites[0]
            identical = model(x['recipient'], interventions={first.hook: mapping_patch(base.cache[first.hook], first)})
            identity_error = float((identical.logits-base.logits).abs().max().cpu())
            def embed_patch(value):
                changed = value.clone()
                changed[:, :WIDTH] = clean.cache['embed'][:, :WIDTH]
                return changed
            restored = model(x['recipient'], interventions={'embed': embed_patch})
            restore_error = float((restored.logits-clean.logits).abs().max().cpu())
            controls['identity_max_logit_error'] = identity_error
            controls['embedding_restore_max_logit_error'] = restore_error
            final_restore = model(x['recipient'], interventions={last_site.hook: mapping_patch(clean.cache[last_site.hook], last_site)})
            controls['full_final_prefix_restore_max_logit_error'] = float(
                (final_restore.logits[:, WIDTH-1]-clean.logits[:, WIDTH-1]).abs().max().cpu())
        if max(controls.values()) > 1e-5:
            raise ValueError(f'Causal control failure: {controls}')
    arrays = {label: np.concatenate(parts) for label, parts in values.items()}
    return {'conditions': {label: summarize_scores(array, rows) for label, array in arrays.items()},
            'controls': controls, 'horizons': list(HORIZONS),
            'metric_columns': ['oracle_kl_bits' if synthetic else 'target_bits', 'donor_model_kl_bits']}, arrays


def select_site(report, sites):
    eligible = [site for site in sites if site.layer == 0 and site.kind in ('head', 'mlp')]
    # Future positions 2 and 4 only. Horizon 1 never drives selection.
    return min(eligible, key=lambda site: (np.array(report['conditions'][site.name+'/donor']['equal_group_mean'])[1:3, 0].mean(), site.name))


def confirmation_decision(report, selected, rows, arrays):
    donor = arrays[selected.name+'/donor'][:, :, 0]
    baseline = arrays['recipient'][:, :, 0]
    random = np.mean([arrays[selected.name+f'/random{i}'][:, :, 0] for i in range(4)], axis=0)
    wrong = arrays[selected.name+'/wrong'][:, :, 0]
    groups = np.array([row['group'] for row in rows])
    def group_mean(value):
        return np.mean([value[groups == group].mean(0) for group in sorted(set(groups))], axis=0)
    gain, random_margin, wrong_margin = [group_mean(value-donor) for value in (baseline, random, wrong)]
    passes = bool(np.all(gain[1:3] >= .02) and np.all(random_margin[1:3] >= .01) and np.all(wrong_margin[1:3] >= .01))
    return {'selected_site': asdict(selected), 'selection_metric': 'Discovery equal-group primary loss averaged over horizons 2 and 4',
            'confirmation_gain_by_horizon': gain.tolist(), 'random_margin_by_horizon': random_margin.tolist(),
            'wrong_donor_margin_by_horizon': wrong_margin.tolist(), 'meets_registered_practical_threshold': passes,
            'meaning': 'Future causal mediation under this intervention, not a complete state machine or historical circuit.'}


def run(output_root, device, max_seconds=7200):
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh output directory')
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started+max_seconds
    provenance = environment()
    if provenance['git_dirty']:
        raise ValueError('Publish clean source before running EXP-0009')
    identities = {name: corpus_identity(f'data/processed/{name}') for name in ('zl3b', 'synthetic_cycle_null')}
    data_root = root/'data'
    data_root.mkdir()
    data = {'synthetic': prepare_synthetic(data_root), 'manuscript': prepare_manuscript(data_root)}
    manifest = {'experiment': 'EXP-0009', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'environment': provenance, 'device': device, 'max_seconds': max_seconds, 'corpus_identities': identities,
                'data': data, 'test_evaluated': False, 'paid_api_calls': 0, 'model_training_updates': 0,
                'overlap_audit_inputs_sha256': {str(path): digest(path)
                    for directory in ('data/processed/synthetic_cycle_null', 'data/processed/alignment_v1')
                    for path in sorted(Path(directory).glob('*.jsonl'))},
                'fresh_prefix_audit': 'All new synthetic 128-character prefixes are unique across pools and absent from same-length substrings of EXP-0004 cycle corpus and EXP-0006 pools.'}
    write_json(root/'manifest.json', manifest)
    jobs = [('synthetic', seed, f'outputs/EXP-0004/cycle_null-seed{seed}/best.pt') for seed in (42, 43, 44)]
    jobs += [('synthetic_untrained', 42, 'outputs/EXP-0004/cycle_null-seed42/initial.pt')]
    jobs += [('manuscript', seed, f'outputs/EXP-0002/small-seed{seed}/best.pt') for seed in (42, 43, 44)]
    results = []
    for kind, seed, checkpoint in jobs:
        job_start = time.monotonic()
        model, payload = load_checkpoint(checkpoint, device)
        model.eval().requires_grad_(False)
        before = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        corpus = 'zl3b' if kind == 'manuscript' else 'synthetic_cycle_null'
        if payload['corpus_identity'] != identities[corpus]:
            raise ValueError('Frozen model corpus provenance differs')
        tokenizer = EVATokenizer.load(Path(f'data/processed/{corpus}')/'tokenizer.json')
        splits = data['manuscript' if kind == 'manuscript' else 'synthetic']
        rows = {}
        for split, spec in splits.items():
            if digest(spec['path']) != spec['sha256']:
                raise ValueError('Prepared data changed')
            rows[split] = json.loads(Path(spec['path']).read_text())
        sites = mapping_sites(model)
        discovery, discovery_arrays = evaluate_mapping(model, rows['discovery'], tokenizer, sites, device, deadline=deadline)
        selected = select_site(discovery, sites)
        # Persist the frozen selection before any confirmation scoring.
        label = f'{kind}-seed{seed}'
        write_json(root/f'{label}-selection.json', {'selected': asdict(selected), 'discovery': discovery,
                                                  'checkpoint_sha256': digest(checkpoint)})
        confirmation, confirmation_arrays = evaluate_mapping(model, rows['confirmation'], tokenizer, sites, device,
                                                             selected=selected, deadline=deadline)
        decision = confirmation_decision(confirmation, selected, rows['confirmation'], confirmation_arrays)
        if any(not torch.equal(before[key], value.cpu()) for key, value in model.state_dict().items()):
            raise ValueError('Frozen backbone changed')
        for split, arrays in [('discovery', discovery_arrays), ('confirmation', confirmation_arrays)]:
            np.savez_compressed(root/f'{label}-{split}-per-example.npz', **arrays)
        report = {'experiment': 'EXP-0009', 'kind': kind, 'seed': seed, 'checkpoint': checkpoint,
                  'checkpoint_sha256': digest(checkpoint), 'frozen_backbone_verified': True,
                  'discovery': discovery, 'confirmation': confirmation, 'decision': decision,
                  'elapsed_seconds': time.monotonic()-job_start, 'test_evaluated': False}
        write_json(root/f'{label}.json', report)
        results.append({'kind': kind, 'seed': seed, 'selected': selected.name, **decision,
                        'report_path': str(root/f'{label}.json'), 'report_sha256': digest(root/f'{label}.json')})
        print(json.dumps({'finished': label, 'selected': selected.name, 'decision': decision,
                          'elapsed_seconds': report['elapsed_seconds']}), flush=True)
        del model, before, discovery_arrays, confirmation_arrays
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    completion = {'experiment': 'EXP-0009', 'jobs_completed': len(results), 'results': results,
                  'elapsed_seconds': time.monotonic()-started, 'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'test_evaluated': False, 'paid_api_calls': 0,
                  'synthetic_all_trained_seeds_pass': all(r['meets_registered_practical_threshold'] for r in results if r['kind'] == 'synthetic'),
                  'manuscript_all_seeds_pass': all(r['meets_registered_practical_threshold'] for r in results if r['kind'] == 'manuscript')}
    write_json(root/'completion.json', completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='outputs/EXP-0009')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--max-seconds', type=int, default=7200)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 7200:
        parser.error('Runtime must be between 1 and 7200 seconds')
    torch.set_num_threads(2)
    device = resolve_device(args.device)
    if device == 'mps':
        torch.mps.set_per_process_memory_fraction(.23)
    try:
        run(args.output_root, device, args.max_seconds)
    except Exception as error:
        write_json(Path(args.output_root)/'failure.json', {'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
