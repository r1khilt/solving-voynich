"""EXP-0006: supervised rank-three interchange alignment with frozen language models."""

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from .model import ActivationContext
from .runtime import corpus_identity, digest, environment, resolve_device, write_json
from .synthetic import ALPHABET, filter_sequence, generate_page
from .tokenizer import EVATokenizer
from .train import load_checkpoint


POOL_SETTINGS = {'fit': (4096, 128), 'validation': (1024, 128), 'test128': (1024, 128),
                 'test96': (512, 96), 'test192': (512, 192)}


def make_pairs(records, limit, seed, preserve=False):
    posterior = np.array([r['posterior'] for r in records])
    labels, confidence = posterior.argmax(-1), posterior.max(-1)
    last = np.array([r['text'][-1] for r in records])
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(len(records)):
        if confidence[i] < .6:
            continue
        eligible = (last == last[i]) & (confidence >= .6)
        eligible &= (labels == labels[i]) if preserve else (labels != labels[i])
        eligible[i] = False
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            continue
        if preserve:
            distance = np.abs(posterior[candidates]-posterior[i]).sum(-1)
            if distance.min() > .05:
                continue
            donor = int(candidates[distance.argmin()])
        else:
            donor = int(rng.choice(candidates))
        pairs.append([i, donor])
        if len(pairs) == limit:
            break
    return pairs


def prepare(root):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use fresh alignment dataset directory')
    root.mkdir(parents=True, exist_ok=True)
    original = corpus_identity('data/processed/synthetic_cycle_null')
    old_pages = []
    for split in ['train', 'validation', 'test']:
        old_pages.extend(json.loads(line)['text'] for line in Path(f'data/processed/synthetic_cycle_null/{split}.jsonl').read_text().splitlines())
    old_substrings = {length: {p[start:start+length] for p in old_pages for start in range(len(p)-length+1)}
                      for length in [96, 128, 192]}
    children = np.random.SeedSequence(6201).spawn(len(POOL_SETTINGS))
    seen, datasets, pairs = set(), {}, {}
    for index, ((name, (count, length)), child) in enumerate(zip(POOL_SETTINGS.items(), children, strict=True)):
        rng = np.random.default_rng(child)
        rows = []
        for i in range(count):
            text, states, _ = generate_page(rng, length, 'cycle_null')
            if text in seen or text in old_substrings[length]:
                raise ValueError('Context duplication across fresh/old pools')
            seen.add(text)
            posteriors, _, predictions = filter_sequence(text)
            rows.append({'id': f'{name}-{i:05d}', 'text': text, 'actual_final_state': states[-1],
                         'posterior': posteriors[-1].tolist(), 'next_probs': predictions[-1].tolist()})
        path = root/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows))
        limit = 2048 if name == 'fit' else 512 if length == 128 else 256
        pairs[name] = make_pairs(rows, limit, 6202+index)
        if len(pairs[name]) < 128:
            raise ValueError('Too few qualifying pairs for registered experiment')
        if name == 'test128':
            pairs['preservation'] = make_pairs(rows, 256, 6290, preserve=True)
        datasets[name] = {'count': count, 'length': length, 'sha256': digest(path), 'pair_count': len(pairs[name])}
    write_json(root/'pairs.json', pairs)
    manifest = {'experiment': 'EXP-0006', 'generator': 'cycle_null', 'master_seed': 6201,
                'old_corpus_identity': original, 'pools': datasets, 'pairs_sha256': digest(root/'pairs.json'),
                'preservation_pair_count': len(pairs['preservation']),
                'overlap_audit': 'No repeated full fresh contexts; no fresh context equals any same-length substring of old structured synthetic corpus.',
                'label_access': 'Oracle probabilities supervise alignment; frozen LM received only text during original training.'}
    write_json(root/'manifest.json', manifest)
    return manifest


def load_data(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if digest(root/'pairs.json') != manifest['pairs_sha256']:
        raise ValueError('Pair provenance mismatch')
    data = {}
    for name, spec in manifest['pools'].items():
        path = root/f'{name}.jsonl'
        if digest(path) != spec['sha256']:
            raise ValueError('Alignment corpus provenance mismatch')
        data[name] = [json.loads(line) for line in path.read_text().splitlines()]
    return data, json.loads((root/'pairs.json').read_text()), manifest


def orthonormalize(value):
    # CPU QR is small and portable; MPS QR support need not be assumed.
    q, r = torch.linalg.qr(value.cpu(), mode='reduced')
    signs = r.diag().sign()
    signs[signs == 0] = 1
    return (q*signs).to(value.device)


def patch_residual(base, donor, basis, desired_norm=None):
    delta = donor[:, -1]-base[:, -1]
    displacement = (delta@basis)@basis.T
    if desired_norm is not None:
        displacement = displacement*desired_norm/displacement.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    result = base.clone()
    result[:, -1] = base[:, -1]+displacement
    return result, displacement


def resumed_logits(model, residual, site):
    if not 0 <= site < model.config.n_layers:
        raise ValueError('Invalid residual site')
    valid = torch.ones(residual.shape[:2], dtype=torch.bool, device=residual.device)
    ctx = ActivationContext(None, None)
    value = residual
    for index in range(site+1, model.config.n_layers):
        value = model.blocks[index](value, valid, ctx, f'blocks.{index}')
    return model.unembedding(model.final_norm(value[:, -1]))


@torch.no_grad()
def cache_records(model, records, tokenizer, device):
    sites = ['blocks.0.resid_post', 'blocks.1.resid_post']
    residuals = [[], []]
    log_probs = []
    parity = [0., 0.]
    letters = torch.tensor([tokenizer.token_to_id[c] for c in ALPHABET], device=device)
    for start in range(0, len(records), 64):
        batch = records[start:start+64]
        x = torch.tensor([tokenizer.encode(r['text'], add_eos=False) for r in batch], device=device)
        out = model(x, cache_names=sites)
        for site in [0, 1]:
            value = out.cache[sites[site]]
            if site == 1:
                value = value[:, -1:]
            if start == 0:
                error = float((resumed_logits(model, value, site)-out.logits[:, -1]).abs().max().cpu())
                parity[site] = error
                if error > 1e-5:
                    raise ValueError('Cached resumed-forward parity failed')
            residuals[site].append(value.cpu())
        log_probs.append(out.logits[:, -1, letters].log_softmax(-1).cpu())
    return {'residuals': [torch.cat(items) for items in residuals], 'log_probs': torch.cat(log_probs),
            'oracle': torch.tensor([r['next_probs'] for r in records], dtype=torch.float32),
            'resume_max_logit_errors': parity}


def batch_values(cache, pairs, site, device):
    indexes = np.asarray(pairs)
    return (cache['residuals'][site][indexes[:, 0]].to(device),
            cache['residuals'][site][indexes[:, 1]].to(device))


def fit_basis(model, caches, pairs, site, letters, seed, device, shuffled=False, steps=400):
    rng = np.random.default_rng(seed)
    initial = torch.randn(model.config.d_model, 3, generator=torch.Generator().manual_seed(seed))
    basis = torch.nn.Parameter(orthonormalize(initial).to(device))
    optimizer = torch.optim.Adam([basis], lr=.01)
    desired = {}
    for split in ['fit', 'validation']:
        desired[split] = caches[split]['oracle'][np.array(pairs[split])[:, 1]].clone()
        if shuffled:
            permutation = np.random.default_rng(seed+10000+(split == 'validation')).permutation(len(desired[split]))
            desired[split] = desired[split][permutation]
    @torch.no_grad()
    def validation_loss():
        total = 0.
        for start in range(0, len(pairs['validation']), 64):
            base, donor = batch_values(caches['validation'], pairs['validation'][start:start+64], site, device)
            changed, _ = patch_residual(base, donor, basis)
            logp = resumed_logits(model, changed, site)[:, letters].log_softmax(-1)
            total += float(-(desired['validation'][start:start+64].to(device)*logp).sum().cpu())
        return total/len(pairs['validation'])
    best_loss = validation_loss()
    history = [{'step': 0, 'validation_cross_entropy': best_loss}]
    best, best_step = basis.detach().cpu().clone(), 0
    for step in range(1, steps+1):
        indexes = rng.integers(len(pairs['fit']), size=64)
        batch = [pairs['fit'][i] for i in indexes]
        base, donor = batch_values(caches['fit'], batch, site, device)
        changed, _ = patch_residual(base, donor, basis)
        logp = resumed_logits(model, changed, site)[:, letters].log_softmax(-1)
        loss = -(desired['fit'][indexes].to(device)*logp).sum(-1).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite alignment objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(basis.grad).all():
            raise ValueError('Nonfinite alignment gradient')
        optimizer.step()
        with torch.no_grad():
            basis.copy_(orthonormalize(basis))
        if step % 50 == 0 or step == steps:
            measured = validation_loss()
            print(json.dumps({'alignment_seed': seed, 'site': site, 'shuffled': shuffled,
                              'step': step, 'validation_cross_entropy': measured}), flush=True)
            history.append({'step': step, 'validation_cross_entropy': measured, 'train_batch_cross_entropy': float(loss.detach().cpu())})
            if measured < best_loss:
                best_loss, best, best_step = measured, basis.detach().cpu().clone(), step
    error = float((best.T@best-torch.eye(3)).abs().max())
    if error > 1e-5:
        raise ValueError('Basis lost orthogonality')
    return best, {'seed': seed, 'shuffled_supervision': shuffled, 'steps': steps, 'best_step': best_step,
                  'validation_cross_entropy': best_loss, 'history': history, 'orthogonality_error': error}


@torch.no_grad()
def score_basis(model, cache, pairs, site, letters, device, basis=None, mode='patch', norm_reference=None):
    if not pairs:
        return {'pair_count': 0}
    values = {key: [] for key in ['oracle_kl', 'donor_model_kl', 'retention_kl', 'category_agreement', 'displacement_norm']}
    identity_error = donor_error = 0.
    if basis is not None:
        basis = basis.to(device)
    if norm_reference is not None:
        norm_reference = norm_reference.to(device)
    for start in range(0, len(pairs), 64):
        subset = pairs[start:start+64]
        indexes = np.array(subset)
        base, donor = batch_values(cache, subset, site, device)
        displacement = torch.zeros_like(base[:, -1])
        changed = base
        if mode == 'full':
            changed = base.clone()
            displacement = donor[:, -1]-base[:, -1]
            changed[:, -1] = donor[:, -1]
        elif mode == 'patch':
            desired_norm = None
            if norm_reference is not None:
                _, target_delta = patch_residual(base, donor, norm_reference)
                desired_norm = target_delta.norm(dim=-1, keepdim=True)
            changed, displacement = patch_residual(base, donor, basis, desired_norm)
        elif mode != 'clean':
            raise ValueError('Unknown intervention mode')
        logp = resumed_logits(model, changed, site)[:, letters].log_softmax(-1)
        original = cache['log_probs'][indexes[:, 0]].to(device)
        donor_logp = cache['log_probs'][indexes[:, 1]].to(device)
        oracle = cache['oracle'][indexes[:, 1]].to(device)
        if mode == 'clean':
            identity_error = max(identity_error, float((logp-original).abs().max().cpu()))
        if mode == 'full' and site == 1:
            donor_error = max(donor_error, float((logp-donor_logp).abs().max().cpu()))
        computed = {'oracle_kl': (oracle*(oracle.log()-logp)).sum(-1)/math.log(2),
                    'donor_model_kl': (donor_logp.exp()*(donor_logp-logp)).sum(-1)/math.log(2),
                    'retention_kl': (original.exp()*(original-logp)).sum(-1)/math.log(2),
                    'displacement_norm': displacement.norm(dim=-1),
                    'category_agreement': logp.exp().reshape(-1, 4, 2).sum(-1).argmax(-1).eq(oracle.reshape(-1, 4, 2).sum(-1).argmax(-1)).float()}
        for key, tensor in computed.items():
            values[key].extend(tensor.cpu().tolist())
    if max(identity_error, donor_error) > 1e-5:
        raise ValueError('Identity or late donor control failed')
    return {'pair_count': len(pairs), 'mean': {key: float(np.mean(v)) for key, v in values.items()},
            'oracle_kl_per_pair': values['oracle_kl'], 'median_oracle_kl': float(np.median(values['oracle_kl'])),
            'identity_max_logprob_error': identity_error if mode == 'clean' else None,
            'late_donor_max_logprob_error': donor_error if mode == 'full' and site == 1 else None}


def readout_basis(model, letters):
    weights = (model.unembedding.weight[letters]*model.final_norm.weight).detach().cpu()
    categories = weights.reshape(4, 2, -1).mean(1)
    centered = (categories-categories.mean(0)).T
    return torch.linalg.svd(centered, full_matrices=False).U[:, :3]


def run(dataset_root, output_root, device):
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use new alignment output root')
    root.mkdir(parents=True, exist_ok=True)
    data, pairs, dataset_manifest = load_data(dataset_root)
    identity = corpus_identity('data/processed/synthetic_cycle_null')
    if dataset_manifest['old_corpus_identity'] != identity:
        raise ValueError('Original synthetic corpus drift')
    tokenizer = EVATokenizer.load('data/processed/synthetic_cycle_null/tokenizer.json')
    letters = torch.tensor([tokenizer.token_to_id[c] for c in ALPHABET], device=device)
    metadata = {'experiment': 'EXP-0006', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'environment': environment(), 'dataset_manifest': dataset_manifest, 'device': device,
                'manuscript_test_evaluated': False, 'supervised_alignment': True,
                'pair_indices': pairs, 'frozen_language_models': True}
    write_json(root/'manifest.json', metadata)
    started = time.monotonic()
    for seed, kind in [(42, 'trained'), (43, 'trained'), (44, 'trained'), (42, 'untrained')]:
        path = Path(f'outputs/EXP-0004/cycle_null-seed{seed}')/('best.pt' if kind == 'trained' else 'initial.pt')
        model, payload = load_checkpoint(path, device)
        if payload['corpus_identity'] != identity or model.config.n_layers != 2:
            raise ValueError('Frozen model identity/architecture mismatch')
        model.eval().requires_grad_(False)
        original_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        caches = {name: cache_records(model, records, tokenizer, device) for name, records in data.items()}
        control_basis = readout_basis(model, letters)
        for site in [0, 1]:
            fit_seed = 6300+seed*10+site
            true_basis, fit_info = fit_basis(model, caches, pairs, site, letters, fit_seed, device)
            alternatives = {'learned': true_basis, 'readout_span': control_basis}
            fitting = {'learned': fit_info}
            if kind == 'trained':
                shuffled, info = fit_basis(model, caches, pairs, site, letters, fit_seed, device, shuffled=True)
                alternatives['shuffled_supervision'] = shuffled
                fitting['shuffled_supervision'] = info
            for random_seed in range(8):
                random = torch.randn(model.config.d_model, 3, generator=torch.Generator().manual_seed(6500+random_seed))
                alternatives[f'random_norm_{random_seed}'] = orthonormalize(random)
            scores = {}
            for split in ['test128', 'test96', 'test192', 'preservation']:
                cache = caches['test128' if split == 'preservation' else split]
                scores[split] = {mode: score_basis(model, cache, pairs[split], site, letters, device, mode=mode)
                                 for mode in ['clean', 'full']}
                for name, basis in alternatives.items():
                    scores[split][name] = score_basis(model, cache, pairs[split], site, letters, device, basis,
                                                       norm_reference=true_basis if name.startswith('random_norm_') else None)
            save_path = root/f'{kind}-seed{seed}-site{site}.pt'
            torch.save({'bases': {k: v.cpu() for k, v in alternatives.items()}, 'fit': fitting}, save_path)
            result = {'experiment': 'EXP-0006', 'model_kind': kind, 'model_seed': seed, 'site': f'blocks.{site}.resid_post',
                      'checkpoint': str(path), 'checkpoint_sha256': digest(path), 'basis_file': str(save_path),
                      'basis_sha256': digest(save_path), 'fit': fitting, 'scores': scores,
                      'readout_subspace_overlap': float((true_basis.T@control_basis).square().sum()/3),
                      'resumed_forward_errors': {k: v['resume_max_logit_errors'] for k, v in caches.items()},
                      'elapsed_seconds_since_start': time.monotonic()-started}
            write_json(root/f'{kind}-seed{seed}-site{site}.json', result)
            print(json.dumps({'model': kind, 'seed': seed, 'site': site,
                              'test128_oracle_kl': {k: v['mean']['oracle_kl'] for k, v in scores['test128'].items()}}), flush=True)
        if any(not torch.equal(value.cpu(), original_weights[key]) for key, value in model.state_dict().items()):
            raise ValueError('Frozen language model weights changed')
        del model, caches, original_weights
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(root/'completion.json', {'elapsed_seconds': time.monotonic()-started,
                                      'peak_resident_bytes': usage if sys.platform == 'darwin' else usage*1024,
                                      'alignment_fits': 14, 'alignment_updates': 5600,
                                      'frozen_weights_verified': True, 'manuscript_test_evaluated': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--data', default='data/processed/alignment_v1')
    parser.add_argument('--output', default='outputs/EXP-0006')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'prepare':
        print(json.dumps(prepare(args.data), indent=2))
    else:
        run(args.data, args.output, resolve_device(args.device))


if __name__ == '__main__':
    main()
