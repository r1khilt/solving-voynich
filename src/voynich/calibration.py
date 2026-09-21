"""EXP-0004: supervised diagnostic probes after ciphertext-only LM training.

The probes use known synthetic labels. They are not unsupervised decipherment.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from voynich.interpret import patch_activation
from voynich.runtime import PageWindows, corpus_identity, digest, environment, resolve_device, write_json
from voynich.synthetic import ALPHABET, filter_sequence
from voynich.train import load_checkpoint


POSITIONS = np.linspace(32, 223, 24, dtype=int)


def fit_readout(features, labels, classes, alpha=1.):
    x = np.asarray(features, dtype=np.float64)
    mean, std = x.mean(0), x.std(0).clip(1e-5)
    z = (x-mean)/std
    y = np.eye(classes)[labels]
    bias = y.mean(0)
    weights = np.linalg.solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@(y-bias))
    return {'mean': mean, 'std': std, 'weights': weights, 'bias': bias}


def predict_readout(probe, features):
    return (((features-probe['mean'])/probe['std'])@probe['weights']+probe['bias']).argmax(-1)


def accuracy(predictions, labels, classes):
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    by_class = [float((predictions[labels == i] == i).mean()) if np.any(labels == i) else None
                for i in range(classes)]
    return {'accuracy': float((predictions == labels).mean()), 'class_recall': by_class,
            'balanced_accuracy': float(np.mean([v for v in by_class if v is not None])), 'count': len(labels)}


@torch.no_grad()
def collect(model, data, family, device):
    site = f'blocks.{model.config.n_layers-1}.resid_post'
    features = {'embedding': [], 'final_residual': []}
    labels = {'state': [], 'signal': []}
    oracle = {'state': [], 'signal': []}
    local_oracle = {'state': [], 'signal': []}
    lm_bits, oracle_bits = [], []
    for start in range(0, len(data.pages), 8):
        pages = data.pages[start:start+8]
        x = torch.tensor([data.tokenizer.encode(p['text'], add_eos=False) for p in pages], device=device)
        out = model(x, cache_names=['embed', site])
        indexes = torch.tensor(POSITIONS+1, device=device)
        for key, name in [('embedding', 'embed'), ('final_residual', site)]:
            features[key].append(out.cache[name][:, indexes].reshape(-1, model.config.d_model).cpu().numpy())
        targets = x[:, indexes+1]
        lm_bits.extend((F.cross_entropy(out.logits[:, indexes].transpose(1, 2), targets, reduction='none')/math.log(2)).flatten().cpu().tolist())
        for p in pages:
            state, role, pred = filter_sequence(p['text'], family)
            labels['state'].extend(np.array(p['metadata']['states'])[POSITIONS].tolist())
            labels['signal'].extend(np.array(p['metadata']['roles'])[POSITIONS].tolist())
            oracle['state'].extend(state[POSITIONS].argmax(-1).tolist())
            oracle['signal'].extend((role[POSITIONS] >= .5).astype(int).tolist())
            for pos in POSITIONS:
                st, ro, _ = filter_sequence(p['text'][pos-3:pos+1], family)
                local_oracle['state'].append(int(st[-1].argmax()))
                local_oracle['signal'].append(int(ro[-1] >= .5))
                oracle_bits.append(float(-np.log2(pred[pos, ALPHABET.index(p['text'][pos+1])])))
    return {'features': {k: np.concatenate(v) for k, v in features.items()},
            'labels': {k: np.array(v) for k, v in labels.items()},
            'oracle': oracle, 'local_oracle': local_oracle,
            'lm_bits': float(np.mean(lm_bits)), 'oracle_bits': float(np.mean(oracle_bits))}


def state_basis(probe):
    raw_weights = probe['weights']/probe['std'][:, None]
    raw_weights = raw_weights - raw_weights.mean(1, keepdims=True)
    u, singular, _ = np.linalg.svd(raw_weights, full_matrices=False)
    return u[:, :3], singular.tolist()


def subspace_patch(donor, basis, match_norm=None):
    def patch(value):
        result = value.clone()
        delta = donor[:, -1]-value[:, -1]
        displacement = (delta@basis)@basis.T
        if match_norm is not None:
            displacement *= match_norm/displacement.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        result[:, -1] += displacement
        return result
    return patch


@torch.no_grad()
def causal_state_test(model, data, probe, device):
    """Same last symbol, different known posterior state; no model-based pair selection."""
    prefixes = [p['text'][:128] for p in data.pages]
    filtered = [filter_sequence(p) for p in prefixes]
    posteriors = np.array([f[0][-1] for f in filtered])
    predictions = np.array([f[2][-1] for f in filtered])
    pairs = []
    for i, prefix in enumerate(prefixes):
        if posteriors[i].max() < .6:
            continue
        donors = [j for j, other in enumerate(prefixes) if prefix[-1] == other[-1]
                  and posteriors[j].max() >= .6 and posteriors[i].argmax() != posteriors[j].argmax()]
        if donors:
            pairs.append((i, donors[0]))
    pairs = pairs[:64]
    if not pairs:
        return {'pair_count': 0, 'status': 'No pairs meet registered criterion; no substitute selection'}
    basis, singular = state_basis(probe)
    random_basis = np.linalg.qr(np.random.default_rng(2207).normal(size=basis.shape))[0]
    basis = torch.tensor(basis, dtype=torch.float32, device=device)
    random_basis = torch.tensor(random_basis, dtype=torch.float32, device=device)
    token_ids = torch.tensor([data.tokenizer.encode(c, False, False)[0] for c in ALPHABET], device=device)
    site = f'blocks.{model.config.n_layers-1}.resid_post'
    values = {k: [] for k in ['clean', 'state_subspace', 'random_norm_matched', 'full_residual']}
    identity_error = restore_error = 0.
    def letter_probs(logits):
        # Compare symbol identity conditional on emitting a letter, excluding special tokens.
        return logits[:, -1, token_ids].log_softmax(-1)
    for start in range(0, len(pairs), 8):
        batch = pairs[start:start+8]
        x = torch.tensor([data.tokenizer.encode(prefixes[i], add_eos=False) for i, _ in batch], device=device)
        donor_x = torch.tensor([data.tokenizer.encode(prefixes[j], add_eos=False) for _, j in batch], device=device)
        clean = model(x, cache_names=[site])
        donor = model(donor_x, cache_names=[site])
        delta = donor.cache[site][:, -1]-clean.cache[site][:, -1]
        norm = ((delta@basis)@basis.T).norm(dim=-1, keepdim=True)
        outputs = {'clean': clean}
        outputs['state_subspace'] = model(x, interventions={site: subspace_patch(donor.cache[site], basis)})
        outputs['random_norm_matched'] = model(x, interventions={site: subspace_patch(donor.cache[site], random_basis, norm)})
        outputs['full_residual'] = model(x, interventions={site: patch_activation(donor.cache[site], positions=[-1])})
        identity = model(x, interventions={site: patch_activation(clean.cache[site], positions=[-1])})
        identity_error = max(identity_error, float((identity.logits-clean.logits).abs().max().cpu()))
        restore_error = max(restore_error, float((outputs['full_residual'].logits[:, -1]-donor.logits[:, -1]).abs().max().cpu()))
        oracle = torch.tensor(predictions[[j for _, j in batch]], dtype=torch.float32, device=device)
        for name, out in outputs.items():
            kl = (oracle*(oracle.log()-letter_probs(out.logits))).sum(-1)/math.log(2)
            values[name].extend(kl.cpu().tolist())
    if max(identity_error, restore_error) > 1e-5:
        raise ValueError('Causal controls failed')
    return {'pair_count': len(pairs), 'pairs': pairs, 'reused_donors': True,
            'basis_rank': 3, 'probe_singular_values': singular,
            'kl_bits_to_donor_oracle': {k: float(np.mean(v)) for k, v in values.items()},
            'kl_bits_per_pair': values, 'identity_max_logit_error': identity_error,
            'full_residual_donor_max_logit_error': restore_error,
            'limitations': 'Supervised probe directions; matched final symbol; fixed single random subspace with matched displacement norm; reused donors, descriptive only; no unique circuit or cross-family claim'}


@torch.no_grad()
def run(checkpoint, data_dir, family, device='cpu', causal=False):
    identity = corpus_identity(data_dir)
    model, payload = load_checkpoint(checkpoint, device)
    model.eval()
    if payload['corpus_identity'] != identity:
        raise ValueError('Checkpoint/corpus mismatch')
    train = collect(model, PageWindows(data_dir, 'train', 256), family, device)
    test_data = PageWindows(data_dir, 'test', 256, allow_test=True)
    test = collect(model, test_data, family, device)
    report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
              'corpus_identity': identity, 'environment': environment(), 'family': family, 'device': device,
              'synthetic_test_evaluated': True, 'manuscript_test_evaluated': False,
              'probe_positions_zero_based': POSITIONS.tolist(), 'ridge_alpha': 1.,
              'test_character_bits': test['lm_bits'], 'oracle_character_bits': test['oracle_bits'],
              'uniform_character_bits': 3., 'probes': {},
              'label_access': 'Language model trained only on synthetic visible text. Diagnostic readouts fitted afterward using training ground truth. Test labels used for scoring only.',
              'limitations': ['One generator/alphabet and one unrecoverable IID control.',
                              'No mechanism-family/key/language transfer claim.',
                              'Probe decodability alone is not causal interpretation or unsupervised recovery.']}
    state_probe = None
    for task, classes in [('state', 4), ('signal', 2)]:
        majority = int(np.bincount(train['labels'][task]).argmax())
        metrics = {'majority': accuracy(np.full_like(test['labels'][task], majority), test['labels'][task], classes),
                   'oracle_full': accuracy(test['oracle'][task], test['labels'][task], classes),
                   'oracle_last4': accuracy(test['local_oracle'][task], test['labels'][task], classes)}
        for feature in ['embedding', 'final_residual']:
            probe = fit_readout(train['features'][feature], train['labels'][task], classes)
            metrics[feature] = accuracy(predict_readout(probe, test['features'][feature]), test['labels'][task], classes)
            if task == 'state' and feature == 'final_residual':
                state_probe = probe
        report['probes'][task] = metrics
    if causal and family == 'cycle_null':
        report['causal_state_patch'] = causal_state_test(model, test_data, state_probe, device)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--family', choices=['cycle_null', 'iid'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--causal', action='store_true')
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Use a new output file')
    torch.set_num_threads(4)
    report = run(args.checkpoint, args.data, args.family, resolve_device(args.device), args.causal)
    write_json(args.output, report)
    print({'family': args.family, 'test_bits': report['test_character_bits'],
           'state': report['probes']['state']['final_residual']['accuracy'],
           'signal': report['probes']['signal']['final_residual']['accuracy']})


if __name__ == '__main__':
    main()
