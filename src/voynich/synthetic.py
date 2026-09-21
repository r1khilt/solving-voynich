"""Known-state homophonic/null generator and exact Bayesian reference filter."""

import argparse
import json
from pathlib import Path

import numpy as np

from voynich.runtime import digest, write_json
from voynich.tokenizer import EVATokenizer

ALPHABET = 'abcdefgh'
SIGNAL_PROBABILITY = .65


def generate_page(rng, length=255, family='cycle_null'):
    if family not in {'cycle_null', 'iid'}:
        raise ValueError('Unknown generator')
    state = int(rng.integers(4))
    text, states, roles = [], [], []
    for _ in range(length):
        signal = int(rng.random() < SIGNAL_PROBABILITY)
        if family == 'iid':
            state = int(rng.integers(4))
            symbol = int(rng.integers(8))
        elif signal:
            state = (state + 1) % 4
            symbol = 2 * state + int(rng.integers(2))
        else:
            symbol = int(rng.integers(8))
        text.append(ALPHABET[symbol])
        states.append(state)
        roles.append(signal)
    return ''.join(text), states, roles


def filter_sequence(text, family='cycle_null', signal_probability=SIGNAL_PROBABILITY):
    """Post-observation state/role posterior and next-symbol predictive distribution."""
    if family not in {'cycle_null', 'iid'} or not 0 <= signal_probability <= 1:
        raise ValueError('Invalid filter setting')
    prior = np.full(4, .25)
    states, roles, predictions = [], [], []
    for char in text:
        category = ALPHABET.index(char) // 2
        if family == 'iid':
            posterior, role = np.full(4, .25), signal_probability
            prediction = np.full(8, .125)
        else:
            null = (1-signal_probability) * prior / 8
            signal = signal_probability * prior[(category-1) % 4] / 2
            posterior = null.copy()
            posterior[category] += signal
            evidence = posterior.sum()
            if evidence == 0:
                raise ValueError('Impossible observation')
            posterior /= evidence
            role = signal/evidence
            prediction = np.repeat(np.roll(posterior, 1), 2)*signal_probability/2 + (1-signal_probability)/8
        states.append(posterior)
        roles.append(role)
        predictions.append(prediction)
        prior = posterior
    return np.array(states), np.array(roles), np.array(predictions)


def prepare(root, family, seed=2201, counts=(600, 96, 192), length=255):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh corpus directory')
    root.mkdir(parents=True, exist_ok=True)
    children = np.random.SeedSequence(seed).spawn(3)
    records, texts_seen = {}, set()
    for split, count, child in zip(('train', 'validation', 'test'), counts, children, strict=True):
        rng = np.random.default_rng(child)
        pages = []
        for i in range(count):
            text, states, roles = generate_page(rng, length, family)
            if text in texts_seen:
                raise ValueError('Duplicate synthetic page')
            texts_seen.add(text)
            key = f'{family}-{split}-{i:04d}'
            pages.append({'page_id': key, 'leaf_id': key, 'split': split, 'text': text,
                          'metadata': {'states': states, 'roles': roles}})
        records[split] = pages
        (root/f'{split}.jsonl').write_text(''.join(json.dumps(p, sort_keys=True)+'\n' for p in pages))
    EVATokenizer.fit(p['text'] for p in records['train']).save(root/'tokenizer.json')
    manifest = {'experiment': 'EXP-0004', 'generator': family, 'generator_seed': seed,
                'signal_probability': SIGNAL_PROBABILITY, 'alphabet': ALPHABET,
                'page_counts': dict(zip(('train', 'validation', 'test'), counts, strict=True)),
                'page_length': length, 'training_data_description': f'Synthetic {family} text only; hidden labels excluded from language model inputs/objective',
                'independence': 'Separate SeedSequence child streams; no duplicate full pages; one fixed alphabet/mechanism',
                'derived_sha256': {name: digest(root/name) for name in ('train.jsonl', 'validation.jsonl', 'test.jsonl', 'tokenizer.json')}}
    write_json(root/'preparation.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--family', choices=['cycle_null', 'iid'], required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    write_json(args.manifest, prepare(args.root, args.family))


if __name__ == '__main__':
    main()
