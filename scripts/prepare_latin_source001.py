"""Frozen source-corpus preparation with author exclusions and exact overlap removal."""
import argparse
import gc
import hashlib
import json
import resource
import signal
import time
from pathlib import Path

from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_blind_channel_dev004 import save_new, resource_report
from voynich.latin_source_corpus import ALPHABET, extract, remove_overlaps, slice_records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/LATIN-SOURCE-001'
ACQUISITION = 'data/manifests/latin_source001_acquisition.json'
RAW_MANIFEST = 'results/LATIN-SOURCE-001/acquisition.json'
PATHS = ['src/voynich/latin_source_corpus.py', 'scripts/prepare_latin_source001.py',
         'scripts/acquire_latin_source001.py', 'tests/test_latin_source_corpus.py',
         'scripts/run_blind_channel_dev001.py', 'scripts/run_blind_channel_dev004.py',
         'docs/research/latin-source001-data-plan.md', ACQUISITION, RAW_MANIFEST,
         'data/manifests/blind_channel_development_corpora.json',
         'data/manifests/blind_channel_confirmation_corpora.json']


def windows(records):
    return {r['text'][i:i + 64] for r in records for i in range(len(r['text']) - 63)}


def run(freeze):
    require_frozen(freeze, PATHS)
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    def timeout(*_):
        raise TimeoutError('Preparation wall limit')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    save_new(OUT / 'prepare_started.json', {'freeze': freeze, 'start_unix': time.time()})
    wall, cpu = time.monotonic(), time.process_time()
    specs = json.loads((ROOT / ACQUISITION).read_text())
    raw_meta = json.loads((ROOT / RAW_MANIFEST).read_text())
    if hashlib.sha256((ROOT / ACQUISITION).read_bytes()).hexdigest() != raw_meta['manifest_sha256']:
        raise ValueError('Acquisition manifest changed')
    if len(raw_meta['files']) != 106:
        raise ValueError('Complete inventory required')
    authors = {a: [] for a in specs['authors']}
    source_audits, work_summaries = {}, []
    for row in raw_meta['files']:
        raw = (ROOT / row['local_path']).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Raw XML changed')
        work = Path(row['upstream_path']).stem
        records, audit = extract(raw, work_id=work)
        authors[row['author']].extend(dict(r, author=row['author'], work=work) for r in records)
        source_audits[work] = audit
        work_summaries.append({'work': work, 'author': row['author'], 'role': row['role'],
                               **{k: v for k, v in audit.items() if k != 'units'}})
    reserved = [a for a, r in specs['authors'].items() if r['role'] == 'reserved_reader_author']
    validation = [a for a, r in specs['authors'].items() if r['role'] == 'source_validation']
    training = [a for a, r in specs['authors'].items() if r['role'] == 'training_pool']
    if len(training) != 12 or validation != ['phi1318'] or sorted(reserved) != ['phi0588', 'phi1212']:
        raise ValueError('Author roles differ')
    previous = []
    dev = json.loads((ROOT / 'data/manifests/blind_channel_development_corpora.json').read_text())
    old = dev['sources']['cicero']
    raw = (ROOT / old['derived_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != old['derived_sha256']:
        raise ValueError('Prior Cicero binding changed')
    previous.append(json.loads(raw))
    conf = json.loads((ROOT / 'data/manifests/blind_channel_confirmation_corpora.json').read_text())
    previous.extend(checked_artifact(conf['sources'][a]['derived']) for a in ('sallust', 'tacitus'))
    # Respect existing body boundaries in the prior comparison allocations.
    prior_segments = []
    for payload in previous:
        start = 0
        for end in payload['body_boundaries']:
            prior_segments.append({'text': payload['text'][start:end]})
            start = end
    protected = windows([r for a in reserved for r in authors[a]] + prior_segments)
    removals = {}
    before = {a: sum(len(r['text']) for r in rs) for a, rs in authors.items()}
    for a in validation:
        authors[a], removals[a] = remove_overlaps(authors[a], protected)
        protected.update(windows(authors[a]))
    for a in training:
        authors[a], removals[a] = remove_overlaps(authors[a], protected)
    # Verify surviving text with a second scan before writing training artifacts.
    collisions = {a: sum(r['text'][i:i + 64] in protected for r in authors[a]
                         for i in range(len(r['text']) - 63)) for a in training}
    if any(collisions.values()):
        raise ValueError('Overlap remained after whole-unit removal')
    protected_count = len(protected)
    del protected
    gc.collect()
    author_artifacts = {}
    for a, rows in authors.items():
        if not rows:
            raise ValueError(f'No usable author records: {a}')
        artifact = save_new(ROOT / 'data/processed/latin-source001' / f'{a}.json.gz',
                            {'alphabet': ALPHABET, 'author': a, 'records': rows}, compressed=True)
        author_artifacts[a] = {**specs['authors'][a], 'artifact': artifact,
            'characters_before_overlap': before[a], 'characters': sum(len(r['text']) for r in rows),
            'records': len(rows), 'overlap_removed_characters': before[a] - sum(len(r['text']) for r in rows)}
    small = [r for a in ('phi0448', 'phi0690') for r in slice_records(authors[a], 50_000)]
    small_artifact = save_new(ROOT / 'data/processed/latin-source001/small.json.gz',
                              {'alphabet': ALPHABET, 'records': small}, compressed=True)
    audits = save_new(ROOT / 'outputs/LATIN-SOURCE-001/extraction_audit.json.gz',
                      {'sources': source_audits, 'overlap_removals': removals}, compressed=True)
    result = {'experiment': 'LATIN-SOURCE-001', 'status': 'prepared_no_model_trained',
        'freeze': freeze, 'upstream_commit': specs['commit'], 'alphabet': ALPHABET,
        'authors': author_artifacts, 'works': work_summaries, 'small': small_artifact,
        'small_characters': sum(len(r['text']) for r in small),
        'large_characters': sum(author_artifacts[a]['characters'] for a in training),
        'extraction_audit': audits, 'duplicate_width': 64, 'protected_unique_windows': protected_count,
        'retained_training_overlaps': collisions, 'no_fuzzy_overlap_claim': True,
        'resources': resource_report(wall, cpu)}
    save_new(OUT / 'corpus.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'small_characters', 'large_characters', 'resources')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    args = parser.parse_args()
    try:
        run(args.freeze)
    except Exception as error:
        if not (OUT / 'prepare_failure.json').exists():
            save_new(OUT / 'prepare_failure.json', {'freeze': args.freeze, 'type': type(error).__name__, 'error': str(error)})
        raise
