"""Read-only corpus replay through original XML paths and prior normalizer."""
import hashlib
import json
import re
import resource
import signal
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from scripts.build_blind_channel_development_corpora import normalize_segment
from scripts.prepare_latin_source001 import PATHS
from scripts.run_blind_channel_dev001 import require_frozen
from scripts.run_blind_channel_dev004 import load_archive, save_new, resource_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/LATIN-SOURCE-001'
OMIT = {'note', 'head', 'speaker', 'stage', 'listPerson', 'person', 'roleName', 'title'}


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    def timeout(*_):
        raise TimeoutError('Audit wall limit')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    wall, cpu = time.monotonic(), time.process_time()
    corpus = json.loads((OUT / 'corpus.json').read_text())
    acquisition = json.loads((OUT / 'acquisition.json').read_text())
    require_frozen(corpus['freeze'], PATHS)
    work_refs = {}
    for row in acquisition['files']:
        raw = (ROOT / row['local_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['sha256'] and len(raw) == row['bytes']
        tree = ET.fromstring(raw)
        edition = next(n for n in tree.iter() if n.get('type') == 'edition')
        work_refs[Path(row['upstream_path']).stem] = edition
    cached = {}
    def reference_unit(work, path):
        if (work, path) in cached:
            return cached[work, path]
        root = work_refs[work]
        for part in path.split('/')[1:]:
            match = re.fullmatch(r'(\w+)\[(\d+)\]', part)
            assert match
            root = root[int(match.group(2))]
            assert root.tag.rsplit('}', 1)[-1] == match.group(1)
        parts = []
        def collect(node, skip=False):
            skip = skip or node.tag.rsplit('}', 1)[-1] in OMIT
            if not skip:
                parts.append(node.text or '')
            for child in node:
                collect(child, skip)
                if not skip:
                    parts.append(child.tail or '')
        collect(root)
        text, _, _ = normalize_segment(''.join(parts))
        cached[work, path] = text
        return text
    characters = Counter()
    all_rows = {}
    records, references = 0, 0
    for author, meta in corpus['authors'].items():
        payload = load_archive(meta['artifact'])
        all_rows[author] = payload['records']
        assert payload['author'] == author and payload['alphabet'] == corpus['alphabet']
        for row in payload['records']:
            texts = [reference_unit(row['work'], path) for path in row['unit_paths']]
            assert list(map(len, texts)) == row['unit_lengths']
            assert ''.join(texts) == row['text']
            assert hashlib.sha256(row['text'].encode()).hexdigest() == row['sha256']
            assert row['author'] == author
            references += len(texts)
            records += 1
            characters[author] += len(row['text'])
        assert characters[author] == meta['characters']
    small = load_archive(corpus['small'])['records']
    for author in ('phi0448', 'phi0690'):
        original = all_rows[author]
        subset = [r for r in small if r['author'] == author]
        assert ''.join(r['text'] for r in subset) == ''.join(r['text'] for r in original)[:50_000]
        for a, b in zip(subset, original):
            assert a['id'] == b['id'] and b['text'].startswith(a['text'])
            assert sum(a['unit_lengths']) == len(a['text'])
            assert a['source_record_sha256'] == b['sha256']
    ledger = load_archive(corpus['extraction_audit'])
    removed = sum(r['removed_characters'] for rows in ledger['overlap_removals'].values() for r in rows)
    assert removed == sum(m['overlap_removed_characters'] for m in corpus['authors'].values())
    result = {'status': 'pass', 'corpus_sha256': hashlib.sha256((OUT / 'corpus.json').read_bytes()).hexdigest(),
        'source_bindings_unchanged': True, 'raw_files_verified': len(acquisition['files']),
        'records_reconstructed_from_xml': records, 'unit_references_replayed': references,
        'author_characters': dict(characters), 'small_is_exact_100k_prefix_subset': True,
        'ledger_removed_characters': removed, 'resources': resource_report(wall, cpu)}
    save_new(OUT / 'corpus_audit.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
