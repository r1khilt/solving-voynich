"""Bounded, pinned Latin XML acquisition; never execute downloaded content."""
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/manifests/latin_source001_acquisition.json'
DESTINATION = ROOT / 'data/raw/latin-source001/xml'


def fetch(spec, commit):
    relative = Path(spec['path'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'data':
        raise ValueError('Unsafe upstream path')
    target = DESTINATION / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix('.download')
        subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
            '--max-time', '60', '--max-filesize', '6000000',
            f'https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/{commit}/{spec["path"]}',
            '-o', str(temporary)], check=True, timeout=65)
        raw = temporary.read_bytes()
    else:
        temporary = None
        raw = target.read_bytes()
    if (len(raw) != spec['size'] or
            hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() != spec['sha']):
        raise ValueError('Downloaded bytes do not match the pinned Git blob')
    if temporary is not None:
        temporary.rename(target)
    return {'upstream_path': spec['path'], 'local_path': str(target.relative_to(ROOT)),
            'git_blob_sha1': spec['sha'], 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'author': spec['author'], 'role': spec['role']}


def main():
    manifest = json.loads(MANIFEST.read_text())
    files = manifest['files']
    if (sum(r['size'] for r in files) > 40_000_000 or len(files) != 106
            or any(r['size'] > 6_000_000 for r in files)):
        raise ValueError('Fixed acquisition exceeds registered limits')
    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(lambda spec: fetch(spec, manifest['commit']), files))
    result = {'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              'commit': manifest['commit'], 'files': rows,
              'total_bytes': sum(r['bytes'] for r in rows), 'paid_cost_usd': 0,
              'status': 'raw_bytes_verified_not_training_ready'}
    path = ROOT / 'results/LATIN-SOURCE-001/acquisition.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'files': len(rows), 'bytes': result['total_bytes'], 'status': result['status']}))


if __name__ == '__main__':
    main()
