"""Frozen resources and input boundaries for the fresh end-to-end qualification."""
import json

from scripts.run_blind_channel_dev001 import checked_artifact, require_frozen
from scripts.run_key_bank_expand001 import admission as native_admission
from scripts.run_key_bank_read002 import PATHS as PREVIOUS_PATHS, load_source
from scripts.run_latin_source_model001 import CORPUS, CORPUS_SHA, ROOT, artifact
from scripts.run_blind_channel_confirm001 import SOURCE_PATHS as INITIAL_PATHS
from voynich.recurrent_latin_source import ALPHABET

EXP = 'BLIND-CHANNEL-CONFIRM-002'
OUT, BULK = ROOT/'results'/EXP, ROOT/'outputs'/EXP
DATA = ROOT/'data/processed'/EXP
NAMES = tuple(f'case-{i:02d}' for i in range(1, 33))
WORKERS = 4
FIT_PANEL = f'results/{EXP}/fit-panel.json'
PANEL = f'results/{EXP}/panel.json'
PREVIOUS_READER = 'results/NEURAL-READER-001/panel_manifest.json'
NEW_PATHS = [
    'src/voynich/fresh_cipher_panel.py', 'scripts/confirm002_common.py',
    'scripts/build_blind_channel_confirm002.py', 'scripts/run_blind_channel_confirm002.py',
    'scripts/audit_blind_channel_confirm002.py', 'scripts/predict_blind_channel_confirm002.py',
    'scripts/evaluate_blind_channel_confirm002.py',
    'tests/test_fresh_cipher_panel.py', 'tests/test_blind_channel_confirm002.py',
    'docs/experiments/BLIND-CHANNEL-CONFIRM-002.md',
    PREVIOUS_READER, 'results/KEY-BANK-READ-002/evaluation.json',
    'results/KEY-BANK-READ-002/evaluation-audit.json',
    'data/manifests/blind_channel_confirmation_corpora.json',
]
PATHS = sorted(set(PREVIOUS_PATHS + INITIAL_PATHS + NEW_PATHS))


def verify(spec):
    if artifact(ROOT/spec['path']) != spec:
        raise ValueError('Artifact identity changed: '+spec['path'])


def source_admission(freeze):
    require_frozen(freeze, PATHS)
    benchmark = native_admission(freeze)
    if artifact(ROOT/CORPUS)['sha256'] != CORPUS_SHA:
        raise ValueError('Source corpus changed')
    overlap = json.loads((ROOT/'results/LATIN-SOURCE-001/overlap_audit.json').read_text())
    if (overlap['status'] != 'pass' or overlap['corpus_sha256'] != CORPUS_SHA
            or any(overlap['training_protected_overlaps'].values()) or overlap['validation_protected_overlaps']):
        raise ValueError('Existing training/reserved-author isolation audit failed')
    previous = json.loads((ROOT/'results/KEY-BANK-READ-002/evaluation.json').read_text())
    audit = json.loads((ROOT/'results/KEY-BANK-READ-002/evaluation-audit.json').read_text())
    if (audit['status'] != 'PASS' or audit['evaluation'] != artifact(ROOT/'results/KEY-BANK-READ-002/evaluation.json')
            or previous['reading_development_gate'] != 'PASS' or previous['iid_screen_development_gate'] != 'PASS'):
        raise ValueError('Prior development pipeline is not admitted')
    return benchmark


def fit_inputs(freeze):
    require_frozen(freeze, [FIT_PANEL, PANEL])
    fit_panel = json.loads((ROOT/FIT_PANEL).read_text())
    benchmark = source_admission(fit_panel['source_freeze'])
    if (fit_panel['experiment'] != EXP or set(fit_panel['cases']) != set(NAMES)
            or fit_panel['status'] != 'prepared_fit_only'):
        raise ValueError('Fresh fit panel mismatch')
    for i, name in enumerate(NAMES):
        row = fit_panel['cases'][name]
        if set(row) != {'fit', 'search_seed'} or row['search_seed'] != 73101+257*i:
            raise ValueError('Fit-only boundary or search seed changed')
    return fit_panel, benchmark


def observed(spec, count):
    row = checked_artifact(spec)
    context = row['context']
    if (len(row['records']) != count or context['source_alphabet'] != list(ALPHABET)
            or context['glyph_alphabet'] != list('ABCDEF') or context['stop_probability'] != 1/225
            or context['max_emission_length'] != 2 or context['source_count'] != 1
            or context['denominator'] != 32 or context['max_states'] != 2 or context['max_alternatives'] != 3
            or any(not text or set(text)-set('ABCDEF') for text in row['records'])):
        raise ValueError('Registered ciphertext context differs')
    return row


def panel_inputs(freeze):
    fit_panel, benchmark = fit_inputs(freeze)
    panel = json.loads((ROOT/PANEL).read_text())
    if (panel['experiment'] != EXP or panel['source_freeze'] != fit_panel['source_freeze']
            or panel['fit_panel'] != artifact(ROOT/FIT_PANEL) or set(panel['cases']) != set(NAMES)
            or not panel['construction_audit']['pass']):
        raise ValueError('Full panel identity/construction audit mismatch')
    for name in NAMES:
        if panel['cases'][name]['fit'] != fit_panel['cases'][name]['fit']:
            raise ValueError('Fit/transfer panel binding mismatch')
    return panel, benchmark


def fit_status(freeze):
    panel, benchmark = panel_inputs(freeze)
    campaign = json.loads((OUT/'fit-campaign.json').read_text())
    if [r['case'] for r in campaign['results']] != list(NAMES):
        raise ValueError('All fit processes must be terminal and accounted')
    if campaign['source_freeze'] != panel['source_freeze']:
        raise ValueError('Campaign source identity mismatch')
    files = [OUT/'fit-campaign.json', OUT/'fit-campaign-started.json']
    admitted = {}
    for process in campaign['results']:
        name = process['case']
        files += [p for p in (OUT/f'{name}-fit.json', OUT/f'{name}-fit-audit.json', OUT/f'{name}-parent.json',
                             OUT/f'{name}-fit-started.json', OUT/f'{name}-fit-audit-started.json',
                             OUT/f'{name}-fit-failure.json', OUT/f'{name}-fit-process.json') if p.exists()]
        if json.loads((OUT/f'{name}-fit-process.json').read_text()) != process:
            raise ValueError('Campaign/per-process record mismatch')
        if process['fit_returncode'] == 0 and process['audit_returncode'] == 0:
            path = OUT/f'{name}-fit.json'
            result = json.loads(path.read_text())
            audit = json.loads((OUT/f'{name}-fit-audit.json').read_text())
            if (result['status'] != 'complete_fit_only' or result['case'] != name or result['fit'] != panel['cases'][name]['fit']
                    or result['data_freeze'] != campaign['data_freeze']
                    or audit['status'] != 'PASS' or audit['result'] != artifact(path)
                    or audit['candidates_accounted'] != result['bank_size']
                    or result['source_freeze'] != panel['source_freeze']
                    or audit['auditor'] != artifact(ROOT/'scripts/audit_blind_channel_confirm002.py')):
                raise ValueError('Complete fitting lacks matching audit')
            admitted[name] = result
        else:
            admitted[name] = None
    require_frozen(freeze, [str(p.relative_to(ROOT)) for p in files])
    return panel, benchmark, admitted, campaign


__all__ = ['load_source']
