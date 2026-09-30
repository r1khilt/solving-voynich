"""Descriptive no-gold posterior accounting; never changes a prediction."""
import json
import math

from scripts.audit_key_bank_read001 import close, logsum, logvalue
from scripts.run_blind_channel_dev004 import load_archive, save_new
from scripts.run_key_bank_read002 import NAMES, OUT, ROOT, admission
from scripts.run_latin_source_model001 import artifact


def posterior_summary(bank, evidence, reading=None):
    active = evidence['active_bank_indices']
    if [r['bank_index'] for r in evidence['per_key']] != active:
        raise ValueError('Evidence index mismatch')
    weights = [bank['bank'][i]['log_weight'] for i in active]
    if any(w is None or not math.isfinite(w) for w in weights):
        raise ValueError('Invalid active fitting weight')
    close(logsum(weights), 0.)
    values = [logvalue(r['log_likelihood']) for r in evidence['per_key']]
    z = logsum(w+v for w, v in zip(weights, values, strict=True))
    close(z, logvalue(evidence['log_likelihood']))
    out = {'fitting_key_entropy_bits': -math.fsum(math.exp(w)*w for w in weights)/math.log(2),
           'fitting_mass_with_no_joint_transfer_support': math.fsum(math.exp(w) for w, v in zip(weights, values, strict=True) if v == -math.inf),
           'transfer_supported_keys': sum(math.isfinite(v) for v in values)}
    if z == -math.inf:
        return {**out, 'status': 'entire_bank_unsupported'}
    logs = [w+v-z for w, v in zip(weights, values, strict=True)]
    close(logsum(logs), 0.)
    top = max(range(len(logs)), key=logs.__getitem__)
    kl = math.fsum(math.exp(p)*(p-w) for p, w in zip(logs, weights, strict=True) if p != -math.inf)/math.log(2)
    if kl < -1e-7:
        raise ValueError('Negative posterior KL beyond roundoff')
    out.update(status='complete', transfer_key_entropy_bits=-math.fsum(math.exp(p)*p for p in logs if p != -math.inf)/math.log(2),
               transfer_to_fitting_key_kl_bits=kl, maximum_posterior_key_index=active[top],
               maximum_posterior_key_mass=math.exp(logs[top]), same_key_fitting_mass=math.exp(weights[top]))
    if reading is not None and reading['mixture']['plaintexts'] is not None:
        score = reading['mixture']['joint_log_probability']
        if score > z + 1e-7:
            raise ValueError('Text mass exceeds all evidence')
        logp = score-z
        out.update(selected_text_log_posterior=logp, selected_text_posterior=math.exp(logp),
                   selected_text_above_half_mass=logp > -math.log(2)+1e-7,
                   floating_map_bound_separated=reading['mixture']['floating_bound_separated'])
    return out


def main():
    campaign = json.loads((OUT/'campaign.json').read_text())
    admitted, _ = admission(campaign['freeze'])
    reports = {}
    for name in NAMES:
        ep = OUT/f'{name}-evidence.json'
        if not ep.exists():
            reports[name] = {'status': 'evidence_unavailable'}
            continue
        header = json.loads(ep.read_text())
        evidence, bank = load_archive(header['evidence']), load_archive(admitted[name]['bank'])
        path = OUT/f'{name}.json'
        result = json.loads(path.read_text()) if path.exists() else None
        reading = None if result is None or result['readings'] is None else load_archive(result['readings'])
        reports[name] = {'evidence': artifact(ep), 'result': artifact(path) if path.exists() else None,
                         **posterior_summary(bank, evidence, reading)}
    save_new(OUT/'posterior-diagnostic.json', {'status': 'complete_descriptive_no_gold',
        'campaign': artifact(OUT/'campaign.json'), 'source': artifact(ROOT/'scripts/diagnose_key_bank_read002.py'),
        'cases': reports, 'does_not_change_registered_predictions': True,
        'model_probabilities_not_historical_confidence': True})


if __name__ == '__main__':
    main()
