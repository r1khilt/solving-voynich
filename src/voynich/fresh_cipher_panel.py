"""Score-free fresh-window allocation with whole-record and exact-overlap guards."""
import hashlib
import random


def window_hashes(text, width=64):
    if type(width) is not int or width < 1:
        raise ValueError('Positive overlap width required')
    return {hashlib.blake2b(text[i:i+width].encode(), digest_size=16).digest()
            for i in range(len(text)-width+1)}


def allocate_fresh(records, excluded_ids, protected, *, count, seed, width=224, stride=512, overlap=64):
    """Randomized fixed ordering; only declared duplicate exclusions may skip slots."""
    if (type(overlap) is not int or overlap < 1 or type(count) is not int or count < 1
            or type(width) is not int or width < overlap
            or type(stride) is not int or stride < width):
        raise ValueError('Invalid fresh allocation bounds')
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)) or not set(excluded_ids) <= set(ids):
        raise ValueError('Unknown excluded record or duplicate record identity')
    slots = []
    for i, row in enumerate(records):
        if hashlib.sha256(row['text'].encode()).hexdigest() != row['sha256']:
            raise ValueError('Source record hash mismatch')
        if row['id'] not in excluded_ids:
            slots.extend((i, start) for start in range(0, len(row['text'])-width+1, stride))
    random.Random(seed).shuffle(slots)
    chosen, blocked, used = [], 0, set(protected)
    for i, offset in slots:
        row = records[i]
        text = row['text'][offset:offset+width]
        hashes = window_hashes(text, overlap)
        if hashes & used:
            blocked += 1
            continue
        chosen.append({'text': text, 'record_id': row['id'], 'record_sha256': row['sha256'],
                       'record_index': i, 'offset': offset, 'characters': width,
                       'sha256': hashlib.sha256(text.encode()).hexdigest()})
        used.update(hashes)
        if len(chosen) == count:
            return chosen, {'candidate_slots': len(slots), 'examined_slots': len(chosen)+blocked,
                            'overlap_exclusions': blocked, 'excluded_records': len(excluded_ids)}
    raise ValueError('Insufficient fresh nonoverlapping windows; no fallback allocation')
