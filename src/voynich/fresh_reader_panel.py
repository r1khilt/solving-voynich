"""Fixed source-segment allocation and known-key controls, with no model scoring."""
import hashlib
import random


def allocate_windows(records, *, count=16, seed=571031, width=224, stride=512):
    if (type(count) is not int or count < 1 or type(width) is not int or width < 1
            or type(stride) is not int or stride < width):
        raise ValueError('Positive, nonoverlapping allocation required')
    slots = [(i, start) for i, row in enumerate(records)
             for start in range(0, len(row['text']) - width + 1, stride)]
    if len(slots) < count:
        raise ValueError('Insufficient prescribed candidate windows; no redraw')
    chosen = random.Random(seed).sample(slots, count)
    windows = []
    for index, start in chosen:
        row = records[index]
        text = row['text'][start:start + width]
        windows.append({'text': text, 'record_id': row['id'], 'record_sha256': row['sha256'],
                        'record_index': index, 'offset': start, 'characters': width,
                        'sha256': hashlib.sha256(text.encode()).hexdigest()})
    return windows


def encode_known(text, alphabet, units):
    if len(units) != len(alphabet) or len(set(alphabet)) != len(alphabet) or any(not u for u in units):
        raise ValueError('One nonempty unit per unique letter required')
    table = dict(zip(alphabet, units, strict=True))
    if set(text) - set(alphabet):
        raise ValueError('Plaintext outside alphabet')
    return ''.join(table[c] for c in text)


def shuffled_plaintext(text, seed):
    letters = list(text)
    random.Random(seed).shuffle(letters)
    return ''.join(letters)


def reader_gate(errors, baseline_errors, *, keys=16, per_key_characters=448):
    if (len(errors) != keys or len(baseline_errors) != keys
            or any(type(e) is not int or e < 0 for e in [*errors, *baseline_errors])):
        raise ValueError('Complete nonnegative error panel required')
    total, base = sum(errors), sum(baseline_errors)
    gates = {'overall_cer_at_most_002': 50 * total <= keys * per_key_characters,
             'every_key_cer_at_most_005': all(20 * e <= per_key_characters for e in errors),
             'relative_reduction_25pct_or_both_perfect': (4 * total <= 3 * base) if base else total == 0}
    return {'gates': gates, 'pass': all(gates.values()),
            'relative_edit_reduction': 1 - total / base if base else None,
            'both_perfect': total == 0 and base == 0}
