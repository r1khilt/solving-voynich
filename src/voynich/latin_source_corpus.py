"""Conservative TEI literary-text extraction for a new Latin source comparison.

Retained strings are normalized characters, not inferred historical spelling.
Uncertain/foreign units and structural boundaries reset context. No XML entity
expansion or external resource loading is permitted.
"""
from __future__ import annotations

import hashlib
import bisect
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter

ALPHABET = 'abcdefghiklmnopqrstuxyz'
NS = '{http://www.tei-c.org/ns/1.0}'
XML_LANG = '{http://www.w3.org/XML/1998/namespace}lang'
EXCLUDE = {'note', 'head', 'speaker', 'stage', 'listPerson', 'person', 'roleName', 'title'}
UNCERTAIN = {'gap', 'foreign', 'unclear', 'add', 'supplied', 'del', 'choice', 'abbr',
             'expan', 'ex', 'app', 'lem', 'sic', 'orig', 'corr'}
INLINE = {'p', 'l', 'hi', 'reg', 'name', 'quote', 'q', 'seg', 'num', 'persName',
          'placeName', 'said', 'emph', 'cit', 'lb', 'pb', 'milestone', 'lg', 'space'}
BOUNDARY = {'book', 'poem', 'letter', 'act', 'scene'}


def tag(node):
    return node.tag.rsplit('}', 1)[-1]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text):
    if '[' in text or ']' in text:
        raise ValueError('bracketed_editorial_text')
    words = []
    dropped_numeric = 0
    for match in re.finditer(r'[^\W_]+', text):
        token = match.group()
        if any(c.isdigit() for c in token):
            dropped_numeric += 1
            continue
        token = token.lower().replace('æ', 'ae').replace('œ', 'oe')
        token = ''.join(c for c in unicodedata.normalize('NFKD', token)
                        if not unicodedata.combining(c)).replace('j', 'i').replace('v', 'u')
        if set(token) - set(ALPHABET):
            raise ValueError('unsupported_alphabetic_text')
        words.append(token)
    return ''.join(words), dropped_numeric


def extract(raw, *, work_id):
    if len(raw) > 6_000_000 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('XML exceeds limits or contains a DTD/entity declaration')
    root = ET.fromstring(raw)
    if root.tag != NS + 'TEI':
        raise ValueError('Expected namespaced TEI root')
    body = root.find(NS + 'text/' + NS + 'body')
    if body is None:
        raise ValueError('Expected one text/body')
    editions = [n for n in body if tag(n) == 'div' and n.get('type') == 'edition']
    if len(editions) != 1 or editions[0].get(XML_LANG) != 'lat':
        raise ValueError('Expected exactly one explicitly Latin edition')
    records, ledger, pending, pending_units, pending_lengths = [], [], [], [], []
    counts = Counter()

    def flush():
        if pending:
            text = ''.join(pending)
            records.append({'id': f'{work_id}:segment{len(records)}', 'text': text,
                            'unit_paths': list(pending_units), 'unit_lengths': list(pending_lengths),
                            'sha256': sha(text)})
            pending.clear()
            pending_units.clear()
            pending_lengths.clear()

    def flatten(node):
        if tag(node) in EXCLUDE:
            return '', set()
        kind = tag(node)
        problems = ({kind} if kind in UNCERTAIN or kind not in INLINE else set())
        if node.get(XML_LANG, 'lat') not in ('lat', 'la'):
            problems.add('non_latin_language_attribute')
        pieces = [node.text or '']
        for child in node:
            text, issues = flatten(child)
            pieces.extend((text, child.tail or ''))
            problems.update(issues)
        return ''.join(pieces), problems

    def visit(node, path):
        kind = tag(node)
        if kind in EXCLUDE:
            counts['excluded_' + kind] += 1
            return
        boundary = kind == 'div' and node.get('subtype') in BOUNDARY
        if boundary:
            flush()
        # A quote outside a p/l is a separate unit; nested quotes are handled
        # inside their outer p/l without double counting.
        if kind in ('p', 'l', 'quote', 'q'):
            text, issues = flatten(node)
            numeric = 0
            if not issues:
                try:
                    normalized, numeric = normalize(text)
                except ValueError as error:
                    issues.add(str(error))
            if issues:
                flush()
                counts['quarantined_units'] += 1
                for reason in issues:
                    counts['reason_' + reason] += 1
                ledger.append({'path': path, 'status': 'quarantined',
                               'reasons': sorted(issues), 'raw_text_sha256': sha(text),
                               'raw_text_characters': len(text)})
            elif normalized:
                counts['retained_units'] += 1
                counts['dropped_numeric_tokens'] += numeric
                pending.append(normalized)
                pending_units.append(path)
                pending_lengths.append(len(normalized))
                ledger.append({'path': path, 'status': 'retained', 'characters': len(normalized),
                               'normalized_sha256': sha(normalized), 'raw_text_sha256': sha(text)})
            else:
                ledger.append({'path': path, 'status': 'empty_after_normalization',
                               'raw_text_sha256': sha(text)})
            return
        if kind not in {'div', 'sp', 'lg', 'milestone', 'pb', 'lb'}:
            raise ValueError(f'Unhandled structural tag: {kind}')
        if node.text and any(c.isalpha() for c in node.text):
            raise ValueError(f'Unaccounted structural text at {path}')
        for index, child in enumerate(node):
            visit(child, f'{path}/{tag(child)}[{index}]')
            if child.tail and any(c.isalpha() for c in child.tail):
                raise ValueError(f'Unaccounted structural tail at {path}')
        if boundary:
            flush()

    visit(editions[0], 'edition[0]')
    flush()
    return records, {'work_id': work_id, 'counts': dict(sorted(counts.items())),
                     'records': len(records), 'characters': sum(len(r['text']) for r in records),
                     'units': ledger}


def slice_records(records, limit):
    """Deterministic prefix without joining any source boundary."""
    if type(limit) is not int or limit < 1:
        raise ValueError('Positive prefix limit required')
    result, remaining = [], limit
    for row in records:
        if not remaining:
            break
        text = row['text'][:remaining]
        if text:
            sizes, amount = [], len(text)
            for size in row['unit_lengths']:
                if not amount:
                    break
                kept = min(size, amount)
                sizes.append(kept)
                amount -= kept
            result.append(dict(row, text=text, sha256=sha(text), source_record_characters=len(row['text']),
                               source_record_sha256=row['sha256'], unit_lengths=sizes,
                               unit_paths=row['unit_paths'][:len(sizes)]))
            remaining -= len(text)
    if remaining:
        raise ValueError('Not enough clean source characters')
    return result


def remove_overlaps(records, protected, width=64):
    """Remove whole contributing units, then split at each removed run.

    This is an exact substring screen, not fuzzy duplication detection. Units
    touching a match spanning their boundary are all removed. Kept pieces are
    never rejoined across the excluded text.
    """
    if type(width) is not int or width < 1 or any(len(s) != width for s in protected):
        raise ValueError('Fixed positive duplicate width required')
    output, removals = [], []
    for record in records:
        text, sizes = record['text'], record['unit_lengths']
        if sum(sizes) != len(text) or len(sizes) != len(record['unit_paths']) or any(n < 1 for n in sizes):
            raise ValueError('Unit coverage differs from record text')
        ends, total = [], 0
        for size in sizes:
            total += size
            ends.append(total)
        bad, matches = set(), 0
        for start in range(len(text) - width + 1):
            if text[start:start + width] in protected:
                matches += 1
                bad.update(range(bisect.bisect_right(ends, start), bisect.bisect_right(ends, start + width - 1) + 1))
        if not bad:
            output.append(record)
            continue
        removals.append({'id': record['id'], 'matched_windows': matches,
                         'removed_unit_paths': [record['unit_paths'][i] for i in sorted(bad)],
                         'removed_characters': sum(sizes[i] for i in bad)})
        index = 0
        while index < len(sizes):
            if index in bad:
                index += 1
                continue
            end = index + 1
            while end < len(sizes) and end not in bad:
                end += 1
            start_char = ends[index - 1] if index else 0
            fragment = text[start_char:ends[end - 1]]
            output.append(dict(record, id=f'{record["id"]}:units{index}-{end}', text=fragment,
                               unit_paths=record['unit_paths'][index:end], unit_lengths=sizes[index:end],
                               sha256=sha(fragment)))
            index = end
    return output, removals
