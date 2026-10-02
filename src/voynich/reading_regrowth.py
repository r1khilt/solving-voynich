"""Revisable joint readings: exact integer proposal law and MH acceptance.

The target uses the unchanged dense binary64 source coefficients as a positive
potential, rational geometric EOS, and an iid-uniform visited-row prior. Dense
row coefficients are not silently renormalized. This is not a key marginal,
posterior initialization, mixing guarantee, or historical decoding claim.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from voynich.source_action_proposal import ReadingEnvironment, ReadingState


@dataclass(frozen=True)
class RegrowthConfig:
    stop: Fraction = Fraction(1, 225)
    root_mass: Fraction = Fraction(1, 8)
    two_glyph_bias: Fraction = Fraction(6)
    grid_bits: int = 32
    cached_source_rows: int = 4096
    maximum_acceptance_blocks: int = 16

    def __post_init__(self):
        if (any(not isinstance(x, Fraction) for x in (self.stop, self.root_mass, self.two_glyph_bias))
                or not 0 < self.stop < 1 or not 0 <= self.root_mass <= 1 or self.two_glyph_bias <= 0
                or self.root_mass.denominator > 2**63
                or type(self.grid_bits) is not int or not 8 <= self.grid_bits <= 32
                or type(self.cached_source_rows) is not int or self.cached_source_rows < 1
                or type(self.maximum_acceptance_blocks) is not int or self.maximum_acceptance_blocks < 1):
            raise ValueError('Fixed rational source/proposal/cut settings and finite bounds required')


def quantized_counts(weights, grid):
    """One count per legal choice plus EXACT integer largest-remainder allocation."""
    weights = tuple(weights)
    if (type(grid) is not int or grid < len(weights) or not weights
            or any(type(w) is not int or w <= 0 for w in weights)):
        raise ValueError('Positive integer weights and a sufficient integer grid required')
    total, budget = sum(weights), grid-len(weights)
    quotient = [divmod(budget*w, total) for w in weights]
    result = [1+q for q, _ in quotient]
    remaining = grid-sum(result)
    order = sorted(range(len(weights)), key=lambda j: (-quotient[j][1], j))
    assert 0 <= remaining < len(weights)
    for j in order[:remaining]:
        result[j] += 1
    assert sum(result) == grid and min(result) >= 1
    return tuple(result)


def cut_probability(length, cut, root_mass=Fraction(1, 8)):
    if (type(length) is not int or length < 1 or type(cut) is not int or not 0 <= cut < length
            or not isinstance(root_mass, Fraction) or not 0 <= root_mass <= 1):
        raise ValueError('A pre-action cut in a complete nonempty path required')
    return (1-root_mass)/length+(root_mass if cut == 0 else 0)


def exact_bernoulli(numerator, denominator, rng, *, maximum_blocks=16):
    """Lazy 64-bit comparison with a rational probability, without log clipping.

    Return decision and raw64 blocks used. A resource cap ABORTS, never rounds
    a tiny acceptance to zero or supplies an approximate decision. One raw block
    is consumed even for probability>=1 to keep the receipt explicit.
    """
    if (type(numerator) is not int or type(denominator) is not int or numerator <= 0 or denominator <= 0
            or type(maximum_blocks) is not int or maximum_blocks < 1
            or not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64)):
        raise ValueError('Positive exact acceptance ratio and bounded work required')
    for count in range(1, maximum_blocks+1):
        value = int(rng.bit_generator.random_raw())
        if numerator >= denominator:
            return True, count
        threshold, remainder = divmod(numerator << 64, denominator)
        if value < threshold:
            return True, count
        if value > threshold or remainder == 0:
            return False, count
        numerator = remainder
    raise RuntimeError('Exact acceptance bit-work cap; no approximate decision')


@dataclass(frozen=True)
class ReadingPath:
    actions: tuple
    state: ReadingState
    counts: tuple
    source_terms: tuple
    complete: bool
    grid_bits: int
    law_token: object

    def policy_log_probability(self):
        return math.fsum(math.log(c) for c in self.counts)-len(self.counts)*self.grid_bits*math.log(2)


class SourceRegrowth:
    """Whole-record revision with past-only source guidance and literal masks.

    Prefix bindings alone are reconstructed. No full old dictionary, true key,
    target source length or old future text is supplied to the proposal.
    """
    def __init__(self, source, environment, config=RegrowthConfig()):
        if not isinstance(environment, ReadingEnvironment) or not isinstance(config, RegrowthConfig):
            raise ValueError('Literal environment and fixed regrowth configuration required')
        p, t = source.probabilities, source.transitions
        if (p.dtype != np.float64 or t.dtype != np.uint32 or p.ndim != 2 or p.shape != t.shape
                or p.shape[1] != environment.rows or not p.flags.c_contiguous or not t.flags.c_contiguous
                or p.flags.writeable or t.flags.writeable or (1 << config.grid_bits) < 2*environment.rows):
            raise ValueError('Matching immutable dense source arrays required')
        self.p, self.t = p, t
        self.root = source.state('')
        if type(self.root) is not int or not 0 <= self.root < len(p):
            raise ValueError('Valid reset source context required')
        self.env, self.config, self.token = environment, config, object()
        self.coefficients = OrderedDict()

    def _row(self, context):
        if not 0 <= context < len(self.p):
            raise ValueError('Source context outside frozen array')
        if context not in self.coefficients:
            row = self.p[context]
            if (not np.isfinite(row).all() or (row <= 0).any() or (row > 1).any()
                    or abs(float(row.sum())-1) > 1e-12):
                raise ArithmeticError('Invalid visited dense source coefficient row')
            pairs = tuple(float(p).as_integer_ratio() for p in row)
            exponents = tuple(d.bit_length()-1 for _, d in pairs)
            assert all(d == 1 << e for (_, d), e in zip(pairs, exponents, strict=True))
            exponent = max(exponents)
            aligned = tuple(n << (exponent-e) for (n, _), e in zip(pairs, exponents, strict=True))
            self.coefficients[context] = (pairs, exponents, aligned)
            if len(self.coefficients) > self.config.cached_source_rows:
                self.coefficients.popitem(last=False)
        else:
            self.coefficients.move_to_end(context)
        return self.coefficients[context]

    def _choices(self, keys, offsets, contexts):
        unfinished = [i for i, r in enumerate(self.env.records) if offsets[i] < len(r)]
        if not unfinished:
            return None, (), (), None
        selected = unfinished[0]
        for i in unfinished[1:]:
            if offsets[i]*len(self.env.records[selected]) < offsets[selected]*len(self.env.records[i]):
                selected = i
        offset, record = offsets[selected], self.env.records[selected]
        pairs, exponents, coefficients = self._row(contexts[selected])
        stop, bias, unit_count = self.config.stop, self.config.two_glyph_bias, len(self.env.pool)
        legal, weights = [], []
        for row, bound in enumerate(keys):
            for length in (1, 2):
                unit = record[offset:offset+length]
                if len(unit) != length or (bound >= 0 and self.env.pool[bound] != unit):
                    continue
                # Common denominators: unit_count*stop.denominator*bias.denominator.
                # Continuation is common to all legal next actions and cancels.
                weight = coefficients[row]*(1 if bound < 0 else unit_count)
                weight *= bias.numerator if bound < 0 and length == 2 else bias.denominator
                weight *= stop.numerator if offset+length == len(record) else stop.denominator
                legal.append(2*row+length-1)
                weights.append(weight)
        if not legal:
            return selected, (), (), None
        return selected, tuple(legal), quantized_counts(weights, 1 << self.config.grid_bits), (pairs, exponents)

    def path(self, *, prefix=(), forced_actions=None, rng=None, progress=lambda: None):
        """One draw or exact forced-path replay; failed draws are not refilled."""
        if (type(prefix) is not tuple or any(type(a) is not int for a in prefix)
                or (forced_actions is not None and (type(forced_actions) is not tuple or prefix
                    or any(type(a) is not int for a in forced_actions)))
                or (forced_actions is None and (not isinstance(rng, np.random.Generator)
                    or not isinstance(rng.bit_generator, np.random.PCG64)))):
            raise ValueError('A fixed prefix plus RNG, or one forced complete/failed path required')
        keys = [-1]*self.env.rows
        offsets, contexts = [0]*len(self.env.records), [self.root]*len(self.env.records)
        texts, actions, counts, terms = [[] for _ in self.env.records], [], [], []
        fixed = prefix if forced_actions is None else forced_actions
        horizon = sum(map(len, self.env.records))
        for j in range(horizon):
            progress()
            selected, legal, allocation, source = self._choices(keys, offsets, contexts)
            if not legal:
                break
            if j < len(fixed):
                action = fixed[j]
                if action not in legal:
                    raise ValueError('Fixed preceding action is not literally legal')
                choice = legal.index(action)
            elif forced_actions is not None:
                raise ValueError('A forced path is truncated before actual completion/death')
            else:
                draw = int(rng.integers(1 << self.config.grid_bits, dtype=np.uint64))
                cumulative, choice = 0, None
                for i, count in enumerate(allocation):
                    cumulative += count
                    if draw < cumulative:
                        choice = i
                        break
                assert choice is not None
                action = legal[choice]
            row, length = action//2, action%2+1
            if keys[row] < 0:
                keys[row] = self.env.pool.index(self.env.records[selected][offsets[selected]:offsets[selected]+length])
            pair, exponent = source[0][row], source[1][row]
            terms.append((pair[0], exponent))
            contexts[selected] = int(self.t[contexts[selected], row])
            texts[selected].append(row)
            offsets[selected] += length
            actions.append(action)
            counts.append(allocation[choice])
        if len(actions) < len(fixed):
            raise ValueError('Fixed actions continue beyond terminal or dead state')
        state = ReadingState(tuple(keys), tuple(offsets), tuple(map(tuple, texts)))
        self.env.validate(state)  # full independent literal check at the boundary
        complete = self.env.selected_record(state) is None
        if not complete and self.env.legal_actions(state):
            raise ArithmeticError('Finite non-erasing horizon did not finish or actually die')
        return ReadingPath(tuple(actions), state, tuple(counts), tuple(terms), complete,
                           self.config.grid_bits, self.token)

    def target_integers(self, path):
        if path.law_token is not self.token or not path.complete:
            raise ValueError('Only a complete state under this immutable proposal instance supported')
        n, records = len(path.actions), len(self.env.records)
        stop, continuation = self.config.stop, 1-self.config.stop
        numerator = math.prod(p for p, _ in path.source_terms)*continuation.numerator**n*stop.numerator**records
        denominator = (1 << sum(e for _, e in path.source_terms))*continuation.denominator**n*stop.denominator**records
        denominator *= len(self.env.pool)**sum(k >= 0 for k in path.state.key)
        return numerator, denominator

    def ratio(self, old, candidate, cut):
        if (old.law_token is not self.token or candidate.law_token is not self.token
                or not old.complete or not candidate.complete or type(cut) is not int
                or not 0 <= cut < min(len(old.actions), len(candidate.actions))
                or old.actions[:cut] != candidate.actions[:cut]):
            raise ValueError('Supported complete paths with identical pre-cut history required')
        on, od = self.target_integers(old)
        nn, nd = self.target_integers(candidate)
        oc = cut_probability(len(old.actions), cut, self.config.root_mass)
        nc = cut_probability(len(candidate.actions), cut, self.config.root_mass)
        if not oc or not nc:
            raise ValueError('Cut has zero probability under the fixed kernel')
        numerator = nn*od*math.prod(old.counts[cut:])*nc.numerator*oc.denominator
        denominator = on*nd*math.prod(candidate.counts[cut:])*nc.denominator*oc.numerator
        difference = self.config.grid_bits*(len(candidate.actions)-len(old.actions))
        if difference >= 0:
            numerator <<= difference
        else:
            denominator <<= -difference
        return numerator, denominator

    def step(self, old, rng, *, progress=lambda: None):
        if (old.law_token is not self.token or not old.complete or not isinstance(rng, np.random.Generator)
                or not isinstance(rng.bit_generator, np.random.PCG64)):
            raise ValueError('A complete supported current reading is required')
        root = self.config.root_mass
        # Component draw is exact rational; the MH correction uses the MARGINAL cut law.
        is_root = int(rng.integers(root.denominator)) < root.numerator
        cut = 0 if is_root else int(rng.integers(len(old.actions)))
        candidate = self.path(prefix=old.actions[:cut], rng=rng, progress=progress)
        if not candidate.complete:
            return old, candidate, {'cut': cut, 'accepted': False, 'failed': True, 'acceptance_blocks': 0}
        numerator, denominator = self.ratio(old, candidate, cut)
        accepted, blocks = exact_bernoulli(numerator, denominator, rng,
                                          maximum_blocks=self.config.maximum_acceptance_blocks)
        return (candidate if accepted else old), candidate, {
            'cut': cut, 'accepted': accepted, 'failed': False, 'acceptance_blocks': blocks}
