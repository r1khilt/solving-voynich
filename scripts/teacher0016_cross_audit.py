"""Independent visible-grid and replacement-vector audit for TEACH-0016.

This module never loads a neural checkpoint. Numerical model replay is separate.
"""

import hashlib
import math

import numpy as np
import torch

from scripts.teacher0015_finite_audit import audit_control_permutations
from scripts.teacher0015_suite_audit import audit_group
from scripts.teacher0016_suite_audit import _episode, _index


VECTOR_NAMES = (
    "base_native", "target_native", "source_native", "same_key_native",
    "wrong_source_native", "distinct_source_native", "random_replacement",
    "wrong_replacement", "distinct_replacement", "source_final_native",
)
PREDICTION_NAMES = (
    "base", "target", "source", "identity", "transfer", "same_key",
    "wrong_key", "distinct", "random", "reverse", "final_donor",
)
WIDTH = 512
VOCAB = 2064
TOLERANCE = 3e-3


def _seed(group_id: str, a: int, b: int, d: int,
          marked: bool, order: int) -> int:
    message = f"TEACH-0016-random|{group_id}|{a}|{b}|{d}|{int(marked)}|{order}"
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8],
                          "big") % (2**63 - 1)


def _close(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    if actual.shape != expected.shape or not np.allclose(
            actual, expected, atol=TOLERANCE, rtol=TOLERANCE):
        raise ValueError(f"TEACH-0016 {name} vector differs")


def _matched(base: np.ndarray, source: np.ndarray, norm: float) -> np.ndarray:
    raw = source.astype(np.float64) - base.astype(np.float64)
    return base + raw * (norm / max(float(np.linalg.norm(raw)), 1e-12))


def _prediction(row: dict, name: str, *, full_logits: bool) -> None:
    value = row.get(f"{name}_prediction")
    if type(value) is not int or not 16 <= value < VOCAB:
        raise ValueError(f"TEACH-0016 {name} prediction invalid")
    key = f"{name}_logits"
    if not full_logits:
        if key in row:
            raise ValueError(f"TEACH-0016 unselected {name} logits present")
        return
    logits = row.get(key)
    if (not isinstance(logits, list) or len(logits) != VOCAB or
            any(type(item) not in (int, float) or not math.isfinite(item)
                for item in logits)):
        raise ValueError(f"TEACH-0016 sampled {name} logits invalid")
    if 16 + max(range(VOCAB - 16), key=lambda index: logits[16 + index]) != value:
        raise ValueError(f"TEACH-0016 {name} argmax/logits disagree")


def audit_cross_surface(
        group: dict, distinct_group: dict, rows: list[dict],
        vectors: np.ndarray, *, distractor: int, marked: bool,
        source_order: int, full_logits: bool) -> dict:
    audit_group(group, group["split"])
    audit_group(distinct_group, distinct_group["split"])
    if (group["split"] != distinct_group["split"] or
            group["group_id"] == distinct_group["group_id"] or
            group["key1"] == distinct_group["key1"] or
            distractor not in (0, 1) or type(marked) is not bool or
            source_order not in (0, 1) or not isinstance(rows, list) or
            len(rows) != 9 or not isinstance(vectors, np.ndarray) or
            vectors.dtype != np.float32 or
            vectors.shape != (9, len(VECTOR_NAMES), WIDTH) or
            not np.isfinite(vectors).all()):
        raise ValueError("TEACH-0016 surface/control/vector grid invalid")
    recipient_order = 1 - source_order
    shifted = 0
    for index, (a, b) in enumerate((a, b) for a in range(3) for b in range(3)):
        row = rows[index]
        if not isinstance(row, dict) or "replacement_vectors" in row:
            raise ValueError("TEACH-0016 row archive malformed")
        source = _episode(group, 1, a, distractor, marked, source_order)
        wrong = _episode(group, 0, a, distractor, marked, source_order)
        same = _episode(group, 1, b, distractor, marked, source_order)
        distinct = _episode(distinct_group, 1, a, distractor, marked,
                            source_order)
        base = _episode(group, 0, b, distractor, marked, recipient_order)
        target = _episode(group, 1, b, distractor, marked, recipient_order)
        source_slot = _index(
            source, group["key1"], group["recipient_outputs"][a][1])
        recipient_slot = _index(
            base, group["key1"], group["recipient_outputs"][b][1])
        expected = {
            "group_id": group["group_id"], "source_g": a,
            "recipient_g": b, "distractor": distractor, "marked": marked,
            "source_order": source_order, "recipient_order": recipient_order,
            "source_slot": source_slot, "recipient_slot": recipient_slot,
            "shifted_slot": source_slot != recipient_slot,
            "base_render_id": base["render_id"],
            "target_render_id": target["render_id"],
            "source_render_id": source["render_id"],
            "same_key_render_id": same["render_id"],
            "wrong_render_id": wrong["render_id"],
            "distinct_render_id": distinct["render_id"],
            "base_answer": base["answer"],
            "target_answer": target["answer"],
            "source_answer": source["answer"],
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("TEACH-0016 visible source/recipient identity differs")
        shifted += int(source_slot != recipient_slot and a != b)
        for name in PREDICTION_NAMES:
            _prediction(row, name, full_logits=full_logits)
        if row["identity_prediction"] != row["base_prediction"] or (
                row["final_donor_prediction"] != row["source_prediction"]):
            raise ValueError("TEACH-0016 native patch identity differs")
        if any(type(row.get(key)) not in (int, float) or not math.isfinite(
                row[key]) or row[key] < 0 or row[key] > TOLERANCE
                for key in ("identity_max_abs_logit_error",
                            "final_donor_max_abs_logit_error")):
            raise ValueError("TEACH-0016 native logit error differs")
        states = {name: vectors[index, offset]
                  for offset, name in enumerate(VECTOR_NAMES)}
        base_vector = states["base_native"]
        source_vector = states["source_native"]
        norm = float(np.linalg.norm(source_vector - base_vector))
        if (type(row.get("source_delta_norm")) not in (int, float) or
                not math.isfinite(row["source_delta_norm"]) or
                abs(row["source_delta_norm"] - norm) > TOLERANCE):
            raise ValueError("TEACH-0016 source delta norm differs")
        for name, original in (("wrong_replacement", "wrong_source_native"),
                               ("distinct_replacement",
                                "distinct_source_native")):
            _close(states[name], _matched(base_vector, states[original], norm),
                   name)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed(group["group_id"], a, b, distractor,
                                    marked, source_order))
        random = torch.randn((WIDTH,), generator=generator).numpy().astype(
            np.float64)
        _close(states["random_replacement"],
               base_vector + random * norm / max(float(np.linalg.norm(random)),
                                                  1e-12), "random")
        for name in ("random_replacement", "wrong_replacement",
                     "distinct_replacement"):
            if abs(float(np.linalg.norm(states[name] - base_vector)) - norm) > TOLERANCE:
                raise ValueError(f"TEACH-0016 {name} norm differs")
        if full_logits:
            first = row["identity_logits"]
            source_logits = row["source_logits"]
            if (max(abs(x - y) for x, y in zip(first, row["base_logits"],
                                               strict=True)) > TOLERANCE or
                    max(abs(x - y) for x, y in zip(
                        row["final_donor_logits"], source_logits,
                        strict=True)) > TOLERANCE):
                raise ValueError("TEACH-0016 sampled native logits differ")
    for a in range(3):
        anchor = rows[a * 3]
        for b in (1, 2):
            other = rows[a * 3 + b]
            if (other["source_prediction"] != anchor["source_prediction"] or
                    other["source_answer"] != anchor["source_answer"] or
                    other["source_render_id"] != anchor["source_render_id"]):
                raise ValueError("TEACH-0016 source donor changed across G")
            _close(vectors[a * 3 + b, VECTOR_NAMES.index("source_native")],
                   vectors[a * 3, VECTOR_NAMES.index("source_native")],
                   "shared source")
    for b in range(3):
        anchor = rows[b]
        for a in (1, 2):
            other = rows[a * 3 + b]
            if (other["base_prediction"] != anchor["base_prediction"] or
                    other["target_prediction"] != anchor["target_prediction"]):
                raise ValueError("TEACH-0016 recipient changed across sources")
            for name in ("base_native", "target_native"):
                _close(vectors[a * 3 + b, VECTOR_NAMES.index(name)],
                       vectors[b, VECTOR_NAMES.index(name)],
                       "shared recipient")
    return {"audit": "pass", "attempts": 9,
            "shifted_offdiag_attempts": shifted}


__all__ = ["VECTOR_NAMES", "PREDICTION_NAMES", "audit_control_permutations",
           "audit_cross_surface"]
