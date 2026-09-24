"""No-model trace and resource arithmetic checks for TEACH-0014."""

from copy import deepcopy

import pytest

from scripts.teacher0014_artifact_audit import (
    ARMS, EXPECTED_CONFIG, PARAMETERS, _audit_benchmark, audit_trace,
)


def _trace(arm, steps=3):
    components = {"answer": 1.2}
    if arm == "latent_rows_causal":
        components["causal"] = .7
    if arm == "latent_rows_diffuse":
        components.update(edge_all_steps=.6, edge_denoise=.5)
    return [{"step": step, "losses": components.copy(), "gradient_norm": .9,
             "answer_input_sha256": "a" * 64,
             "answer_label_sha256": "b" * 64,
             "causal_input_sha256": "c" * 64 if arm == "latent_rows_causal"
             else None} for step in range(steps)]


def test_trace_audits_every_step_and_arm_specific_components():
    for arm in ("latent_rows_answer", "latent_rows_causal",
                "latent_rows_diffuse"):
        fingerprints = audit_trace(_trace(arm), arm, steps=3)
        assert len(fingerprints["answer_inputs"]) == 3
    broken = _trace("latent_rows_causal")
    broken[1]["causal_input_sha256"] = None
    with pytest.raises(ValueError, match="missing causal"):
        audit_trace(broken, "latent_rows_causal", steps=3)
    broken = _trace("latent_rows_diffuse")
    broken[2]["step"] = 1
    with pytest.raises(ValueError, match="malformed loss row"):
        audit_trace(broken, "latent_rows_diffuse", steps=3)
    broken = _trace("latent_rows_answer")
    broken[0]["losses"]["answer"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        audit_trace(broken, "latent_rows_answer", steps=3)


def _benchmark_fixture():
    measured = {arm: {
        "parameters": PARAMETERS[arm], "warmup_steps": 4,
        "timed_steps": 20, "timed_step_seconds": [.1] * 20,
        "median_timed_step_seconds": .1} for arm in ARMS}
    projection = .1 * len(ARMS) * EXPECTED_CONFIG["steps_per_arm"] * 2
    report = {"source_git_head": "a" * 40, "source_sha256": {"x": "a" * 64},
              "config_sha256": "b" * 64}
    benchmark = {**report, "status": "pass", "config": EXPECTED_CONFIG,
                 "measured": measured,
                 "projected_training_seconds": projection,
                 "conservative_projected_seconds": projection * 1.5 + 1800,
                 "peak_sampled_mps_allocated_bytes": 1024}
    return benchmark, report


def test_benchmark_audit_recomputes_cost_and_rejects_tampering():
    benchmark, report = _benchmark_fixture()
    _audit_benchmark(benchmark, report)
    bad = deepcopy(benchmark)
    bad["measured"]["latent_rows_diffuse"]["median_timed_step_seconds"] = .05
    with pytest.raises(ValueError, match="timing median"):
        _audit_benchmark(bad, report)
    bad = deepcopy(benchmark)
    bad["conservative_projected_seconds"] -= 1
    with pytest.raises(ValueError, match="projection"):
        _audit_benchmark(bad, report)
    bad = deepcopy(benchmark)
    del bad["measured"]["raw_null"]
    with pytest.raises(ValueError, match="arm set"):
        _audit_benchmark(bad, report)
