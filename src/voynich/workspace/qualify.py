"""Numerical qualification with analytic time mixing and real-model controls."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .campaign import load_config, write_json
from .mlx_backend import QwenWorkspace


def temporal_toy(mx):
    a = np.array([[1., .2, -.1], [0, 2., .3], [.1, .5, 1.]])
    b = np.array([[.3, .2, .4], [.2, -.1, .6], [.5, .3, .1]])

    class Temporal:
        def __call__(self, h, mask=None):
            previous = mx.concatenate([mx.zeros_like(h[:, :1]), h[:, :-1]], axis=1)
            return h @ mx.array(a.T, dtype=mx.float32) + previous @ mx.array(b.T, dtype=mx.float32)

    toy = QwenWorkspace.__new__(QwenWorkspace)
    toy.mx, toy.width, toy.n_layers = mx, 3, 3
    toy.layers = [lambda h, mask=None: h, Temporal(), lambda h, mask=None: 3 * h]
    toy.model = SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: mx.ones((*ids.shape, 3))))
    rows = np.array([[1., 0., 0.], [0., 1., 2.]])
    actual = toy.jacobian_rows(list(range(7)), rows, (0, 1), skip_first=2)
    expected = np.stack([rows @ (3 * (a + 3 / 4 * b)), 3 * rows])
    error = float(np.max(np.abs(actual - expected)))
    if error > 1e-5:
        raise AssertionError(f"Temporal estimator orientation/scale error: {error}")
    return {"max_error": error, "valid_positions": 4, "expected_transport": "3(A+3B/4), 3I"}


def main():
    config = load_config("configs/jspace0001.json")
    m = QwenWorkspace(config["model_snapshot"], dense_transport=True)
    mx = m.mx
    mx.set_memory_limit(config["memory_bytes_cap"])
    ids = m.encode('The old library contains books about history, art, and science. Careful readers compare competing ideas and examine available evidence. ' * 6)[:128]
    rows = m.output_rows([49000, 50170, 67199, 85667])
    sources = tuple(config["layers"])
    t = time.monotonic()
    j = m.jacobian_rows(ids, rows, sources)
    elapsed = time.monotonic() - t
    one = m.jacobian_rows(ids, rows[:1], sources)
    batch_error = float(np.linalg.norm(j[:, :1] - one) / np.linalg.norm(one))
    clean, _ = m.forward(ids)
    native = m.model.model(mx.array(ids)[None], input_embeddings=m.model.model.embed_tokens(mx.array(ids)[None]).astype(mx.float32))
    native = m.model.lm_head(native[:, -1:])
    mx.eval(native)
    native_error = float(np.max(np.abs(clean - np.array(native[0]))))
    identity, _ = m.forward(ids, patches=[{"layer": 15, "position": len(ids)-1, "delta": np.zeros(m.width)}])
    identity_error = float(np.max(np.abs(clean - identity)))
    earlier, _ = m.forward(ids, logit_positions=[5])
    future, _ = m.forward(ids, logit_positions=[5], patches=[{"layer": 3, "position": 20, "delta": np.ones(m.width)}])
    causal_error = float(np.max(np.abs(earlier - future)))
    fd = []
    for index in (0, 3, 7):
        v = j[index, 0] / np.linalg.norm(j[index, 0])
        expected = float(j[index, 0] @ v)
        for epsilon in (.03, .1, .3):
            actual = (m.transport_scalar(ids, rows[0], sources[index], v * epsilon)
                      - m.transport_scalar(ids, rows[0], sources[index], -v * epsilon)) / (2 * epsilon)
            fd.append({"layer": sources[index], "epsilon": epsilon, "expected": expected, "actual": actual,
                       "relative_error": abs(actual - expected) / abs(expected)})
    # Numerical gates are engineering criteria, fixed before this qualification
    # and scientific scoring; failures remain recorded instead of hidden.
    passed = (batch_error < 1e-4 and native_error < 1e-5 and identity_error == 0 and causal_error == 0
              and all(sum(x["relative_error"] < .02 for x in fd if x["layer"] == layer) >= 2 for layer in (3, 15, 31)))
    report = {"passed": passed, "temporal_toy": temporal_toy(mx), "batch_relative_error": batch_error,
              "native_max_error": native_error, "identity_max_error": identity_error,
              "future_to_past_max_error": causal_error, "finite_differences": fd,
              "batch_seconds": elapsed, "peak_memory_bytes": mx.get_peak_memory(),
              "precision": "dense float32 block projections expanded from cached four-bit values"}
    write_json(Path(config["results"]) / "qualification.json", report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit("Numerical qualification failed; do not launch science")


if __name__ == "__main__":
    main()
