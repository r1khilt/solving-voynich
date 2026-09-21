"""Independent checks of split isolation, scoring, and resumable training."""

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from voynich import evaluate as evaluation
from voynich import train as training
from voynich.evaluate import NGram, copy_probability, score_baseline
from voynich.runtime import PageWindows, corpus_identity, evaluate_model, loss_sum
from voynich.tokenizer import (
    ALTERNATIVE,
    UNKNOWN_SPAN,
    UNREADABLE,
    UNCERTAIN_SPACE,
    EVATokenizer,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_corpus(root, *, train=None, validation=None, test=None):
    root.mkdir(parents=True, exist_ok=True)
    texts = {
        "train": train if train is not None else ["ab cab ab\nca", "cc ba ca ba"],
        "validation": validation if validation is not None else ["ab ca", "c ba cab"],
        "test": test if test is not None else ["ca ab", "bc ab"],
    }
    tokenizer = EVATokenizer.fit(texts["train"])
    tokenizer.save(root / "tokenizer.json")
    for split, pages in texts.items():
        records = [
            {"page_id": f"{split}-{index}", "split": split, "text": text}
            for index, text in enumerate(pages)
        ]
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
    register_preparation(root)
    return root, tokenizer


def register_preparation(root, path=None):
    path = path or root / "preparation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "fixture": "synthetic test text; no manuscript data",
        "derived_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "tokenizer.json")
        },
    }
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def corpus(tmp_path):
    return write_corpus(tmp_path / "corpus")


def training_config(path, **overrides):
    config = {
        "model": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 24,
            "context_length": 4,
            "dropout": 0.2,
            "qk_norm": True,
            "attention_gate": True,
            "auxiliary_horizons": [2, 3],
        },
        "training": {
            "steps": 4,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_steps": 1,
            "eval_interval": 1,
            "seed": 117,
            "auxiliary_weight": 0.1,
            "patience": 10,
        },
    }
    config["training"].update(overrides)
    path.write_text(json.dumps(config))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


@pytest.mark.parametrize("context_length", [1, 3, 8])
def test_next_targets_partition_each_page_once_and_horizons_stop_at_page_end(tmp_path, context_length):
    root, tokenizer = write_corpus(
        tmp_path / "corpus", train=["aaa", "bbbbbbb", "c"],
    )
    data = PageWindows(root, "train", context_length)
    inputs, targets, pages = data.batch(range(len(data.windows)), "cpu", horizons=(1, 2, 3))
    scored_positions = Counter()
    for row, window in enumerate(data.windows):
        own_page = tokenizer.encode(data.pages[int(window.page_id.rsplit("-", 1)[1])]["text"])
        assert pages[row] == window.page_id
        for offset in range(context_length):
            if offset >= window.length:
                assert inputs[row, offset] == tokenizer.pad_id
                assert all(target[row, offset] == -100 for target in targets.values())
                continue
            position = window.start + offset
            assert inputs[row, offset] == own_page[position]
            scored_positions[(window.page_id, position + 1)] += 1
            for horizon, target in targets.items():
                expected = own_page[position + horizon] if position + horizon < len(own_page) else -100
                assert target[row, offset] == expected
                assert target[row, offset] != tokenizer.bos_id
    expected_positions = {
        (page, position) for page, sequence in data.sequences.items()
        for position in range(1, len(sequence))
    }
    assert set(scored_positions) == expected_positions
    assert set(scored_positions.values()) == {1}
    assert int(targets[1].eq(tokenizer.eos_id).sum()) == len(data.pages)


def test_uncertainty_unknown_and_padding_are_masked_but_remain_available_in_context(tmp_path):
    uncertain_text = "a" + ALTERNATIVE + UNREADABLE + UNKNOWN_SPAN + UNCERTAIN_SPACE + "z b\n"
    root, tokenizer = write_corpus(
        tmp_path / "corpus", train=["ab"], validation=[uncertain_text, "b"],
    )
    data = PageWindows(root, "validation", context_length=16)
    inputs, targets, _ = data.batch(range(len(data.windows)), "cpu", horizons=(1, 2, 3))
    assert set(tokenizer.uncertainty_ids).issubset(set(inputs[0].tolist()))
    for target in targets.values():
        assert not any(target.eq(ignored).any() for ignored in data.ignored_ids)
        assert target[inputs.eq(tokenizer.pad_id)].eq(-100).all()
    # Two visible letters, a definite space, a newline, and EOS; second page b + EOS.
    assert targets[1].ne(-100).sum() == 7
    assert targets[1].eq(tokenizer.space_id).sum() == 1
    assert targets[1].eq(tokenizer.line_id).sum() == 1
    logits = torch.randn(2, 16, tokenizer.vocab_size, requires_grad=True)
    total, count = loss_sum(logits, targets[1])
    assert count == 7
    total.backward()
    assert logits.grad[targets[1].eq(-100)].eq(0).all()
    assert logits.grad[targets[1].ne(-100)].abs().sum() > 0


def test_all_ignored_loss_is_finite_zero_with_zero_gradients():
    logits = torch.randn(2, 3, 7, requires_grad=True)
    loss, count = loss_sum(logits, torch.full((2, 3), -100, dtype=torch.long))
    assert count == 0
    assert loss == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad.eq(0).all()


class UniformPredictor(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.seen_inputs = []

    def forward(self, inputs):
        self.seen_inputs.extend(inputs.detach().cpu().tolist())
        return SimpleNamespace(logits=torch.zeros(*inputs.shape, self.vocab_size, device=inputs.device))


@pytest.mark.parametrize("batch_size", [1, 2, 20])
@pytest.mark.parametrize("training_mode", [False, True])
def test_evaluation_counts_every_valid_target_once_and_restores_model_mode(corpus, batch_size, training_mode):
    root, tokenizer = corpus
    data = PageWindows(root, "validation", context_length=3)
    model = UniformPredictor(tokenizer.vocab_size).train(training_mode)
    measured = evaluate_model(model, data, "cpu", batch_size=batch_size)
    expected_per_page = {
        page: sum(token not in data.ignored_ids for token in sequence[1:])
        for page, sequence in data.sequences.items()
    }
    assert measured["scored_tokens"] == sum(expected_per_page.values())
    assert {key: value["scored_tokens"] for key, value in measured["pages"].items()} == expected_per_page
    assert measured["bits_per_token"] == pytest.approx(math.log2(tokenizer.vocab_size), abs=1e-6)
    assert measured["accuracy"] == 0  # Uniform argmax is PAD, never an eligible target.
    assert len(model.seen_inputs) == len(data.windows)
    assert model.training is training_mode


def test_evaluation_restores_model_mode_after_forward_failure(corpus):
    root, tokenizer = corpus
    data = PageWindows(root, "validation", context_length=3)

    class FailingPredictor(UniformPredictor):
        def forward(self, inputs):
            raise RuntimeError("expected test interruption")

    model = FailingPredictor(tokenizer.vocab_size).train()
    with pytest.raises(RuntimeError, match="test interruption"):
        evaluate_model(model, data, "cpu")
    assert model.training


def test_baseline_receives_prefix_only_with_same_window_resets_and_scoring(corpus):
    root, tokenizer = corpus
    data = PageWindows(root, "validation", context_length=3)
    calls = []

    def record_probability(prefix, target):
        calls.append((tuple(prefix), target))
        return 1 / tokenizer.vocab_size

    report = score_baseline(data, record_probability)
    expected = []
    for page in data.pages:
        sequence = tokenizer.encode(page["text"])
        for target_position in range(1, len(sequence)):
            if sequence[target_position] in data.ignored_ids:
                continue
            beginning = ((target_position - 1) // 3) * 3
            expected.append((tuple(sequence[beginning:target_position]), sequence[target_position]))
    assert calls == expected
    assert report["scored_tokens"] == len(expected)
    assert report["bits_per_token"] == pytest.approx(math.log2(tokenizer.vocab_size))
    assert report["scored_tokens"] == evaluate_model(UniformPredictor(tokenizer.vocab_size), data, "cpu")["scored_tokens"]


def test_ngram_fits_only_training_pages_and_never_creates_cross_page_bigrams(tmp_path):
    root, tokenizer = write_corpus(tmp_path / "corpus", train=["aa", "bb"], validation=["ababa"])
    train_pages = PageWindows(root, "train", context_length=2)
    ngram = NGram(tokenizer.vocab_size, order=2).fit(train_pages)
    a, b = tokenizer.token_to_id["a"], tokenizer.token_to_id["b"]
    assert ngram.counts[0][()][a] == 2
    assert ngram.counts[0][()][b] == 2
    assert ngram.counts[0][()][tokenizer.eos_id] == 2
    assert not ngram.counts[1][(a,)][b]
    assert not ngram.counts[1][(b,)][a]
    assert not ngram.counts[1].get((tokenizer.eos_id,))
    assert sum(ngram.probability([a], token) for token in range(tokenizer.vocab_size)) == pytest.approx(1)


def test_copy_model_uses_only_already_observed_continuations():
    class Uniform:
        def probability(self, prefix, target):
            return 0.1

    unigram = Uniform()
    # Earlier [1, 2] was followed by 3; the trailing [1, 2] has no observed follower yet.
    prefix = [1, 2, 3, 1, 2]
    assert copy_probability(prefix, 3, unigram) == pytest.approx(0.82)
    assert copy_probability(prefix, 9, unigram) == pytest.approx(0.02)
    assert sum(copy_probability(prefix, target, unigram) for target in range(10)) == pytest.approx(1)
    assert copy_probability([1, 2, 3], 9, unigram) == pytest.approx(0.1)
    assert copy_probability([], 3, unigram) == pytest.approx(0.1)
    assert prefix == [1, 2, 3, 1, 2]


def test_test_access_requires_explicit_gate_before_reading_test_records(corpus):
    root, _ = corpus
    (root / "test.jsonl").write_text("not valid JSON; authorization must be checked first\n")
    with pytest.raises(ValueError, match="allow_test=True"):
        PageWindows(root, "test", context_length=4)
    with pytest.raises(json.JSONDecodeError):
        PageWindows(root, "test", context_length=4, allow_test=True)


def test_evaluation_cli_gates_test_and_writes_authorized_result(corpus, tmp_path, monkeypatch):
    root, _ = corpus
    output = tmp_path / "test-report.json"
    command = ["voynich-evaluate", "--data", str(root), "--baselines", "--split", "test",
               "--context", "3", "--device", "cpu", "--threads", "1", "--output", str(output)]
    monkeypatch.setattr("sys.argv", command)
    with pytest.raises(ValueError, match="allow_test=True"):
        evaluation.main()
    assert not output.exists()
    monkeypatch.setattr("sys.argv", [*command, "--allow-test"])
    evaluation.main()
    report = read_json(output)
    assert report["split"] == "test"
    assert report["corpus_identity"] == corpus_identity(root)
    assert set(report["models"]) == {"unigram", "fivegram", "local_copy_unigram"}
    assert len({result["scored_tokens"] for result in report["models"].values()}) == 1


def test_training_checkpoint_roundtrip_and_provenance_without_reading_test(corpus, tmp_path):
    root, tokenizer = corpus
    # Hashing an opaque test file for provenance is allowed; parsing it is not needed for training.
    (root / "test.jsonl").write_text("reserved test bytes; deliberately not JSON\n")
    register_preparation(root)
    config = training_config(tmp_path / "config.json", steps=2)
    run = tmp_path / "run"
    summary = training.train(config, root, run, device="cpu", threads=1)
    assert summary["completed_steps"] == 2
    assert summary["test_evaluated"] is False
    manifest = read_json(run / "manifest.json")
    assert manifest["corpus_identity"] == corpus_identity(root)
    assert manifest["selection_split"] == "validation"
    assert manifest["test_evaluated"] is False
    assert read_json(run / "tokenizer.json") == read_json(root / "tokenizer.json")
    restored, payload = training.load_checkpoint(run / "last.pt")
    assert payload["step"] == 2
    assert payload["corpus_identity"] == manifest["corpus_identity"]
    assert payload["model_config"]["vocab_size"] == tokenizer.vocab_size
    assert payload["training_config"] == manifest["training_config"]
    measured = evaluate_model(restored, PageWindows(root, "validation", 4), "cpu", batch_size=2)
    assert measured["nll_nats"] == pytest.approx(read_json(run / "history.json")[-1]["validation"]["nll_nats"])
    with pytest.raises(ValueError, match="never silently overwritten"):
        training.train(config, root, run, device="cpu", threads=1)


def test_cpu_resume_restores_optimizer_sampler_dropout_and_schedule_exactly(corpus, tmp_path, monkeypatch):
    root, _ = corpus
    config = training_config(tmp_path / "config.json")
    full = tmp_path / "full"
    training.train(config, root, full, device="cpu", threads=1)
    reference_evaluate = training.evaluate_model
    calls = 0

    def interrupt_after_second_checkpoint(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:  # initial, step 1, step 2 saved, then interrupted step 3
            raise RuntimeError("simulated training interruption")
        return reference_evaluate(*args, **kwargs)

    interrupted = tmp_path / "interrupted"
    monkeypatch.setattr(training, "evaluate_model", interrupt_after_second_checkpoint)
    with pytest.raises(RuntimeError, match="simulated training interruption"):
        training.train(config, root, interrupted, device="cpu", threads=1)
    _, interrupted_payload = training.load_checkpoint(interrupted / "last.pt")
    assert interrupted_payload["step"] == 2
    monkeypatch.setattr(training, "evaluate_model", reference_evaluate)
    resumed = tmp_path / "resumed"
    training.train(config, root, resumed, device="cpu", threads=1, resume=interrupted / "last.pt")
    _, expected = training.load_checkpoint(full / "last.pt")
    _, actual = training.load_checkpoint(resumed / "last.pt")
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)
    assert actual["step"] == expected["step"] == 4
    assert actual["best_validation_nll"] == expected["best_validation_nll"]
    assert actual["stale_evaluations"] == expected["stale_evaluations"]
    assert torch.equal(actual["sampler_rng"], expected["sampler_rng"])
    assert torch.equal(actual["torch_rng"], expected["torch_rng"])


@pytest.mark.parametrize("changed", ["train.jsonl", "validation.jsonl", "test.jsonl", "tokenizer.json"])
def test_resume_rejects_changed_corpus_or_tokenizer_identity(corpus, tmp_path, changed):
    root, _ = corpus
    config = training_config(tmp_path / "config.json", steps=2)
    run = tmp_path / "original"
    training.train(config, root, run, device="cpu", threads=1)
    changed_file = root / changed
    changed_file.write_text(changed_file.read_text() + "\n")
    register_preparation(root)  # Valid new corpus version must still reject an old checkpoint.
    with pytest.raises(ValueError, match="identical model, tokenizer and corpus/splits"):
        training.train(config, root, tmp_path / "resume", device="cpu", threads=1, resume=run / "initial.pt")


def test_checkpoint_evaluation_rejects_changed_data(corpus, tmp_path, monkeypatch):
    root, _ = corpus
    config = training_config(tmp_path / "config.json", steps=2)
    run = tmp_path / "run"
    training.train(config, root, run, device="cpu", threads=1)
    validation = root / "validation.jsonl"
    validation.write_text(validation.read_text() + "\n")
    register_preparation(root)
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["voynich-evaluate", "--checkpoint", str(run / "best.pt"),
                                    "--data", str(root), "--device", "cpu", "--threads", "1",
                                    "--output", str(output)])
    with pytest.raises(ValueError, match="identity differ"):
        evaluation.main()
    assert not output.exists()


def test_resume_preserves_best_checkpoint_even_when_later_validation_never_improves(corpus, tmp_path, monkeypatch):
    root, _ = corpus
    config = training_config(tmp_path / "config.json")
    real_evaluate = training.evaluate_model
    calls = 0

    def constant_validation_then_interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("stop after checkpoint two")
        measured = real_evaluate(*args, **kwargs)
        measured["nll_nats"] = 1.0
        measured["bits_per_token"] = 1 / math.log(2)
        return measured

    original = tmp_path / "original"
    monkeypatch.setattr(training, "evaluate_model", constant_validation_then_interrupt)
    with pytest.raises(RuntimeError, match="checkpoint two"):
        training.train(config, root, original, device="cpu", threads=1)
    best_model, expected_payload = training.load_checkpoint(original / "best.pt")
    expected_weights = copy.deepcopy(best_model.state_dict())
    calls = 0  # Resume has initial + step 3 + step 4; no interruption.
    resumed = tmp_path / "resumed"
    training.train(config, root, resumed, device="cpu", threads=1, resume=original / "last.pt")
    assert (resumed / "best.pt").is_file(), "A resumed run must retain its actual selected best model"
    recovered_best, actual_payload = training.load_checkpoint(resumed / "best.pt")
    assert actual_payload["step"] == expected_payload["step"] == 0
    for name, tensor in expected_weights.items():
        torch.testing.assert_close(recovered_best.state_dict()[name], tensor, rtol=0, atol=0)


def test_initial_checkpoint_records_initial_validation_as_selection_baseline(corpus, tmp_path):
    root, _ = corpus
    config = training_config(tmp_path / "config.json", steps=2)
    run = tmp_path / "run"
    training.train(config, root, run, device="cpu", threads=1)
    _, checkpoint = training.load_checkpoint(run / "initial.pt")
    expected = read_json(run / "history.json")[0]["validation"]["nll_nats"]
    assert checkpoint["best_validation_nll"] == expected


@pytest.mark.parametrize("changed", ["train.jsonl", "validation.jsonl", "test.jsonl", "tokenizer.json"])
def test_registered_corpus_rejects_unregistered_byte_drift(corpus, changed):
    root, _ = corpus
    registered = read_json(root / "preparation.json")["derived_sha256"]
    original = corpus_identity(root)
    assert all(original[name] == value for name, value in registered.items())
    changed_file = root / changed
    changed_file.write_text(changed_file.read_text() + "\n")
    with pytest.raises(ValueError, match="mismatch|differ|drift|match"):
        corpus_identity(root)


def test_corpus_identity_rejects_missing_preparation_manifest(corpus):
    root, _ = corpus
    (root / "preparation.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="manifest|preparation"):
        corpus_identity(root)


def test_corpus_identity_accepts_explicit_or_project_layout_manifest(tmp_path):
    root, _ = write_corpus(tmp_path / "data" / "processed" / "example")
    local = root / "preparation.json"
    expected = read_json(local)["derived_sha256"]
    explicit = register_preparation(root, tmp_path / "custom" / "registered.json")
    local.unlink()
    actual = corpus_identity(root, preparation_manifest=explicit)
    assert all(actual[name] == value for name, value in expected.items())
    discovered = register_preparation(root, tmp_path / "data" / "manifests" / "example_preparation.json")
    actual = corpus_identity(root)
    assert all(actual[name] == value for name, value in expected.items())
    assert discovered.is_file()


def test_training_refuses_unregistered_corpus_before_writing_run(corpus, tmp_path):
    root, _ = corpus
    config = training_config(tmp_path / "config.json", steps=2)
    (root / "validation.jsonl").write_text((root / "validation.jsonl").read_text() + "\n")
    destination = tmp_path / "run"
    with pytest.raises(ValueError, match="mismatch|differ|drift|match"):
        training.train(config, root, destination, device="cpu", threads=1)
    assert not destination.exists()


def test_training_accepts_explicit_preparation_manifest(corpus, tmp_path):
    root, _ = corpus
    config = training_config(tmp_path / "config.json", steps=2)
    explicit = register_preparation(root, tmp_path / "custom-preparation.json")
    (root / "preparation.json").unlink()
    summary = training.train(config, root, tmp_path / "run", device="cpu", threads=1,
                             preparation_manifest=explicit)
    assert summary["completed_steps"] == 2


def test_requested_intermediate_snapshots_retain_correct_steps(corpus, tmp_path):
    root, _ = corpus
    config = training_config(tmp_path / "config.json", steps=2, snapshot_steps=[1, 2])
    run = tmp_path / "snapshots"
    training.train(config, root, run, device="cpu", threads=1)
    for step in [1, 2]:
        _, payload = training.load_checkpoint(run / f"step-{step:06d}.pt")
        assert payload["step"] == step
        assert payload["best_checkpoint"]["step"] <= step


def test_snapshot_steps_must_align_with_evaluation(corpus, tmp_path):
    root, _ = corpus
    config = training_config(tmp_path / "config.json", eval_interval=2, snapshot_steps=[1])
    with pytest.raises(ValueError, match="snapshot_steps"):
        training.train(config, root, tmp_path / "bad", device="cpu", threads=1)
