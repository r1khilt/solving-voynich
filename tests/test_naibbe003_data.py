import copy

import pytest

from scripts.build_naibbe003_data import (
    encoded_json,
    prepare_split,
    reconstruct_support,
    validate_identities,
    write_immutable_bundle,
)


def toy_inputs():
    ids = [f"c_{number:020x}" for number in range(4)]
    fit = {"class_ids": [ids[2], ids[0], ids[3], ids[1]], "alphabet": list("ab"),
           "group_ids": [f"t_{number:020x}" for number in range(2)],
           "groups": [[ids[1], ids[0]], [ids[2], ids[3]]]}
    answers = {"class_to_letter": dict(zip(ids, "abab", strict=True)),
               "class_to_table": dict(zip(ids, ["one", "one", "two", "two"], strict=True))}
    return fit, answers


def test_preserves_parent_identities_without_reanonymizing():
    fit, answers = toy_inputs()
    original = copy.deepcopy((fit, answers))
    mapping = validate_identities(fit, answers, alphabet="ab", tables=("one", "two"))
    assert mapping[("one", "a")] == fit["groups"][0][1]
    assert mapping[("two", "a")] == fit["groups"][1][0]
    assert (fit, answers) == original


@pytest.mark.parametrize("defect", ["overlap", "missing", "mixed", "duplicate_letter", "extra_answer"])
def test_rejects_corrupt_inherited_partition(defect):
    fit, answers = toy_inputs()
    if defect == "overlap":
        fit["groups"][1][0] = fit["groups"][0][0]
    elif defect == "missing":
        fit["class_ids"].pop()
    elif defect == "mixed":
        fit["groups"][0][0], fit["groups"][1][0] = fit["groups"][1][0], fit["groups"][0][0]
    elif defect == "duplicate_letter":
        answers["class_to_letter"][fit["groups"][0][0]] = "a"
    else:
        answers["class_to_table"]["unexpected"] = "one"
    with pytest.raises(ValueError):
        validate_identities(fit, answers, alphabet="ab", tables=("one", "two"))


def test_collisions_cross_table_parses_and_unigram_precedence():
    rows = [{"role": "unigram", "table": table, "letter": "a", "glyphs": "xy"}
            for table in ("one", "two")]
    rows += [{"role": "prefix", "table": "one", "letter": "b", "glyphs": "x"},
             {"role": "suffix", "table": "two", "letter": "b", "glyphs": "y"},
             {"role": "suffix", "table": "two", "letter": "a", "glyphs": "z"}]
    fit, answers = toy_inputs()
    mapping = validate_identities(fit, answers, alphabet="ab", tables=("one", "two"))
    support = reconstruct_support(rows, mapping)
    assert support["xy"] == sorted([[mapping[("one", "a")]], [mapping[("two", "a")]]])
    assert support["xz"] == [[mapping[("one", "b")], mapping[("two", "a")]]]
    assert support == reconstruct_support(list(reversed(rows)), mapping)


def test_exact_slice_and_support_check_do_not_select_a_model_reading():
    support = {"glyph1": [["b"], ["a"]], "glyph2": [["c", "d"]]}
    letters = {"a": "a", "b": "b", "c": "a", "d": "b"}
    split, answer, counts = prepare_split(["skip", "glyph1", "glyph2"], ["x", "a", "ab"],
                                           (1, 3), support, letters)
    assert split == {"token_range": [1, 3], "tokens": ["glyph1", "glyph2"],
                     "candidates": [[["b"], ["a"]], [["c", "d"]]]}
    assert answer == {"token_range": [1, 3], "plaintext_chunks": ["a", "ab"]}
    assert counts["candidate_count_histogram"] == {2: 1, 1: 1}
    assert counts["gold_characters"] == 3
    assert counts["unique_candidate_tokens"] == 1


@pytest.mark.parametrize("gold,interval", [(["x"], (0, 1)), (["a"], (0, 2)), ([], (0, 1))])
def test_rejects_out_of_support_or_unaligned_sources(gold, interval):
    with pytest.raises(ValueError):
        prepare_split(["glyph"], gold, interval, {"glyph": [["id"]]}, {"id": "a"})


def test_immutable_bundle_checks_all_conflicts_before_any_write(tmp_path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    original = encoded_json({"value": 1})
    write_immutable_bundle({old: original})
    write_immutable_bundle({old: original})
    with pytest.raises(FileExistsError):
        write_immutable_bundle({new: b"new\n", old: encoded_json({"value": 2})})
    assert not new.exists()
    assert old.read_bytes() == original
