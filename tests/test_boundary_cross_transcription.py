"""Source-free preflight tests for EXP-0039."""

from pathlib import Path

from voynich.boundary_cross_transcription import (
    evaluate,
    load_gc,
    parse_gc_body,
    shuffle,
    zl_symbols,
)
from voynich.boundary_order import toy


def test_gc_atoms_and_uncertain_junctures() -> None:
    runs = parse_gc_body("<%>ab.@253;c.&9,foo.bar.baz.(q.q9<$>")
    assert runs == [[("f", "o", "o"), ("b", "a", "r"), ("b", "a", "z"),
                     ("(", "q"), ("q", "9")]]
    assert parse_gc_body("aa.bb.cc.dd") == [[("a", "a"), ("b", "b"),
                                                ("c", "c"), ("d", "d")]]
    assert parse_gc_body("aa.bb.?cc.dd.ee.ff.gg") == [[("d", "d"), ("e", "e"),
                                                        ("f", "f"), ("g", "g")]]
    assert parse_gc_body("aa.bb..cc.dd.ee.ff") == [[("c", "c"), ("d", "d"),
                                                    ("e", "e"), ("f", "f")]]
    assert parse_gc_body("aa.@253;b.cc.dd") == [[("a", "a"), ("@253;", "b"),
                                                   ("c", "c"), ("d", "d")]]


def test_gc_loader_skips_reserved_leaf_without_parsing_body(tmp_path: Path) -> None:
    source = tmp_path / "gc.txt"
    source.write_text("#=IVTFF v101 2.0 M 6\n"
                      "<f1r> <! $I=H $H=1>\n"
                      "<f1r.1,@P0> <%>aa.bb.cc.dd\n"
                      "<f2r> <! $I=H $H=1>\n"
                      "<f2r.1,@P0> invalid <unfinished\n")
    groups, counts, ids = load_gc(source, {"f1": "train", "f2": "test"})
    assert len(groups) == 1 and len(groups[0]["words"]) == 4
    assert ids == {"f1r.1"}
    assert counts["train_p0_loci"] == 1


def test_zl_segmentation_and_shuffle_preserve_endpoints_and_halves() -> None:
    assert zl_symbols("cthdaiin") == ("cth", "d", "a", "i", "i", "n")
    group = {"leaf": "toy", "context": ("H", "1"),
             "words": [(str(i),) for i in range(10)]}
    changed = shuffle([group], 123)[0]["words"]
    assert changed[0] == group["words"][0] and changed[-1] == group["words"][-1]
    assert set(changed[1:5]) == set(group["words"][1:5])
    assert set(changed[5:9]) == set(group["words"][5:9])


def test_toy_terminal_dependency_detected() -> None:
    def render(seed: int, n: int) -> list[dict]:
        return [{"leaf": group["leaf"], "context": ("toy", "toy"),
                 "words": [tuple(word) for word in group["words"]]}
                for group in toy(seed, n)]
    report = evaluate(render(390239, 100), render(390339, 30), 390439)
    assert report["contrasts"]["last_minus_null"] >= 0.1
    assert report["contrasts"]["last_minus_first"] >= 0.1
