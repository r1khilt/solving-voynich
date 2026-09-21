"""Corpus integrity tests using synthetic IVTFF, never a training experiment."""

import json
from pathlib import Path
import tempfile
import unittest

from voynich.data import (
    assign_splits, leakage_audit, leaf_id, normalize_locus, parse_ivtff,
    prepare, sha256_bytes,
)
from voynich.tokenizer import (
    ALTERNATIVE, UNREADABLE, UNKNOWN_SPAN, UNCERTAIN_SPACE, RARE_BASE, EVATokenizer,
)


HEADER = "#=IVTFF Eva- 2.0 M 5\n"


def synthetic_pages():
    return [{"page_id": f"f{i}r", "leaf_id": f"f{i}",
             "text": chr(96 + i) * 24, "loci": []} for i in range(1, 11)]


class NormalizationTests(unittest.TestCase):
    def test_editorial_and_layout_markup_never_become_training_text(self):
        text, annotations = normalize_locus("<%>ab<!LATIN PLANT NAME>.{cth}<->@185;<$>")
        self.assertEqual(text, "ab cth " + chr(RARE_BASE + 185))
        self.assertNotIn("LATIN", text)
        self.assertNotIn("<", text)
        self.assertEqual([a["kind"] for a in annotations],
                         ["paragraph_start", "editorial_comment", "ligature",
                          "drawing_interruption", "rare_eva", "paragraph_end"])
        self.assertEqual(annotations[2]["start"], 3)
        self.assertEqual(annotations[2]["end"], 6)

    def test_uncertainty_is_not_silently_first_choice_or_word_space(self):
        text, annotations = normalize_locus("a[o:{cth}]b,c?d???[?:]")
        self.assertEqual(text, "a" + ALTERNATIVE + "b" + UNCERTAIN_SPACE + "c" +
                         UNREADABLE + "d" + UNKNOWN_SPAN + ALTERNATIVE)
        self.assertEqual(annotations[0]["alternatives"], ["o", "{cth}"])
        self.assertEqual(annotations[-1]["alternatives"], ["?", ""])

    def test_malformed_markup_fails_closed(self):
        for raw in ("a[x]", "a[:]", "a[x:y", "a{xy", "a{}", "a@127;", "a@1234;",
                    "a@185", "a<!note", "a<invented>", "a>note", "a:b", "a??? ? ? ? ?☃",
                    "a????", "a{x.y}", "a1"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_locus(raw)


class IVTFFTests(unittest.TestCase):
    def test_comments_extraneous_loci_and_page_tags_are_separate(self):
        raw = HEADER + """# commentary English
<f1r> <! $I=H $H=@>
<f1r.1,@P0> <%>ab.cd<!Latin name>
<f1r.2,+P0> <@H=2>ef,gh<$>
<f1r.3,@Lx> latin.margin
<f1r.4,!L0> invalid
<f1v> <! $I=T>
<f1v.1,@P0> ij.kl
"""
        pages, report = parse_ivtff(raw)
        self.assertEqual(pages[0]["text"], "ab cd\nef" + UNCERTAIN_SPACE + "gh")
        self.assertEqual(pages[0]["loci"][1]["text_tags"], {"H": "2"})
        self.assertEqual(pages[1]["loci"][0]["text_tags"], {})
        self.assertEqual(pages[0]["leaf_id"], pages[1]["leaf_id"])
        self.assertEqual(len(report["excluded_loci"]), 2)
        self.assertEqual(pages[0]["loci"][1]["page_start"], 6)

    def test_continuation_keeps_one_locus(self):
        pages, _ = parse_ivtff(HEADER + "<f1r>\n<f1r.1,@P0> ab.cd /\n/ ef.gh\n")
        self.assertEqual(pages[0]["text"], "ab cdef gh")
        self.assertEqual(len(pages[0]["loci"]), 1)

    def test_bad_records_fail_closed(self):
        cases = ["not IVTFF", HEADER + "<f1r.1,@P0> ab", HEADER + "<f1r>\n<f2r.1,@P0> ab",
                 HEADER + "<f1r>\n<f1r.1,@P0> ab /\n# comment\n/ cd",
                 HEADER + "<f1r>\n<f1r.1,@P0> ab\n<f1r.1,+P0> cd",
                 HEADER + "<f1r>\n<f1r.1,@P0> ab\n<f1r>\n",
                 HEADER + "<f1r>\nHTML goes here"]
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_ivtff(content)

    def test_interlinear_requires_one_explicit_stream(self):
        raw = HEADER + "<f1r>\n<f1r.1,@P0;T> ab\n<f1r.1,@P0;Z> cd\n"
        with self.assertRaisesRegex(ValueError, "single transcriber"):
            parse_ivtff(raw)
        pages, _ = parse_ivtff(raw, transcriber="T")
        self.assertEqual(pages[0]["text"], "ab")

    def test_rosettes_foldout_is_one_group(self):
        self.assertEqual({leaf_id(x) for x in ("f85r1", "f86v4", "fRos")}, {"f85-f86-Ros"})
        self.assertEqual(leaf_id("f72v3"), "f72")


class SplitTests(unittest.TestCase):
    def test_feasible_section_strata_cover_all_splits_without_breaking_leaves(self):
        pages = [{"page_id": f"f{i}r", "leaf_id": f"f{i}", "text": chr(0x500 + i) * 24,
                  "loci": [], "metadata": {"page_variables": {"I": section}}}
                 for i, section in enumerate(["H"] * 10 + ["B"] * 10 + ["P"] * 10, 1)]
        pages.append({**pages[0], "page_id": "f1v", "text": "another side"})
        result = assign_splits(pages)
        self.assertEqual(result, assign_splits(list(reversed(pages))))
        for section in ("H", "B", "P"):
            self.assertEqual({p["split"] for p in pages if p["metadata"]["page_variables"]["I"] == section},
                             {"train", "validation", "test"})
        self.assertEqual(pages[0]["split"], pages[-1]["split"])
        self.assertEqual(sum(p["split"] == "validation" for p in pages[:-1]), 3)
        self.assertEqual(sum(p["split"] == "test" for p in pages[:-1]), 3)
        leakage_audit(pages)

    def test_rare_section_policy_is_explicit(self):
        sections = ["H"] * 16 + ["A", "A", "Z", "P"]
        pages = [{"page_id": f"f{i}r", "leaf_id": f"f{i}", "text": chr(0x500 + i) * 24,
                  "loci": [], "metadata": {"page_variables": {"I": section}}}
                 for i, section in enumerate(sections, 1)]
        report = assign_splits(pages)["stratification"]
        self.assertEqual(report["section_achieved_by_component"]["A"],
                         {"train": 1, "validation": 1, "test": 0})
        self.assertEqual(report["section_achieved_by_component"]["P"],
                         {"train": 1, "validation": 0, "test": 0})
        self.assertEqual(report["rare_sections"], {"A": 2, "P": 1, "Z": 1})

    def test_infeasible_coverage_requires_explicit_redesign(self):
        pages = synthetic_pages()[:6]
        for index, page in enumerate(pages):
            page["metadata"] = {"page_variables": {"I": "H" if index < 3 else "B"}}
        with self.assertRaisesRegex(ValueError, "coverage is infeasible"):
            assign_splits(pages)

    def test_split_is_deterministic_under_page_reordering_and_groups_duplicates(self):
        pages = synthetic_pages()
        pages.append({"page_id": "f1v", "leaf_id": "f1", "text": "other text", "loci": []})
        pages[1]["text"] = pages[0]["text"]  # Exact duplicate on a different leaf.
        first = assign_splits(pages)
        second = assign_splits(list(reversed(pages)))
        self.assertEqual(first, second)
        self.assertEqual(pages[0]["split"], pages[1]["split"])
        self.assertEqual(pages[0]["split"], pages[-1]["split"])
        self.assertEqual(leakage_audit(pages)["cross_split_leaf_page_long_span_conflicts"], 0)

    def test_long_shared_span_merges_distinct_pages(self):
        pages = synthetic_pages()
        span = "abcdefgh" * 16
        pages[0]["text"] = "first" + span + "tail"
        pages[1]["text"] = "second" + span + "end"
        assign_splits(pages)
        self.assertEqual(pages[0]["split"], pages[1]["split"])
        leakage_audit(pages)

    def test_leakage_audit_rejects_broken_leaf_assignment(self):
        pages = synthetic_pages()
        assign_splits(pages)
        pages.append({"page_id": "f1v", "leaf_id": "f1", "text": "different", "loci": [],
                      "split": "test" if pages[0]["split"] != "test" else "train"})
        with self.assertRaisesRegex(ValueError, "contamination"):
            leakage_audit(pages)


class TokenizerTests(unittest.TestCase):
    def test_train_only_vocabulary_and_atomic_rare_units(self):
        text = "ab " + chr(RARE_BASE + 185) + "\n" + ALTERNATIVE
        tokenizer = EVATokenizer.fit([text])
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.encode("z", False, False), [tokenizer.unk_id])
        self.assertEqual(len(tokenizer.encode(chr(RARE_BASE + 185), False, False)), 1)
        self.assertIn(tokenizer.unk_id, tokenizer.uncertainty_ids)
        self.assertNotIn(tokenizer.space_id, tokenizer.uncertainty_ids)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tokenizer.json"
            tokenizer.save(path)
            self.assertEqual(EVATokenizer.load(path).tokens, tokenizer.tokens)


class PreparationTests(unittest.TestCase):
    def test_rebuild_is_identical_and_source_and_split_are_pinned(self):
        raw = HEADER + "".join(f"<f{i}r>\n<f{i}r.1,@P0> {chr(96+i) * 20}\n" for i in range(1, 11))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            raw_path = base / "input.txt"
            raw_path.write_text(raw)
            manifests = base / "manifests"
            manifests.mkdir()
            (manifests / "zl3b_source.json").write_text(json.dumps({"sha256": sha256_bytes(raw.encode())}))
            first = prepare(raw_path, base / "output", manifests)
            second = prepare(raw_path, base / "output", manifests)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "Frozen split"):
                prepare(raw_path, base / "output", manifests, seed="different")
            raw_path.write_text(raw + "# modified\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare(raw_path, base / "output", manifests)


if __name__ == "__main__":
    unittest.main()
