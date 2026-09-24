from collections import Counter
import itertools
import math

from voynich.word_copy_channel import edit_parts, slots
from voynich.word_copy_followup import GlobalEdit, shuffled_slots


def test_global_edit_inverse_equals_direct_normalized_channel():
    counts = Counter({"aa": 2, "ab": 1})
    model = GlobalEdit(counts, 3, "ab")
    targets = [""] + ["".join(chars) for n in range(1, 4)
                      for chars in itertools.product("ab", repeat=n)]
    for target in targets:
        direct = 0.0
        for source, count in counts.items():
            exact, sub, ins, delete = edit_parts(source, target, 2)
            direct += count / 3 * (0.75 * exact + 0.125 * sub + 0.0625 * ins + 0.0625 * delete)
        assert math.isclose(model.prob(target), direct, abs_tol=1e-12)
    assert math.isclose(sum(model.prob(t) for t in targets), 1.0, abs_tol=1e-12)


def test_layout_permutations_keep_declared_inventories():
    page = {"page_id": "f1r", "loci": [
        {"locus_type": "P0", "text": "chol chor \ue000 daiin"},
        {"locus_type": "P0", "text": "chedy shedy"},
        {"locus_type": "L0", "text": "qokedy qokeey"},
    ]}
    original = slots(page)
    for mode in ("type", "locus"):
        altered = shuffled_slots([page], 280127, mode)["f1r"]
        assert Counter(altered) == Counter(original)
        assert [kind for _, kind in altered] == [kind for _, kind in original]
        assert altered[2] == ("\ue000", "P0")
        if mode == "locus":
            for start, end in ((0, 4), (4, 6), (6, 8)):
                assert Counter(altered[start:end]) == Counter(original[start:end])
