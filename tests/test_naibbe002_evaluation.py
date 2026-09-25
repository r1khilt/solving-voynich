from scripts.evaluate_naibbe002 import metrics


def test_unknown_table_draws_do_not_become_gold_frequency_labels():
    data = {"class_ids": ["a", "b", "c", "d"], "alphabet": ["x", "y"],
            "groups": [["a", "b"], ["c", "d"]],
            "split": {"candidates": [[["a"], ["c"]], [["a", "d"]]]}}
    result = metrics(["x", "xx"], ["x", "xy"], data, [0, 1, 1, 0], [0, 1, 0, 1])
    assert result["unique_class_characters"] == 2
    assert result["unique_class_weighted_key_accuracy"] == .5
    assert result["key_macro_accuracy"] == .5
    assert result["per_group_key_correct"] == [2, 0]
    assert result["no_unique_evidence_classes"] == 2
    assert result["edit_distance"] == 1
