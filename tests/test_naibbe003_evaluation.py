from scripts.evaluate_naibbe003 import decision


def test_stricter_rare_key_gate_is_preserved_despite_good_plaintext():
    primary = {"cer": .001, "key_macro_accuracy": 130 / 138,
               "unique_class_weighted_key_accuracy": .999}
    checks, verdict = decision(primary, {"cer": .0003})
    assert verdict == "FAIL"
    assert [name for name, passed in checks.items() if not passed] == ["macro_key_at_least_95"]
    primary["key_macro_accuracy"] = 132 / 138
    assert decision(primary, {"cer": .0003})[1] == "PASS"
