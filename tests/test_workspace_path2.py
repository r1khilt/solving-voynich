import numpy as np

from voynich.workspace.path2_campaign import field_conditions
from voynich.workspace.path2_tasks import BUNDLES, path2_tasks
from voynich.workspace.path_tasks import CONFIRMATION, DISCOVERY


def test_new_bundles_are_disjoint_and_reverse_paired():
    rows = path2_tasks()
    assert len(rows) == 72
    previous = {word for bundle in DISCOVERY+CONFIRMATION for word in bundle[:4]}
    current = {word for bundle in BUNDLES for word in bundle[:4]}
    assert not previous & current
    assert {r['template'] for r in rows} == {0, 1, 2}
    lookup = {r['prompt']: r for r in rows}
    for row in rows:
        donor = lookup[row['donor_prompt']]
        assert donor['donor_prompt'] == row['prompt']
        assert donor['answer'] == row['donor_answer']
        assert donor['family'] == row['family']


def test_registered_fields_partition_the_donor_displacement():
    delta = np.random.default_rng(5).normal(size=(7, 16)).astype(np.float32)
    positions = [2, 4]
    fields = field_conditions(delta, positions, seed=51051)
    np.testing.assert_array_equal(fields['value_donor']+fields['other_earlier_donor']+fields['last_donor'], delta)
    np.testing.assert_array_equal(fields['earlier_donor']+fields['last_donor'], delta)
    for position in range(len(delta)):
        if position in positions:
            np.testing.assert_allclose(np.linalg.norm(fields['value_random'][position]),
                                       np.linalg.norm(delta[position]), rtol=1e-6)
        else:
            np.testing.assert_array_equal(fields['value_random'][position], np.zeros(16))
