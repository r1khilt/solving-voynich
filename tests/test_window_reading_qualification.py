"""One artificial preparation panel exercises alternate families and RNG audit."""
from scripts.check_window_reading001 import DEGREES, STEPS, categorical_qualification, finite_panel


def test_small_artificial_window_panel_alternate_map_and_decisions():
    result = finite_panel(3, True, ((0, 0),))
    assert result['states'] == result['cold_stationarity_equations'] > 0
    assert result['split_components'] == result['merge_components'] > 0
    assert result['exact_power_flux_checks'] == len(DEGREES)*result['augmented_components']
    assert result['wrong_prior_flux_failures'] and result['wrong_continuation_flux_failures']
    assert result['uncorrected_valid_anchor_flux_failures'] and result['sole_window_row_rebindings']
    assert result['seeded_steps'] == STEPS and result['independent_root_decisions'] > 0
    assert result['heatbath_stationarity_equations'] == result['states']
    assert result['seeded_heatbath_steps'] == STEPS and result['independent_categorical_draws'] > 0
    assert result['exact_incremental_target_checks'] == result['valid_components']


def test_separate_rational_categorical_quantile_and_cap_qualification():
    assert categorical_qualification() == {'independent_categorical_decisions': 1024,
                                         'two_block_and_abort_witnesses': 4}
