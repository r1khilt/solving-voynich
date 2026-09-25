"""Known-coordinate controls for the metadata-only folio challenger."""

import pytest

from voynich.herbal_folio_order import fit_development_maps, folio_distance_matrix, folio_side


def test_folio_side_rejects_qualified_or_missing_titles() -> None:
    assert folio_side("File:BnF Latin 6823, f.021r.jpg") == 42
    assert folio_side("File:BnF Latin 6823, f.021v.jpg") == 43
    with pytest.raises(ValueError, match="unqualified"):
        folio_side("File:BnF Latin 6823, f.021v2.jpg")
    with pytest.raises(ValueError, match="unqualified"):
        folio_side("File:unknown plant.jpg")


def test_theil_sen_map_recovers_cross_book_order_despite_one_bad_page() -> None:
    manuscripts = ("bnf", "egerton", "casanatense")
    roles = {role: [] for role in (
        "development_known", "development_unknown", "evaluation_known", "evaluation_unknown")}
    for index in range(36):
        bnf = 100 + 2 * index
        egerton = 2 * bnf + 5 + (300 if index == 23 else 0)
        casa = 3 * bnf + 7
        coordinates = dict(zip(manuscripts, (bnf, egerton, casa), strict=True))
        item = {"chapter": f"class_{index:02d}", "pages": {
            m: {"title": f"File:{m}, f.{coordinate // 2:03d}{'v' if coordinate % 2 else 'r'}.jpg"}
            for m, coordinate in coordinates.items()}}
        roles["development_known" if index < 24 else "development_unknown"].append(item)
    panel = {"roles": roles}
    maps = fit_development_maps(panel)
    assert maps["bnf->egerton"]["slope"] == 2.0
    assert maps["bnf->egerton"]["intercept"] == 5.0
    assert maps["bnf->casanatense"]["slope"] == 3.0
    assert maps["bnf->casanatense"]["intercept"] == 7.0
    matrix, rows = folio_distance_matrix(panel, "development", maps)
    assert len(rows) == len(matrix) == 108
    assert all(len(row) == 108 for row in matrix)
    # A known class with no coordinate corruption lands on its own reference.
    assert matrix[0][1] == 0.0
    assert matrix[0][2] == 0.0
