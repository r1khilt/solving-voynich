"""Source transport checks for the fixed historical image panel."""

import pytest

from scripts.herbal_control_0003_fetch import commons_upload_url, requested_sources


def test_upload_route_matches_commons_original_link() -> None:
    # Commons' original-file link for BnF Latin 6823 f011v, independently
    # visible on its public file page, has this exact path.
    assert commons_upload_url("BnF_Latin_6823,_f.011v.jpg") == (
        "https://upload.wikimedia.org/wikipedia/commons/7/73/"
        "BnF_Latin_6823%2C_f.011v.jpg")
    with pytest.raises(ValueError, match="ASCII"):
        commons_upload_url("../bad.jpg")


def test_requested_sources_rejects_reused_page() -> None:
    first = {"chapter": "a", "pages": {
        m: {"pageid": i, "title": f"File:{m} f.001r.jpg"}
        for i, m in enumerate(("bnf", "egerton", "casanatense"), 1)}}
    duplicate = {"chapter": "b", "pages": first["pages"]}
    panel = {"roles": {"development_known": [first, duplicate],
                       "development_unknown": [], "evaluation_known": [],
                       "evaluation_unknown": []}}
    with pytest.raises(ValueError, match="repeated source page"):
        requested_sources(panel)
