"""Offline checks for the Commons metadata inventory parser."""

import pytest

from scripts.herbal_open_inventory import build_inventory, chapter_tags, parse_pages


def test_chapter_tags_use_api_space_titles_and_deduplicate() -> None:
    assert chapter_tags([
        "Category:BnF Latin 6823",
        "Category:De arthemisia in Tractatus de herbis",
        "Category:De_arthemisia_in_Tractatus_de_herbis",
        "Category:De arthemisia tagantes in Tractatus de herbis",
        "Category:PD-Art (PD-old-70)",
    ]) == ["arthemisia", "arthemisia tagantes"]


def test_inventory_requires_all_three_manuscripts_and_keeps_multiple_files() -> None:
    records = [
        {"manuscript": "bnf", "pageid": 1, "title": "File:B full.jpg"},
        {"manuscript": "bnf", "pageid": 2, "title": "File:B detail.jpg"},
        {"manuscript": "egerton", "pageid": 3, "title": "File:E.jpg"},
        {"manuscript": "casanatense", "pageid": 4, "title": "File:C.jpg"},
    ]
    categories = {
        1: ["Category:De iris in Tractatus de herbis"],
        2: ["Category:De iris in Tractatus de herbis"],
        3: ["Category:De iris in Tractatus de herbis"],
        4: ["Category:De iris in Tractatus de herbis"],
    }
    inventory = build_inventory(records, categories, [])
    assert inventory["counts"]["three_manuscript_chapter_tags"] == 1
    assert len(inventory["three_manuscript_chapters"]["iris"]["bnf"]) == 2


def test_parse_pages_rejects_missing_batch_page() -> None:
    with pytest.raises(ValueError, match="differ"):
        parse_pages({"query": {"pages": [{"pageid": 1, "categories": []}]}},
                    [{"pageid": 1}, {"pageid": 2}])
