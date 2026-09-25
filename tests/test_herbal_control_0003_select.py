"""The large historical panel must be selected without prior pages or shared pages."""

from scripts.herbal_control_0003_select import candidate_classes, page_disjoint_set


def test_exposed_and_shared_pages_cannot_enter_panel() -> None:
    entries = {
        "alpha": {
            "bnf": [{"pageid": 1, "title": "File:B f.001r.jpg"}],
            "egerton": [{"pageid": 2, "title": "File:E f.001r.jpg"}],
            "casanatense": [{"pageid": 3, "title": "File:C f.001r.jpg"}],
        },
        "beta": {
            "bnf": [{"pageid": 1, "title": "File:B f.001r.jpg"}],
            "egerton": [{"pageid": 4, "title": "File:E f.002r.jpg"}],
            "casanatense": [{"pageid": 5, "title": "File:C f.002r.jpg"}],
        },
        "gamma": {
            "bnf": [{"pageid": 6, "title": "File:B f.003r.jpg"}],
            "egerton": [{"pageid": 7, "title": "File:E f.003r.jpg"}],
            "casanatense": [{"pageid": 8, "title": "File:C f.003r.jpg"}],
        },
    }
    files = [{"pageid": row["pageid"], "chapters": [chapter]}
             for chapter, pages in entries.items() for rows in pages.values() for row in rows]
    inventory = {"files": files, "three_manuscript_chapters": entries}
    candidates, rejected = candidate_classes(inventory, {"File:B f.003r.jpg"})
    assert rejected["exposed_page"] == 1
    assert set(candidates) == {"alpha", "beta"}
    assert page_disjoint_set(candidates) == ["alpha"]
