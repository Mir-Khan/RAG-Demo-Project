"""Wikitext -> markdown cleanup. Fixtures are hand-written, not fetched, so this
runs offline and stays stable regardless of what Bulbapedia's live content does."""

from __future__ import annotations

from docqa.ingestion.loaders._wikitext import wikitext_to_markdown


def test_headings_convert_and_dont_collide_with_wikitext_lists():
    src = "==Summary==\nSome text.\n\n#First step\n#Second step\n\n===Detail===\nMore."
    out = wikitext_to_markdown(src)
    assert "## Summary" in out
    assert "### Detail" in out
    assert "- First step" in out and "- Second step" in out
    # no line should start with a bare "#" left over from the wikitext list marker
    assert not any(line.startswith("#") and not line.startswith(("##", "###")) for line in out.splitlines())


def test_wikilinks_and_emphasis():
    src = "A [[Bulbasaur|Grass-type]] Pokémon. It is '''strong''' and ''fast''."
    out = wikitext_to_markdown(src)
    assert "[[" not in out and "]]" not in out
    assert "Grass-type" in out and "Bulbasaur" not in out  # display text wins
    assert "**strong**" in out and "*fast*" in out


def test_templates_keep_display_text_and_drop_bare_markers():
    src = "The {{t|Fire}} type is strong. {{CURRENTGEN}} As of now."
    out = wikitext_to_markdown(src)
    assert "Fire" in out and "{{" not in out
    assert "CURRENTGEN" not in out


def test_refs_comments_and_tables_stripped():
    src = (
        "Text with a note.<ref>some citation</ref> More text.\n"
        "<!-- editorial comment -->\n"
        "{|\n|-\n|cell1||cell2\n|}\n"
        "After the table."
    )
    out = wikitext_to_markdown(src)
    assert "citation" not in out
    assert "editorial comment" not in out
    assert "cell1" not in out
    assert "After the table." in out


def test_file_links_removed():
    src = "Intro.\n[[File:Bulbasaur.png|thumb|A Bulbasaur.]]\nMore text."
    out = wikitext_to_markdown(src)
    assert "File:" not in out and "Bulbasaur.png" not in out
    assert "More text." in out
