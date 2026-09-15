"""Path globbing for the github_markdown loader — `**` crosses `/`, `*` does not."""

from __future__ import annotations

from docqa.ingestion.loaders.github_markdown import _matches_any


def test_double_star_matches_zero_or_more_dirs():
    pat = ["docs/en/docs/**/*.md"]
    assert _matches_any("docs/en/docs/async.md", pat)               # zero dirs — was the bug
    assert _matches_any("docs/en/docs/tutorial/body.md", pat)
    assert _matches_any("docs/en/docs/how-to/a/b/c.md", pat)
    assert not _matches_any("docs/en/mkdocs.yml", pat)
    assert not _matches_any("scripts/docs.py", pat)


def test_single_star_does_not_cross_slash():
    assert _matches_any("a/b.md", ["a/*.md"])
    assert not _matches_any("a/b/c.md", ["a/*.md"])


def test_exclude_patterns():
    pat = ["docs/en/docs/reference/**"]
    assert _matches_any("docs/en/docs/reference/response.md", pat)
    assert _matches_any("docs/en/docs/reference/deep/x.md", pat)
    assert not _matches_any("docs/en/docs/tutorial/response-model.md", pat)


def test_release_notes_exclude():
    assert _matches_any("docs/en/docs/release-notes.md", ["**/release-notes.md"])
    assert _matches_any("docs/en/docs/x/release-notes.md", ["**/release-notes.md"])
