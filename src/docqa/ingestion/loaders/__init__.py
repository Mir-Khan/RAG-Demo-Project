"""Loader registry. Add a corpus type by writing a class with a `.load()` method
that yields `RawDoc`s and registering it here."""

from __future__ import annotations

from docqa.config import LoaderConfig
from docqa.ingestion.loaders.base import RawDoc, SourceLoader
from docqa.ingestion.loaders.github_markdown import GithubMarkdownLoader
from docqa.ingestion.loaders.markdown_dir import MarkdownDirLoader

_REGISTRY: dict[str, type] = {
    "github_markdown": GithubMarkdownLoader,
    "markdown_dir": MarkdownDirLoader,
}

__all__ = ["RawDoc", "SourceLoader", "build_loader"]


def build_loader(cfg: LoaderConfig) -> SourceLoader:
    try:
        cls = _REGISTRY[cfg.type]
    except KeyError:
        raise ValueError(
            f"Unknown loader type {cfg.type!r}. Known types: {sorted(_REGISTRY)}"
        ) from None
    return cls(cfg)
