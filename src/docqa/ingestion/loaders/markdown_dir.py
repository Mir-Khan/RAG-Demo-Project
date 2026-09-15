"""Load a local folder of markdown files.

This is the "bring your own corpus" loader — point `path` at a directory of notes
(the Pokémon demo corpus uses this) and the entire downstream pipeline works
unchanged. Proof that the ingestion seam is real and not FastAPI-shaped.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from docqa.config import REPO_ROOT, LoaderConfig
from docqa.ingestion.loaders._markdown import first_heading, light_clean
from docqa.ingestion.loaders.base import RawDoc


class MarkdownDirLoader:
    def __init__(self, cfg: LoaderConfig) -> None:
        if not cfg.path:
            raise ValueError("markdown_dir loader requires 'path'")
        self.cfg = cfg
        self.root = (REPO_ROOT / cfg.path).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"markdown_dir path does not exist: {self.root}")

    def load(self) -> Iterable[RawDoc]:
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root).as_posix()
            slug = rel[:-3]
            text = path.read_text(encoding="utf-8")
            url = (
                self.cfg.url_base.rstrip("/") + "/" + slug
                if self.cfg.url_base
                else path.as_uri()
            )
            section_path = [p.replace("-", " ").title() for p in Path(rel).parts[:-1]]
            yield RawDoc(
                source_id=slug,
                title=first_heading(text) or path.stem.replace("-", " ").title(),
                url=url,
                body_markdown=light_clean(text),
                section_path=section_path,
            )
