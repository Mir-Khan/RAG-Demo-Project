"""Load pages from a live MediaWiki instance via its public API.

Why not a tarball (like github_markdown): a wiki isn't version-controlled the
way a docs repo is, so there's nothing to pin a `ref` to. Instead we pin an
explicit `pages` list — reproducible in the sense that matters here (the same
config always asks for the same pages), and it keeps the corpus small and
curated rather than a full-site crawl. Uses the standard `action=parse` API
with a descriptive User-Agent, one request per page, cached to disk exactly
like the GitHub tarball is — polite to the source and fast to re-ingest.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import httpx

from docqa.config import REPO_ROOT, LoaderConfig
from docqa.ingestion.loaders._wikitext import wikitext_to_markdown
from docqa.ingestion.loaders.base import RawDoc

_USER_AGENT = "docqa-portfolio-project/0.1 (educational RAG demo; see repo README)"


class MediaWikiLoader:
    def __init__(self, cfg: LoaderConfig) -> None:
        if not cfg.api_base or not cfg.pages:
            raise ValueError("mediawiki loader requires 'api_base' and a non-empty 'pages' list")
        self.cfg = cfg

    def _cache_path(self, title: str):
        safe = title.replace("/", "_").replace(" ", "_")
        cache_dir = REPO_ROOT / "data" / "cache" / "mediawiki"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{safe}.json"

    def _fetch(self, client: httpx.Client, title: str) -> dict | None:
        cache = self._cache_path(title)
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        resp = client.get(
            self.cfg.api_base,
            params={
                "action": "parse",
                "page": title,
                "prop": "wikitext",
                "format": "json",
                "formatversion": 2,
                # follow redirects server-side (e.g. "Terastallization" -> "Terastal
                # phenomenon") instead of ingesting a one-line "#REDIRECT [[...]]" stub
                "redirects": 1,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return None  # missing/renamed page -- skip rather than fail the whole ingest
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data

    def load(self) -> Iterable[RawDoc]:
        with httpx.Client(headers={"User-Agent": _USER_AGENT}, timeout=30.0) as client:
            for title in self.cfg.pages:
                data = self._fetch(client, title)
                if data is None:
                    continue
                page_title = data["parse"]["title"]
                wikitext = data["parse"]["wikitext"]
                slug = page_title.replace(" ", "_")
                url = self.cfg.url_base.rstrip("/") + "/" + slug
                yield RawDoc(
                    source_id=slug,
                    title=page_title,
                    url=url,
                    body_markdown=wikitext_to_markdown(wikitext),
                    section_path=[],
                )
