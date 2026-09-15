"""Load markdown docs from a pinned GitHub tag.

Why a tarball and not `git clone`: no git binary dependency, no full history
download, trivially cacheable, and pinning `ref` to a tag makes every re-ingest
byte-for-byte reproducible — which matters because the eval baseline must be
compared against the *same* corpus after tuning.
"""

from __future__ import annotations

import functools
import io
import re
import tarfile
from collections.abc import Iterable
from pathlib import PurePosixPath

import httpx

from docqa.config import REPO_ROOT, LoaderConfig
from docqa.ingestion.loaders._markdown import first_heading, light_clean
from docqa.ingestion.loaders.base import RawDoc

_CODELOAD = "https://codeload.github.com/{repo}/tar.gz/refs/tags/{ref}"


@functools.lru_cache(maxsize=128)
def _glob_re(pattern: str) -> re.Pattern[str]:
    """gitignore/rsync-style globbing: `*` matches within a path segment, `**`
    crosses `/`. stdlib `fnmatch` has neither property (its `*` eats `/`), which
    silently dropped every top-level `docs/en/docs/*.md` under a `**/*.md` include."""
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(_glob_re(p).match(path) for p in patterns)


class GithubMarkdownLoader:
    def __init__(self, cfg: LoaderConfig) -> None:
        if not cfg.repo or not cfg.ref:
            raise ValueError("github_markdown loader requires 'repo' and 'ref'")
        self.cfg = cfg

    # -- fetch + cache -------------------------------------------------------
    def _tarball_bytes(self) -> bytes:
        cache_dir = REPO_ROOT / "data" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{self.cfg.repo.replace('/', '__')}-{self.cfg.ref}.tar.gz"
        if cached.exists():
            return cached.read_bytes()
        url = _CODELOAD.format(repo=self.cfg.repo, ref=self.cfg.ref)
        resp = httpx.get(url, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        cached.write_bytes(resp.content)
        return resp.content

    # -- filtering --------------------------------------------------------- #
    def _wanted(self, rel: str) -> bool:
        includes = self.cfg.include_globs or ["**/*.md"]
        if not _matches_any(rel, includes):
            return False
        return not _matches_any(rel, self.cfg.exclude_globs)

    # -- main ------------------------------------------------------------- #
    def load(self) -> Iterable[RawDoc]:
        with tarfile.open(fileobj=io.BytesIO(self._tarball_bytes()), mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) < 2:  # first component is "<repo>-<ref>/"
                    continue
                rel = "/".join(parts[1:])
                if not self._wanted(rel):
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                yield self._to_rawdoc(rel, fh.read().decode("utf-8", errors="replace"))

    def _to_rawdoc(self, rel: str, text: str) -> RawDoc:
        prefix = self.cfg.strip_path_prefix
        docpath = rel[len(prefix):] if prefix and rel.startswith(prefix) else rel
        slug = docpath[:-3] if docpath.endswith(".md") else docpath
        if slug.endswith("/index"):
            slug = slug[: -len("index")]
        elif slug == "index":
            slug = ""

        url = self.cfg.url_base.rstrip("/") + "/" + slug
        if slug and not url.endswith("/"):
            url += "/"

        stem = PurePosixPath(docpath).stem
        title = first_heading(text) or stem.replace("-", " ").title()
        section_path = [p.replace("-", " ").title() for p in PurePosixPath(docpath).parts[:-1]]

        return RawDoc(
            source_id=slug or docpath,
            title=title,
            url=url,
            body_markdown=light_clean(text),
            section_path=section_path,
        )
