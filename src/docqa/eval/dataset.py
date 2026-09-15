"""The frozen evaluation set.

One JSONL file, `evals/eval_set.jsonl`, committed to the repo. It is curated once
and NOT edited because a config change made an item fail — that's the whole point
of a fixed target. `split` partitions it: tune against `dev`, run `holdout` once
at the end to check the tuning generalised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from docqa.config import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "evals" / "eval_set.jsonl"

Difficulty = Literal["single_chunk", "multi_chunk", "negative"]
Split = Literal["dev", "holdout"]


class EvalItem(BaseModel):
    id: str
    question: str
    expected_category: str
    reference_urls: list[str] = Field(default_factory=list)  # empty for `negative` items
    difficulty: Difficulty
    split: Split = "dev"
    notes: str = ""

    @property
    def is_negative(self) -> bool:
        return self.difficulty == "negative"


def load_eval_set(path: str | Path = DEFAULT_PATH, split: str = "all") -> list[EvalItem]:
    path = Path(path)
    items: list[EvalItem] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            items.append(EvalItem.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 - want the line number in the message
            raise ValueError(f"{path}:{lineno} invalid eval item: {exc}") from exc

    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate ids in {path}")

    if split != "all":
        items = [i for i in items if i.split == split]
    return items
