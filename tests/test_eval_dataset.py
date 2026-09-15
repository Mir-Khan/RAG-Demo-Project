"""The committed eval set must stay well-formed."""

from __future__ import annotations

import yaml

from docqa.config import CORPORA_DIR
from docqa.eval.dataset import DEFAULT_PATH, load_eval_set

ITEMS = load_eval_set(DEFAULT_PATH, split="all")
FASTAPI_CATEGORIES = {
    c["id"] for c in yaml.safe_load((CORPORA_DIR / "fastapi.yaml").read_text())["router_categories"]
}


def test_size_in_spec_range():
    assert 20 <= len(ITEMS) <= 30


def test_ids_unique():
    assert len({i.id for i in ITEMS}) == len(ITEMS)


def test_expected_categories_exist_in_corpus_config():
    for it in ITEMS:
        assert it.expected_category in FASTAPI_CATEGORIES, it.id


def test_negatives_have_no_reference_and_normals_do():
    for it in ITEMS:
        if it.difficulty == "negative":
            assert it.reference_urls == [], it.id
        else:
            assert it.reference_urls, it.id
            assert all(u.startswith("https://fastapi.tiangolo.com/") for u in it.reference_urls)


def test_has_a_holdout_slice_and_all_categories_and_negatives():
    assert sum(i.split == "holdout" for i in ITEMS) >= 5
    assert sum(i.split == "dev" for i in ITEMS) >= 10
    assert {i.expected_category for i in ITEMS} == FASTAPI_CATEGORIES
    assert sum(i.difficulty == "negative" for i in ITEMS) >= 2


def test_split_filter():
    dev = load_eval_set(DEFAULT_PATH, split="dev")
    holdout = load_eval_set(DEFAULT_PATH, split="holdout")
    assert len(dev) + len(holdout) == len(ITEMS)
    assert all(i.split == "dev" for i in dev)
