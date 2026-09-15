"""Reciprocal Rank Fusion math — pure, no DB."""

from __future__ import annotations

from docqa.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_beats_item_in_one():
    dense = ["a", "b", "c"]
    sparse = ["b", "d", "e"]
    scores = reciprocal_rank_fusion([dense, sparse], k=60)
    # b is rank 2 (dense) + rank 1 (sparse); a is only rank 1 (dense)
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["d"]


def test_score_formula_is_sum_of_reciprocal_ranks():
    scores = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=10)
    # x: 1/(10+1) + 1/(10+2);  y: 1/(10+2) + 1/(10+1)  -> equal
    assert abs(scores["x"] - scores["y"]) < 1e-12
    assert abs(scores["x"] - (1 / 11 + 1 / 12)) < 1e-12


def test_higher_k_compresses_differences():
    lists = [["a", "b", "c", "d", "e"]]
    tight = reciprocal_rank_fusion(lists, k=1)
    loose = reciprocal_rank_fusion(lists, k=1000)
    assert (tight["a"] - tight["e"]) > (loose["a"] - loose["e"])


def test_ranking_order_is_recoverable():
    dense = ["doc1", "doc2", "doc3", "doc4"]
    sparse = ["doc3", "doc1", "doc9"]
    scores = reciprocal_rank_fusion([dense, sparse], k=60)
    order = sorted(scores, key=scores.get, reverse=True)
    # doc1 (1,2) and doc3 (3,1) are the two agreed-upon docs -> they lead
    assert set(order[:2]) == {"doc1", "doc3"}
    # doc4 appears once at dense rank 4 (1/64); doc9 once at sparse rank 3 (1/63) -> doc4 is last
    assert order[-1] == "doc4"


def test_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == {}
    assert reciprocal_rank_fusion([]) == {}
