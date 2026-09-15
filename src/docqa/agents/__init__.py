"""Multi-agent orchestration (LangGraph). Import the graph lazily so unit tests of
context/router don't require langgraph."""

from docqa.agents.state import AnswerResult, Citation, QAState

__all__ = ["AnswerResult", "Citation", "QAState", "QAPipeline", "build_pipeline"]


def __getattr__(name: str):  # PEP 562
    if name in ("QAPipeline", "build_pipeline"):
        from docqa.agents import graph

        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
