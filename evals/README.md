# Evaluation

## The set: `eval_set.jsonl`

25 questions over the FastAPI docs. **Frozen**: curated once, never edited because
a config change made an item fail. Fields per item:

| field | purpose |
|---|---|
| `question` | the input |
| `expected_category` | for router accuracy (`api_reference` / `conceptual` / `troubleshooting`) |
| `reference_urls` | doc pages that *should* be retrieved, used for deterministic recall@k. Empty for negatives. |
| `difficulty` | `single_chunk` / `multi_chunk` / `negative` |
| `split` | `dev` (tune against this) / `holdout` (run once at the end) |

Composition: 17 `dev`, 8 `holdout`; roughly balanced across the three categories;
3 negatives whose answers are genuinely not in the FastAPI docs (rate limiting,
MongoDB ODM choice, a non-existent built-in auth DB); the correct behaviour there
is refusal, scored by `negative_refusal_rate`.

### How it was built (leakage guard)

Questions are written against FastAPI *concepts and real user problems*, then
retrieval has to find support, never by paraphrasing a specific chunk. No gold
answer prose is stored, so RAGAS scoring can't overfit to one phrasing (we use the
reference-free metrics + our own URL-based recall). See `docs/architecture.md` §3.11.

## Running

```bash
python -m docqa.eval.run --split dev --label baseline-v1     # first honest run -> commit it
# ...tune chunking / retrieval params in config/corpora/fastapi.yaml...
python -m docqa.eval.run --split dev                          # auto-diffs vs newest baseline-*.json
python -m docqa.eval.run --split holdout --label holdout-final
```

**Rate limits.** One run makes dozens of LLM calls (≈4 per item for the pipeline +
≈3 per item for the RAGAS judge). Hosted free tiers (Gemini free: 5 RPM / 20-1000
RPD depending on model) can't sustain a tuning loop. Point the whole run at a local
model instead:

```bash
python -m docqa.eval.run --split dev --label baseline-v1 --provider ollama
```

`--provider` overrides `LLM_PROVIDER` for both the pipeline generator and the judge.
Needs Ollama running with a model pulled (`ollama pull llama3.1`). Slower per call
on CPU but unmetered and $0.

`--no-ragas` runs only the deterministic metrics (recall@k, router accuracy,
refusal rate), no LLM judge, so ~4 calls/item instead of ~7.

## Metrics

| metric | source | meaning |
|---|---|---|
| `faithfulness` | RAGAS (LLM judge) | answer claims entailed by retrieved context |
| `answer_relevancy` | RAGAS (judge + bge-small embeddings) | answer addresses the question |
| `context_precision` | RAGAS (`LLMContextPrecisionWithoutReference`) | retrieved chunks relevant & well-ranked |
| `retrieval_recall_at_k` | deterministic | fraction of `reference_urls` present in retrieved chunks |
| `router_accuracy` | deterministic | `expected_category` == routed category |
| `negative_refusal_rate` | deterministic (cue heuristic) | negatives answered with a refusal |
| `fallback_rate` | deterministic | share of queries that hit the fallback agent (lower better) |

Judge model is pinned (`EVAL_JUDGE_MODEL`) so scores are comparable across runs.
It's currently the same model as the generator, a known self-preference risk,
noted in the architecture doc; swap it for a different judge if you have a key.

## `results/`

`baseline-*.json` is committed; every other run is gitignored working output.
