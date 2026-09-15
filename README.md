# Documentation Q&A Assistant — Multi-Agent RAG

A RAG system that answers questions over the FastAPI documentation. The goal was to build it the
way a production system would actually be built, not a tutorial pipeline — so it has hybrid
retrieval, a reranking step, a small multi-agent router, and an evaluation harness that measures
whether any of this is actually working.

What's in here:

- Section-aware chunking, driven by the docs' own heading hierarchy instead of fixed-size splitting.
- Hybrid retrieval: pgvector dense search plus Postgres full-text (BM25-style), fused with
  Reciprocal Rank Fusion, then a cross-encoder reranker before the LLM sees anything.
- Multi-agent orchestration with LangGraph. A router classifies each question and hands it to a
  specialized sub-agent (API reference, conceptual, or troubleshooting), all sharing one retrieval
  tool.
- An evaluation harness using RAGAS (faithfulness, answer relevancy, context precision) over a
  frozen eval set, with baseline-vs-tuned deltas committed under `evals/results/`.
- $0 to run: local `bge-small` embeddings and reranker, Gemini's free tier for generation (Groq or
  a local Ollama model both work too, one env var to switch), Postgres either in Docker or on a
  free managed tier.
- A second demo corpus, Pokémon games plus the current competitive format, to prove the ingestion
  side is actually corpus-agnostic and not just built to fit FastAPI. It pulls from a live wiki
  (Bulbapedia's API) blended with some hand-written notes, and none of the chunking, retrieval,
  agent, or eval code had to change to support it. Run with `CORPUS=pokemon`.

Every architectural decision here — what I considered instead, and why I picked what I picked — is
written up in [`docs/architecture.md`](docs/architecture.md). That doc is really the main artifact;
this README is more of a tour.

---

## Status

| Stage | State |
|---|---|
| 1. Ingestion pipeline (loaders, section-aware chunker, embeddings, pgvector store) | done |
| 2. Hybrid retrieval — dense + sparse + RRF + cross-encoder rerank | done |
| 3. LangGraph multi-agent orchestration — router + 3 sub-agents + fallback | done |
| 4. Evaluation harness — RAGAS + deterministic metrics, frozen 25-item set, baseline/diff | done |
| 5. Streamlit frontend — cited answers + per-stage retrieval transparency | done |
| 6. Deployment (Fly.io / Railway) | done |

---

## Repo layout

```
config/corpora/*.yaml     one file per knowledge base — loader, chunk budgets, router categories
src/docqa/
  config.py               env settings + corpus-config loading
  ingestion/
    loaders/               SourceLoader protocol + github_markdown / markdown_dir / mediawiki + registry
    chunker.py             markdown -> heading-stack sections -> token-budgeted chunks
    embed.py               bge-small wrapper (query-prefix, L2-normalise)
    pipeline.py            config -> load -> chunk -> embed -> upsert   (python -m docqa.ingestion.pipeline)
  db/
    schema.sql             single `chunks` table: vector(384) + tsvector, HNSW + GIN
    store.py               raw-SQL data access + vector_search / keyword_search (psycopg3)
  retrieval/
    fusion.py              Reciprocal Rank Fusion (pure, unit-tested)
    rerank.py              bge-reranker-base cross-encoder wrapper
    retriever.py           Retriever: dense + sparse -> RRF -> rerank -> final_k
    debug.py               python -m docqa.retrieval.debug "<query>"  (stage-by-stage tables)
  llm/
    base.py                LLM protocol (complete / json_mode) + ChatMessage
    gemini_provider.py  groq_provider.py  ollama_provider.py  fake.py   get_llm() factory
  agents/
    state.py               QAState (TypedDict) + AnswerResult + Citation
    context.py             assemble_context: chunks -> numbered [n] source block, token-capped
    router.py              parse_route: forgiving JSON parse -> category or fallback
    prompts.py             router + per-category generation prompts
    graph.py               QAPipeline: route -> retrieve -> {sub-agent | fallback} -> END
    run.py                 python -m docqa.agents.run "<query>" [--provider fake]
  eval/
    dataset.py             EvalItem + load_eval_set(split=dev|holdout|all)
    metrics.py             deterministic: recall@k, router accuracy, refusal, aggregate (no LLM)
    ragas_metrics.py       the only file that imports ragas — faithfulness / relevancy / precision
    run.py                 python -m docqa.eval.run --split dev --label baseline-v1
    diff.py                python -m docqa.eval.diff base.json current.json
  app/
    components.py          pure render helpers (score chips, citation status) — unit-tested
    main.py                streamlit run src/docqa/app/main.py
evals/
  eval_set.jsonl           the frozen 25-question set (committed)
  README.md                composition, leakage guard, metric definitions
  results/                 baseline-v1.json + holdout-final.json committed; other runs gitignored
docs/architecture.md      decision rationale
examples/pokemon/         second demo corpus: Bulbapedia (mediawiki loader) + curated competitive notes
```

---

## Quickstart (ingestion)

You need Python 3.13 (the ML stack doesn't have 3.14 wheels yet) and a Postgres 16 + pgvector
instance. Either run `docker compose up -d --wait` for a local one, or point `DATABASE_URL` at a
free managed instance — Neon works fine, just use the direct, non-pooled connection string.

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"           # ingestion deps only; retrieval/agents/eval/app are separate extras

copy .env.example .env            # set DATABASE_URL (Docker default works as-is); GOOGLE_API_KEY only needed later

# 1. chunk-only smoke test — no DB, no model download
python -m docqa.ingestion.pipeline --dry-run --limit 5

# 2. start Postgres + pgvector  (skip if using managed DATABASE_URL)
docker compose up -d --wait

# 3. full ingest (downloads the pinned FastAPI docs tarball + bge-small on first run)
python -m docqa.ingestion.pipeline --fresh

# swap corpora with one env var
set CORPUS=pokemon && python -m docqa.ingestion.pipeline --fresh
```

Tests need no network and no ML stack (the chunker falls back to a heuristic token counter, RRF is
pure math):

```bash
pytest -q
```

### Inspecting retrieval

Once a corpus is ingested, you can see what each retrieval stage actually does with a query:

```bash
python -m docqa.retrieval.debug "how do I return a 422 error" --corpus fastapi
python -m docqa.retrieval.debug "background tasks" --no-rerank      # pre-rerank order
```

It prints dense, sparse, fused, and reranked results as separate tables, which chunks each method
found that the other missed, and how far the reranker moved things around.

### Asking a question (multi-agent)

```bash
copy .env.example .env
# set GOOGLE_API_KEY (default provider, free: https://aistudio.google.com/apikey)
# or LLM_PROVIDER=groq + GROQ_API_KEY, or LLM_PROVIDER=ollama for a local model
python -m docqa.agents.run "How do I add CORS middleware?" --corpus fastapi
```

Shows the routing decision (category and confidence, or fallback), the sources it used, the cited
answer, and how long each node took.

### The web app

```bash
pip install -e ".[app]"
streamlit run src/docqa/app/main.py        # or: make app
```

Chat interface. Every answer comes with a caption showing the routed category and per-node timings,
plus a Sources panel listing each retrieved chunk with its dense/sparse/RRF/rerank score, a link to
the real doc page, and whether the answer actually cited it, included it without citing, or dropped
it when the context budget ran out. The embedder and reranker are shared across corpora and loaded
once, not reloaded when you switch.

---

## Design decisions, briefly

The full reasoning for each of these — including what else I considered and why I didn't go with it
— is in `docs/architecture.md`. Short version:

| Decision | Why |
|---|---|
| Section-aware chunking, 450/64/60 token budgets | Follows the author's own heading boundaries instead of cutting mid-thought. Stays under bge-small's 512-token limit once you account for the breadcrumb prefix. Overlap only happens when a section has to be split. |
| `bge-small-en-v1.5` (384-dim), run locally | Free, fast enough on CPU, and small vectors keep the index small. The reranker makes up most of the quality gap you'd get from a bigger embedding model anyway. |
| pgvector, one `chunks` table | Vectors sit next to the text and the `tsvector` column, so hybrid search is a single SQL query instead of syncing two separate stores. |
| HNSW index | Handles incremental inserts well and gives good recall for this size of corpus. IVFFlat would need periodic rebuilds as the data changes. |
| Dense + sparse hybrid | Dense catches paraphrasing; sparse catches exact tokens like `HTTPException` or a pasted error string. Docs corpora need both. |
| Reciprocal Rank Fusion | Cosine similarity and `ts_rank` aren't on comparable scales, so RRF just uses rank position with one constant (k=60) instead of trying to normalize two different score distributions. |
| Cross-encoder reranker, late in the pipeline | It's slow enough that it only makes sense once the candidate list is down to ~20, but the precision it buys there is worth the 50-200ms. |
| LangGraph router + 3 sub-agents | Different question types genuinely want different prompts and retrieval shaping. If the router gets it wrong, answer quality drops, but nothing actually breaks — every sub-agent can still answer any question. |
| RAGAS, deltas on a frozen eval set | These metrics are LLM-judged and noisy on their own, so what matters is the delta on a fixed set, especially confirmed against a held-out slice. |
| Gemini free tier, behind a provider abstraction | Fast, hosted, $0. Groq and local Ollama are both a one-line config change away, which mattered — I actually had to make that switch mid-project. |

---

## Evaluation results

These numbers are from the `dev` split (17 items). RAGAS metrics are judged by a local Ollama
`llama3.1:8b`; the deterministic ones don't need an LLM at all. Baseline config was
`max_tokens=450`, `min_tokens=64`, `overlap=60`, `dense_k=sparse_k=30`, `rerank_top_n=20`,
`final_k=6`, temperatures 0.0-0.3, and the original prompt.

```bash
python -m docqa.eval.run --split dev --label baseline-v1 --provider ollama
# tune config/corpora/fastapi.yaml + prompts.py, then:
python -m docqa.eval.run --split dev --provider ollama        # auto-diffs vs baseline-v1
python -m docqa.eval.run --split holdout --label holdout-final --provider ollama
```

| metric | baseline-v1 (dev) | tuned (dev) | Δ | held-out (8 Qs, never tuned against) |
|---|---|---|---|---|
| retrieval_recall@k (all) | 0.80 | 0.87 | +0.07 | 1.00 |
| &nbsp;&nbsp;— single-chunk Qs | 1.00 | 1.00 | — | 1.00 |
| &nbsp;&nbsp;— multi-chunk Qs | 0.57 | 0.71 | +0.14 | 1.00 |
| negative_refusal_rate | 1.00 | 1.00 | — | 1.00 |
| router_accuracy | 0.88 | 0.88 | — | 1.00 |
| answer_relevancy | 0.85 | 0.87–0.99* | +0.02 to +0.14 | 0.85 |
| faithfulness (excl. negatives) | 0.65 | 0.81–0.82* | +0.16 | 0.63 |
| context_precision (excl. negatives) | ~0.97 | 0.95–0.97* | flat (not a tuning target) | 0.94 |

\* Ranges across repeated dev runs, not real volatility in the system — see the judge-noise note below.

The held-out run is the one that matters most: recall, router accuracy, and refusal discipline all
held up or improved on questions I never looked at while tuning, and recall actually hit a clean
1.00 there (found every reference page, including both held-out multi-chunk questions). Faithfulness
on held-out (0.63) came in below the dev-tuned range, which the next section gets into.

### What I actually changed, in order

1. **Ingestion bug fix.** `fnmatch` doesn't support `**`, and its `*` crosses `/`, so
   `docs/en/docs/**/*.md` was silently skipping every top-level page — `async.md`, `python-types.md`,
   and others. I replaced it with a gitignore-style glob matcher. Source doc count went from 99 to
   107 after also excluding some community/meta pages the fix newly exposed.
2. **Keyword search, AND to OR.** `websearch_to_tsquery` required every query term to appear in a
   single chunk. I rewrote it to OR-match, with a bonus for chunks that do contain the full phrase,
   so a rare exact match still wins over a bunch of partial ones.
3. **Smaller, cleaner chunks.** Dropped `max_tokens` from 450 to 300. Sections now merge into their
   parent when their prose content (code stripped out) is under the minimum, which gets rid of stubs
   like a bare `## Recap` heading or a "Run the App" section that's mostly a shell command.
4. **Wider retrieval.** `dense_k`/`sparse_k` from 30 to 40, `rerank_top_n` from 20 to 30, `final_k`
   from 6 to 8.
5. **Prompt hardened, temperatures lowered.** Added an explicit instruction: if the sources describe
   a related feature but not the specific thing being asked, say so instead of answering about the
   related feature. This fixed a regression where wider retrieval had started handing the model
   plausible-but-wrong content on questions the docs don't actually cover.

### On the RAGAS judge noise

A local 8B model isn't fully reliable at the structured claim-decomposition that faithfulness and
context precision require — some runs score every item, others only manage 10 of 17. Two separate
post-tuning runs still agree on direction and rough size, and every deterministic metric (nothing
LLM-judged) is exactly reproducible. Context precision was never something I was trying to move here
and it stayed flat across every run, which is what you'd expect since the tuning targeted recall and
refusal behavior, not precision.

### What the baseline showed, and what I did about it

1. Multi-hop retrieval was weak — 1.00 recall on single-fact questions but only 0.57 on ones needing
   two or more pages. Traced back to the missing top-level docs and the AND-only keyword search,
   both fixed above. That's the 0.57 → 0.71 line.
2. Faithfulness was low (0.65 excluding negatives) even where retrieval worked fine — the model was
   elaborating past what the sources actually said. Fixed with the prompt change and lower
   temperature, which brought it to about 0.82.
3. A regression showed up partway through tuning: once retrieval got wider, a question with no real
   answer in the docs ("enable FastAPI's built-in user account database") got *harder* to refuse,
   because the model now had plausible-looking related content (the security tutorial) to work with.
   The same prompt fix that helped faithfulness also fixed this — both come down to sticking to what
   was actually asked.
4. RAGAS's faithfulness and context precision both score 0.0 on a correct refusal, since there's no
   supporting context for a claim that says "not covered." I've reported those separately
   (`*_excl_negatives`) since iteration 1 so a correct refusal doesn't drag the headline number down.

### One thing the held-out run surfaced that I haven't fully explained

Faithfulness splits by answer shape more than by category. Conceptual, prose-style answers scored
0.94-1.00 (type hints, app structure, settings). Code-heavy API-reference answers and step-by-step
troubleshooting answers scored 0.26-0.6 — file uploads, path-param validation, the nginx `root_path`
fix — even though retrieval found the right page every single time. I have two guesses, not
confirmed either way: the model might genuinely add things when it turns prose into steps or adapts
a code sample, or RAGAS's claim-based scoring might just not map well onto code blocks and
imperative instructions in the first place, since a line of code isn't really a "claim" the same way
a sentence is. If I kept working on this, the next thing I'd build is an eval item type that scores
"does the code match the docs' own example" separately from prose faithfulness, instead of scoring
both with the same metric.

Holdout run is saved at `evals/results/holdout-final.json`. Reproduce with `--split holdout --provider ollama`.

---

## Deployment

One small container runs Streamlit, the LangGraph pipeline, and the local models (bge-small,
bge-reranker-base). Postgres is already Neon — the same database ingestion wrote to — so nothing
about the data layer changes for deployment. Gemini/Groq calls just go out over the network. See
[docs/architecture.md §3.13](docs/architecture.md) for the full topology and the memory/rate-limit/
single-instance trade-offs that come with it.

On resources: bge-small (~130MB) plus bge-reranker-base (~1.1GB) plus the torch runtime need real
headroom. It fits comfortably on a 2GB instance and won't fit a typical smallest-tier free
allowance (256-512MB). Fly's `auto_stop_machines` scales the machine to zero when nobody's using it,
so in practice this stays close to $0 even on a paid-by-usage tier. If you want it to fit an
actually-free tier, swap `RERANKER_MODEL` to something smaller like
`cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB) — the Dockerfile bakes in whatever model is set at
build time, so it's a one-line change. Just know that the eval numbers above were measured with the
bigger reranker, so a smaller one won't match them exactly.

### Option A — Fly.io (what I used)

`flyctl deploy` ships your local directory straight to Fly's remote builder, so it doesn't need a
git push or a working local Docker daemon — which mattered here, since Docker Desktop was broken on
my machine the whole time.

```powershell
# install (open a new PowerShell window afterward — PATH only updates for new shells)
iwr https://fly.io/install.ps1 -useb | iex

fly auth login                                    # opens a browser
# rename the app in fly.toml first — `app` must be globally unique:
#   (edit fly.toml, or) fly apps create <your-unique-name>

# Load secrets straight from .env, not `source .env` or manual $VAR expansion.
# Neon's connection string contains `&channel_binding=require`, and a shell that
# evaluates the file reads that `&` as the background operator and silently drops
# the value. `fly secrets import` reads NAME=VALUE from stdin with no shell
# evaluation at all, which is the one safe way to do this:
grep -E '^(DATABASE_URL|GOOGLE_API_KEY|LLM_PROVIDER|CORPUS)=' .env | fly secrets import -a <your-app-name>

fly deploy
fly open                                          # launches the live URL
```

If `fly auth login` hits a browser OAuth redirect loop: try a private window, Edge instead of
Firefox, or turn off Enhanced Tracking Protection for `fly.io`.

### Option B — Railway (dashboard-only alternative)

No CLI needed. On railway.app: New Project → Deploy from GitHub repo (push the repo to GitHub
first), it picks up the `Dockerfile` and `railway.json` automatically, then set `DATABASE_URL`,
`GOOGLE_API_KEY`, `LLM_PROVIDER=gemini`, and `CORPUS=fastapi` in the Variables tab and deploy.
Railway injects `PORT` itself; the Dockerfile already reads it.

### After deploying

Visit the URL, ask something, and check that the Sources panel is actually showing retrieved chunks
with real scores. If it 500s on first load, check the platform logs — usually a missing or typo'd
env var, or the instance being sized under 2GB.
