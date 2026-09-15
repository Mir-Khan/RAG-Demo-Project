# Documentation Q&A Assistant — Multi-Agent RAG

This is a RAG system that answers questions over the FastAPI docs. I wanted to build it the way
you'd actually build one for a real product, not the tutorial version, so it's got hybrid
retrieval, a reranking step, a small multi-agent router, and an eval harness that actually tells me
whether any of this is working instead of me just guessing.

What's in here:

- Section-aware chunking, following the docs' own heading structure instead of just splitting every
  N characters.
- Hybrid retrieval — pgvector for dense search plus Postgres full-text for keyword search, fused
  with Reciprocal Rank Fusion, then a cross-encoder reranker before any of it reaches the LLM.
- Multi-agent orchestration with LangGraph. A router looks at the question and hands it to one of
  three specialized sub-agents (API reference, conceptual, or troubleshooting), all sharing the
  same retrieval tool.
- An eval harness built on RAGAS (faithfulness, answer relevancy, context precision) over a frozen
  set of questions, with the baseline-vs-tuned deltas committed under `evals/results/` so I'm not
  just claiming it got better.
- $0 to run — local `bge-small` embeddings and reranker, Gemini's free tier for generation (Groq or
  a local Ollama model both work too, it's a one-line env change), Postgres either in Docker or on
  a free managed tier.
- A second demo corpus for Pokémon games plus the current competitive format, mostly just to prove
  the ingestion side actually is corpus-agnostic and I didn't quietly build it to only work for
  FastAPI's docs. It pulls from Bulbapedia's live API and blends in some notes I wrote myself, and
  none of the chunking/retrieval/agent/eval code had to change for it. `CORPUS=pokemon` to switch.
  (This is an unofficial fan project — not affiliated with, endorsed by, or sponsored by Nintendo,
  Game Freak, Creatures Inc., or The Pokémon Company. Chat avatars come from
  `assets/avatars/pokemon/` if you drop images in — see the README in that folder for the
  fan-art-vs-official-sprite trade-off and what's actually in there right now.)

I wrote up the reasoning behind every real decision in [`docs/architecture.md`](docs/architecture.md)
— what I considered instead and why I didn't go with it. That doc is honestly the main thing here;
this README is more of a tour so you don't have to read all of that just to get the app running.

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

You'll need Python 3.13 — the ML stack doesn't have 3.14 wheels yet, I checked — and a Postgres 16 +
pgvector instance. Either `docker compose up -d --wait` for a local one, or point `DATABASE_URL` at
a free managed instance. I used Neon; just grab the direct, non-pooled connection string.

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

Tests don't need network or the ML stack — the chunker falls back to a heuristic token counter and
RRF is just arithmetic:

```bash
pytest -q
```

### Inspecting retrieval

Once something's ingested, you can watch what each retrieval stage actually does with a query,
which is honestly the most useful debugging tool in here:

```bash
python -m docqa.retrieval.debug "how do I return a 422 error" --corpus fastapi
python -m docqa.retrieval.debug "background tasks" --no-rerank      # pre-rerank order
```

It prints dense, sparse, fused, and reranked results side by side, which chunks each method found
that the other one missed, and how much the reranker moved things around.

### Asking a question (multi-agent)

```bash
copy .env.example .env
# set GOOGLE_API_KEY (default provider, free: https://aistudio.google.com/apikey)
# or LLM_PROVIDER=groq + GROQ_API_KEY, or LLM_PROVIDER=ollama for a local model
python -m docqa.agents.run "How do I add CORS middleware?" --corpus fastapi
```

Shows you the routing decision (category and confidence, or fallback), which sources it used, the
cited answer, and how long each step took.

### The web app

```bash
pip install -e ".[app]"
streamlit run src/docqa/app/main.py        # or: make app
```

It's a chat interface. Every answer has a caption with the routed category and per-node timings,
and a Sources panel underneath listing every retrieved chunk — its dense/sparse/RRF/rerank scores, a
link to the real doc page, and whether the answer actually cited it, included it without citing, or
dropped it once the context budget ran out. The embedder and reranker load once and get shared
across corpora, so switching between FastAPI and Pokémon mid-session doesn't reload a second copy of
either model.

---

## Design decisions, briefly

The real reasoning for each of these, plus what else I looked at and why I didn't go with it, is in
`docs/architecture.md`. Short version:

| Decision | Why |
|---|---|
| Section-aware chunking, 450/64/60 token budgets | Follows the doc author's own heading boundaries instead of cutting mid-thought. Stays under bge-small's 512-token limit once you leave room for the breadcrumb prefix. Overlap only kicks in when a section has to get split up. |
| `bge-small-en-v1.5` (384-dim), run locally | Free, fast enough on CPU, and the small vectors keep the index small too. The reranker makes up most of the quality you'd otherwise lose by not using a bigger embedding model. |
| pgvector, one `chunks` table | Vectors sit right next to the text and the `tsvector` column, so hybrid search is one SQL query instead of keeping two separate stores in sync. |
| HNSW index | Handles inserts as they come in and gives good recall at this size. IVFFlat would need rebuilding periodically as the data shifts, which felt like more moving parts than I wanted. |
| Dense + sparse hybrid | Dense is good at paraphrasing, sparse is good at catching exact tokens like `HTTPException` or someone pasting an actual error string. Docs really do need both. |
| Reciprocal Rank Fusion | Cosine similarity and `ts_rank` aren't even on the same scale, so RRF just uses where something ranked instead of trying to normalize two totally different score distributions against each other. |
| Cross-encoder reranker, late in the pipeline | It's slow enough that it only makes sense once the candidate list is already down to ~20 or so, but the precision it buys there is worth the 50-200ms it costs. |
| LangGraph router + 3 sub-agents | Different kinds of questions really do want different prompts and different retrieval shaping. If the router gets it wrong, the answer's just a bit worse, not broken — any of the sub-agents can still answer any question. |
| RAGAS, deltas on a frozen eval set | The metrics themselves are LLM-judged and kind of noisy on their own, so what actually matters is the delta on a fixed set, and whether that delta holds up on questions I never looked at while tuning. |
| Gemini free tier, behind a provider abstraction | Fast, hosted, $0. Groq and a local Ollama model are both one config line away — which mattered, because I actually had to make that switch partway through this project. |

---

## Evaluation results

These numbers are from the `dev` split (17 items). RAGAS metrics are judged by a local Ollama
`llama3.1:8b`; the deterministic ones don't touch an LLM at all. Baseline config was
`max_tokens=450`, `min_tokens=64`, `overlap=60`, `dense_k=sparse_k=30`, `rerank_top_n=20`,
`final_k=6`, temperatures 0.0-0.3, and the original prompt before I touched anything.

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

\* these are ranges across a couple of repeated dev runs, not the system actually being unstable —
see the judge-noise bit below.

The held-out run is the one I actually trust the most here. Recall, router accuracy, and refusal
behavior all held up or got better on questions I hadn't looked at while tuning, and recall hit a
clean 1.00 there — every reference page found, including both of the held-out multi-chunk
questions. Faithfulness on held-out (0.63) came in lower than the dev-tuned range though, which I
get into further down.

### What I actually changed, in order

1. **Ingestion bug I didn't expect.** `fnmatch` doesn't understand `**`, and its `*` crosses `/`, so
   `docs/en/docs/**/*.md` was quietly skipping every top-level page — `async.md`, `python-types.md`,
   pages like that. I wrote a small gitignore-style glob matcher to replace it. Source doc count
   went from 99 to 107 after that fix, and after also excluding a few community/meta pages the fix
   ended up exposing.
2. **Keyword search, AND to OR.** `websearch_to_tsquery` wanted every query term to show up in the
   same chunk, which almost never happens for a real question. I rewrote it to OR-match, with a
   bonus for chunks that do contain the whole phrase, so an exact match still wins over a pile of
   partial ones.
3. **Smaller, cleaner chunks.** Dropped `max_tokens` from 450 to 300. Sections now merge into their
   parent when their actual prose (code stripped out) is under the minimum, which gets rid of stub
   sections — a bare `## Recap` heading, or a "Run the App" section that's basically just a shell
   command.
4. **Wider retrieval.** `dense_k`/`sparse_k` went from 30 to 40, `rerank_top_n` from 20 to 30,
   `final_k` from 6 to 8.
5. **Prompt hardened, temperatures lowered.** Added an explicit line telling it: if the sources
   describe something related but not the specific thing being asked, say that instead of answering
   about the related thing. This fixed a regression where wider retrieval had started handing the
   model plausible-but-wrong content on questions the docs genuinely don't cover.

### About the RAGAS judge noise

An 8B model running locally just isn't that reliable at the structured claim-decomposition that
faithfulness and context precision need — some runs it scores every single item, other runs it only
manages about 10 of 17. Two separate runs after tuning still agree on direction and roughly how much
things moved, and every deterministic metric (nothing LLM-judged in it) reproduces exactly every
time. Context precision was never something I was trying to move in the first place, and it stayed
flat across every run, which is what I'd expect — the tuning was aimed at recall and refusal
behavior, not precision.

### What the baseline told me, and what I did about it

1. Multi-hop retrieval was weak — 1.00 recall on single-fact questions, but only 0.57 once a
   question needed two or more pages. Traced it back to the missing top-level docs and the AND-only
   keyword search, both fixed above. That's the 0.57 to 0.71 move.
2. Faithfulness was low (0.65 excluding negatives) even where retrieval was working fine — the model
   was just adding stuff past what the sources actually said. The prompt change plus a lower
   temperature brought that up to around 0.82.
3. A regression showed up partway through tuning that I didn't see coming: once retrieval got wider,
   a question with no real answer in the docs ("enable FastAPI's built-in user account database")
   got *harder* to refuse, because the model suddenly had plausible-looking related content (the
   security tutorial) to work with instead of nothing. The same prompt fix that helped faithfulness
   fixed this too — they're really the same problem, sticking to what was actually asked.
4. RAGAS scores both faithfulness and context precision as 0.0 on a correct refusal, since there's
   no context to support a claim like "not covered." I've been reporting those separately
   (`*_excl_negatives`) since the first tuning pass so a correct refusal doesn't drag the headline
   number down for doing the right thing.

### One thing from the held-out run I still can't fully explain

Faithfulness seems to split by what shape the answer is, more than by category. Conceptual, prose
answers scored 0.94-1.00 — type hints, app structure, settings, stuff like that. Code-heavy
api_reference answers and step-by-step troubleshooting answers scored 0.26-0.6 — file uploads,
path-param validation, the nginx `root_path` fix — even though retrieval found the correct page
every single time on those. I've got two guesses and haven't confirmed either: maybe the model
genuinely adds things when it's turning prose into steps or adapting a code sample, or maybe RAGAS's
claim-based scoring just doesn't map well onto code blocks and imperative instructions, since a line
of code isn't really a "claim" the way a sentence is. If I kept going on this, the next thing I'd
build is an eval item type that scores whether the code matches the docs' own example separately
from prose faithfulness, instead of running both through the same metric.

Holdout run is saved at `evals/results/holdout-final.json`. To reproduce: `--split holdout --provider ollama`.

---

## Deployment

One small container runs Streamlit, the LangGraph pipeline, and the local models (bge-small,
bge-reranker-base). Postgres is already Neon — the same database ingestion wrote to — so nothing
about the data layer changes just because it's deployed. Gemini/Groq calls just go out over the
network like normal. See [docs/architecture.md §3.13](docs/architecture.md) for the full topology
and the memory/rate-limit/single-instance stuff that comes with it.

On resources: bge-small (~130MB) plus bge-reranker-base (~1.1GB) plus the torch runtime actually
need some real headroom. It fits fine on a 2GB instance, but it won't fit a typical smallest-tier
free allowance (256-512MB). Fly's `auto_stop_machines` scales the machine down to zero when nobody's
using it, so in practice this stays close to $0 even on a paid-by-usage tier. If you want it to fit
an actually-free tier instead, swap `RERANKER_MODEL` to something smaller like
`cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB) — the Dockerfile bakes in whatever model is set at
build time, so it's a one-line change. Just know the eval numbers above were measured with the
bigger reranker, so a smaller one won't get you the exact same results.

### Option A — Fly.io (what I actually used)

`flyctl deploy` ships your local folder straight to Fly's remote builder, so it doesn't need a git
push or a working local Docker daemon, which mattered a lot here since Docker Desktop was broken on
my machine for basically this whole project.

```powershell
# install it (open a new PowerShell window after — PATH only updates for new shells)
iwr https://fly.io/install.ps1 -useb | iex

fly auth login                                    # opens a browser
# rename the app in fly.toml first — `app` has to be globally unique:
#   (edit fly.toml, or) fly apps create <your-unique-name>

# Load secrets straight from .env, not `source .env` or manually expanding $VARs.
# Neon's connection string has `&channel_binding=require` in it, and a shell that
# evaluates the file reads that `&` as the background operator and just silently
# drops the value — found that one the hard way. `fly secrets import` reads
# NAME=VALUE from stdin with no shell evaluation at all, so it's the one safe way:
grep -E '^(DATABASE_URL|GOOGLE_API_KEY|LLM_PROVIDER|CORPUS)=' .env | fly secrets import -a <your-app-name>

fly deploy
fly open                                          # opens the live URL
```

If `fly auth login` gets stuck in a browser OAuth redirect loop: try a private window, Edge instead
of Firefox, or turn off Enhanced Tracking Protection for `fly.io`.

### Option B — Railway (dashboard-only alternative)

No CLI needed for this one. On railway.app: New Project → Deploy from GitHub repo (push it to
GitHub first), it picks up the `Dockerfile` and `railway.json` on its own, then set
`DATABASE_URL`, `GOOGLE_API_KEY`, `LLM_PROVIDER=gemini`, and `CORPUS=fastapi` in the Variables tab
and deploy. Railway injects `PORT` itself, and the Dockerfile already reads it.

### After deploying

Open the URL, ask it something, and check the Sources panel is actually showing retrieved chunks
with real scores. If it 500s on first load, check the platform logs first — it's usually a missing
or typo'd env var, or the instance being sized under 2GB.
