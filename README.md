# Documentation Q&A Assistant — Multi-Agent RAG

A production-shaped RAG system that answers questions over the **FastAPI documentation**, built to
demonstrate senior-level retrieval architecture rather than a tutorial pipeline.

- **Section-aware chunking** driven by the docs' own heading hierarchy — not fixed-size splitting.
- **Hybrid retrieval**: pgvector dense search + Postgres full-text (BM25-style), fused with Reciprocal
  Rank Fusion, then a **cross-encoder reranker** before the LLM sees anything.
- **Multi-agent orchestration** (LangGraph): a router classifies the query and dispatches to a
  specialized sub-agent (API reference / conceptual / troubleshooting) over a shared retrieval tool.
- **Evaluation harness** (RAGAS): faithfulness, answer relevancy, context precision on a frozen eval
  set, with baseline-vs-tuned deltas committed under `evals/results/`.
- **$0 to run**: local `bge-small` embeddings + reranker, Gemini free-tier LLM (Groq / local Ollama
  swappable via one env var), Postgres in Docker or a managed free tier.
- **Corpus-agnostic ingestion**: FastAPI is the evaluated corpus; `examples/pokemon/` shows the same
  pipeline over a hand-authored knowledge base via one config file.

> **Every architectural decision — with the alternatives considered and the trade-off accepted — is
> written up in [`docs/architecture.md`](docs/architecture.md).** That doc is the main artifact.

---

## Status

| Stage | State |
|---|---|
| 1. Ingestion pipeline (loaders, section-aware chunker, embeddings, pgvector store) | ✅ built |
| 2. Hybrid retrieval — dense + sparse + RRF + cross-encoder rerank | ✅ built |
| 3. LangGraph multi-agent orchestration — router + 3 sub-agents + fallback | ✅ built |
| 4. Evaluation harness — RAGAS + deterministic metrics, frozen 25-item set, baseline/diff | ✅ built |
| 5. Streamlit frontend — cited answers + per-stage retrieval transparency | ✅ built |
| 6. Deployment (Fly.io / Railway) | ✅ built |

---

## Repo layout

```
config/corpora/*.yaml     one file per knowledge base — loader, chunk budgets, router categories
src/docqa/
  config.py               env settings + corpus-config loading
  ingestion/
    loaders/               SourceLoader protocol + github_markdown / markdown_dir + registry
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
    ragas_metrics.py       the ONLY file importing ragas — faithfulness / relevancy / precision
    run.py                 python -m docqa.eval.run --split dev --label baseline-v1
    diff.py                python -m docqa.eval.diff base.json current.json
  app/
    components.py          pure render helpers (score chips, citation status) — unit-tested
    main.py                streamlit run src/docqa/app/main.py
evals/
  eval_set.jsonl           the frozen 25-question set (committed)
  README.md                composition, leakage guard, metric definitions
  results/                 baseline-v1.json + holdout-final.json committed; other runs gitignored
docs/architecture.md      decision rationale — the thing to read
examples/pokemon/         demo corpus for the pluggable loader
```

---

## Quickstart (ingestion)

Requires **Python 3.13** (the ML stack has no 3.14 wheels yet) and a **Postgres 16 + pgvector**.
For the DB, either `docker compose up -d --wait` (uses `docker-compose.yml`), or point
`DATABASE_URL` at a free managed instance with pgvector — [Neon](https://neon.tech) works with no
extra setup (use the *direct*, non-pooled connection string).

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

Run the tests (no network / no ML stack needed — chunker uses a heuristic token counter, RRF is pure):

```bash
pytest -q
```

### Inspecting retrieval

Once a corpus is ingested, see what each retrieval stage does to a query:

```bash
python -m docqa.retrieval.debug "how do I return a 422 error" --corpus fastapi
python -m docqa.retrieval.debug "background tasks" --no-rerank      # pre-rerank order
```

Prints dense / sparse / fused / reranked as separate tables, plus which chunks each
method found that the other missed, and how far the reranker moved things.

### Asking a question (multi-agent)

```bash
copy .env.example .env
# set GOOGLE_API_KEY (default provider, free: https://aistudio.google.com/apikey)
# or LLM_PROVIDER=groq + GROQ_API_KEY, or LLM_PROVIDER=ollama for a local model
python -m docqa.agents.run "How do I add CORS middleware?" --corpus fastapi
```

Shows the routing decision (category + confidence, or fallback), the sources used,
the cited answer, and per-node timings.

### The web app

```bash
pip install -e ".[app]"
streamlit run src/docqa/app/main.py        # or: make app
```

Chat UI. Each answer carries a caption (routed category + confidence, per-node timings) and a
**Sources** panel: every retrieved chunk with its dense / sparse / RRF / rerank score, a link to
the real doc page, and whether the answer cited it (`[n] cited`), included it without citing, or
dropped it at the context-token budget. One `QAPipeline` is cached per corpus.

---

## Design decisions in one line each

| Decision | Why (full rationale in `docs/architecture.md`) |
|---|---|
| Section-aware chunking, 450/64/60 token budgets | Respect the author's heading boundaries; stay under bge-small's 512 limit with breadcrumb headroom; overlap only on forced splits. |
| `bge-small-en-v1.5` (384-dim), local | $0, CPU-friendly, small vectors → small index; reranker recovers the quality gap vs bigger models. |
| pgvector, single `chunks` table | Vectors next to text + `tsvector` → hybrid search is one transactional SQL query; no separate vector DB to sync. |
| HNSW index | Incremental inserts, good recall/latency at this scale; IVFFlat would need rebuilds as data drifts. |
| Dense + sparse hybrid | Dense handles paraphrase; sparse nails `HTTPException`, `status_code=422`, pasted error strings. Docs need both. |
| Reciprocal Rank Fusion | Cosine and `ts_rank` aren't on comparable scales; RRF uses rank only, one constant (`k=60`). |
| Cross-encoder reranker, late + short list | Precision worth ~50–200 ms once the field is down to ~20 candidates. |
| LangGraph router + 3 sub-agents | Query types want genuinely different prompts/retrieval shaping; misrouting degrades quality, not correctness. |
| RAGAS, deltas on a frozen set | LLM-judged metrics are noisy; a held-out slice proves tuning generalized. |
| Gemini free tier, provider-abstracted | Hosted, fast, $0; Groq / local Ollama are a one-env-var swap (the abstraction already earned its keep once). |

---

## Evaluation results

`dev` split (17 items). RAGAS metrics judged by a local Ollama `llama3.1:8b`; deterministic metrics
need no LLM. Baseline config: `max_tokens=450`, `min_tokens=64`, `overlap=60`, `dense_k=sparse_k=30`,
`rerank_top_n=20`, `final_k=6`, temperatures 0.0–0.3, baseline prompt.

```bash
python -m docqa.eval.run --split dev --label baseline-v1 --provider ollama
# tune config/corpora/fastapi.yaml + prompts.py, then:
python -m docqa.eval.run --split dev --provider ollama        # auto-diffs vs baseline-v1
python -m docqa.eval.run --split holdout --label holdout-final --provider ollama
```

| metric | baseline-v1 (dev) | tuned (dev) | Δ | **held-out** (8 Qs, never tuned against) |
|---|---|---|---|---|
| retrieval_recall@k (all) | 0.80 | **0.87** | **+0.07** | **1.00** |
| &nbsp;&nbsp;— single-chunk Qs | 1.00 | 1.00 | — | 1.00 |
| &nbsp;&nbsp;— multi-chunk Qs | **0.57** | **0.71** | **+0.14** | **1.00** |
| negative_refusal_rate | 1.00 | 1.00 | — | 1.00 |
| router_accuracy | 0.88 | 0.88 | — | **1.00** |
| answer_relevancy | 0.85 | 0.87–0.99* | +0.02 to +0.14 | 0.85 |
| faithfulness (excl. negatives) | 0.65 | 0.81–0.82* | **+0.16** | 0.63 |
| context_precision (excl. negatives) | ~0.97 | 0.95–0.97* | flat (not a tuning target) | 0.94 |

\* *Ranges across repeated dev runs — see the judge-noise note below.*

**Held-out confirms the tuning generalized**, on every metric it targeted: retrieval recall, router
accuracy, and refusal discipline all held or improved on questions never seen while tuning — recall
even hit a clean 1.00 (8/8 reference pages found, including both held-out multi-chunk questions).
Faithfulness on held-out (0.63) sits *below* the dev-tuned range — see the next point.

**Tuning changes, in order:**
1. **Ingestion bug fix**: `fnmatch` doesn't support `**` and its `*` crosses `/`, so
   `docs/en/docs/**/*.md` silently skipped every top-level page (`async.md`, `python-types.md`, …).
   Replaced with a gitignore-style glob matcher. 99→107 source docs after also excluding
   community/meta pages the fix newly exposed.
2. **Keyword search AND→OR**: `websearch_to_tsquery` required *every* query term in one chunk;
   rewritten to OR-match with an AND-match bonus (rare full-phrase hits still win).
3. **Smaller, cleaner chunks**: `max_tokens` 450→300; sections merge into their parent when their
   *non-code prose* (not raw length) is under `min_tokens` — kills `## Recap` / "Run the App" stubs
   that are mostly a shell snippet and embed poorly.
4. **Wider retrieval**: `dense_k`/`sparse_k` 30→40, `rerank_top_n` 20→30, `final_k` 6→8.
5. **Generation prompt hardened** + temperatures lowered: explicit "if the sources cover a *related*
   feature but not the specific thing asked, say so — don't substitute" (fixed a regression where
   wider retrieval handed the model plausible-but-wrong related content on out-of-scope questions).

**On the RAGAS judge noise** (`answer_relevancy`/`faithfulness`/`context_precision` ranges above):
a local 8B model is unreliable at the structured claim-decomposition these metrics require —
`faithfulness`/`context_precision` come back fully scored on some runs, partially on others (as few
as 10/17). Two independent post-tuning runs still agree on direction and rough magnitude, and every
*deterministic* metric (no LLM judge involved) is reproducible exactly. `context_precision` was
never a tuning target here and stayed flat across all runs, as expected — the tuning targeted
recall and refusal discipline, not precision.

**What the baseline exposed, and how each was addressed:**
1. **Multi-hop retrieval was weak** (single-chunk 1.00 vs multi-chunk 0.57) — traced to the missing
   top-level docs and AND-only keyword search; both fixed above. → 0.57 → 0.71.
2. **Faithfulness was low (0.65 excl. negatives)** — the generator elaborated beyond the sources even
   when retrieval succeeded. → prompt hardening + lower temperature. → 0.65 → ~0.82.
3. **A negative regression surfaced mid-tuning**: wider retrieval made an out-of-scope question
   ("enable FastAPI's built-in user account database") *harder* to refuse, because it now surfaced
   plausible-looking related content (the security tutorial). Fixed by the same prompt hardening
   that fixed faithfulness — both are "stick to what's actually asked" failures.
4. **RAGAS faithfulness/context_precision penalise correct refusals** (0.0 for a question with no
   relevant content). Reported separately (`*_excl_negatives`) from iteration 1 onward.

**New finding from the held-out run — faithfulness splits by answer shape, not just by category.**
Per-question: conceptual/prose answers scored 0.94–1.00 (Python type hints, app structure, settings);
code-heavy `api_reference` answers and numbered-step `troubleshooting` answers scored 0.26–0.6 (file
upload, path-param validation, the nginx `root_path` fix) despite retrieving the correct page every
time (recall 1.00 across the board). Two plausible causes, not yet disambiguated: the model
genuinely infers beyond the source when turning prose into "steps" or adapting a code sample, **or**
RAGAS's claim-decomposition doesn't map cleanly onto code blocks and imperative instructions in the
first place (a code line isn't a "claim" in the way a declarative sentence is). Next step: an eval
item type that separates "does the code compile from the docs' own example" from "is the prose
faithful," rather than scoring both with the same claim-based metric.

Holdout run: `evals/results/holdout-final.json`. To reproduce: `--split holdout --provider ollama`.

---

## Deployment

One small always-on container runs Streamlit + the LangGraph pipeline + the local
models (bge-small, bge-reranker-base). Postgres is already Neon (managed) — the
same database ingestion already wrote to, so **nothing changes about the data
layer for deployment**. Groq/Gemini are outbound API calls. See
[docs/architecture.md §3.13](docs/architecture.md) for the topology diagram and
the memory/rate-limit/single-instance bottlenecks this implies.

**Resource note, honestly stated:** bge-small (~130 MB) + bge-reranker-base
(~1.1 GB) + the torch runtime need real headroom — comfortably fits a **2 GB**
instance, will not fit a typical smallest-tier (256–512 MB) free allowance. Fly's
`auto_stop_machines` scales the VM to zero when idle, so cost is close to $0 in
practice even on a paid-by-usage tier. Swapping `RERANKER_MODEL` to a smaller
cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80 MB) fits a
genuinely free tier, at the cost of the reranker quality this project's eval
numbers were measured with — the Dockerfile bakes in whatever `RERANKER_MODEL`
is set to at build time, so this is a one-line change if you want it.

### Option A — Fly.io (recommended here)

Chosen because `flyctl deploy` ships the local directory straight to Fly's
**remote builder** — no git push and no working local Docker daemon required,
which matters on a machine where Docker Desktop is broken.

```powershell
# install (new PowerShell window afterwards — PATH only updates for new shells)
iwr https://fly.io/install.ps1 -useb | iex

fly auth login                                    # opens a browser
# rename the app in fly.toml first — `app` must be globally unique:
#   (edit fly.toml, or) fly apps create <your-unique-name>

# Load secrets straight from .env — NOT `source .env` / manual $VAR expansion.
# Neon's connection string contains `&channel_binding=require`; a shell that
# evaluates the file (source, `.`) reads that `&` as the background operator and
# silently drops the value. `fly secrets import` reads NAME=VALUE from stdin with
# no shell evaluation at all, so this is the one safe way to set it:
grep -E '^(DATABASE_URL|GOOGLE_API_KEY|LLM_PROVIDER|CORPUS)=' .env | fly secrets import -a <your-app-name>

fly deploy
fly open                                          # launches the live URL
```

If `fly auth login` hits the same browser OAuth redirect-loop issue Groq did
earlier in this build: same fix — try a private window, or Edge, or disable
Enhanced Tracking Protection for `fly.io`.

### Option B — Railway (dashboard-driven alternative)

No CLI needed: railway.app → **New Project → Deploy from GitHub repo** (pushes
the repo to GitHub first) → it auto-detects the `Dockerfile` and `railway.json`
→ **Variables** tab: set `DATABASE_URL`, `GOOGLE_API_KEY`, `LLM_PROVIDER=gemini`,
`CORPUS=fastapi` → deploy. Railway injects `PORT` itself; the Dockerfile already
reads it.

### After deploying

Visit the URL, ask a question, confirm the Sources panel shows retrieved chunks
with real scores. If it 500s on first load, check the platform's logs — the most
likely causes are a missing/typo'd env var or the instance being sized below 2 GB.
