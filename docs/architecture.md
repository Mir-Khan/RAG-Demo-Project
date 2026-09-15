# Architecture Rationale — Documentation Q&A Assistant with Multi-Agent RAG

> Purpose of this document: for every non-trivial architectural decision in this project, record
> **what was chosen**, **what else was on the table**, and **the trade-off that was accepted**.
> It is written so the decisions can be defended in a technical interview, not just described.

---

## 1. System Overview

This is a production-shaped (not tutorial-shaped) Retrieval-Augmented Generation system that answers
questions over the FastAPI documentation. Documentation is ingested from a **pinned GitHub tarball**
(`fastapi/fastapi` tag `0.115.6`, no `git` binary dependency) through a pluggable `SourceLoader`
protocol, parsed into a heading stack, and split into **section-aware, token-budgeted chunks**
(≤450 tokens, ≥64, 60-token overlap only on section overflow) with a breadcrumb prefix. Chunks are
embedded locally with **BAAI/bge-small-en-v1.5** (384-dim, $0) and stored in a single `chunks` table
in **Postgres 16 + pgvector**, which holds both a `vector(384)` column (HNSW, cosine) and a generated
`tsvector` column (GIN) so one store serves dense and sparse retrieval. A query is answered by a
**LangGraph** graph: a router classifies it (`api_reference` / `conceptual` / `troubleshooting`,
defined per-corpus in config) and dispatches to one of three specialized sub-agents that share a
**hybrid retrieval tool** — pgvector cosine + Postgres full-text, fused with **Reciprocal Rank
Fusion**, then reranked with a **bge-reranker-base** cross-encoder before context assembly. The
generator is **Groq `llama-3.3-70b-versatile`** behind a provider abstraction (Ollama swappable
locally). Quality is tracked with **RAGAS** over a curated 20–30 item eval set, with before/after
deltas committed as JSON under `evals/results/`. A minimal **Streamlit** UI shows the answer
alongside retrieved sources and their scores. Everything corpus-specific lives in
`config/corpora/<name>.yaml`, so a hand-authored Pokémon knowledge base can be swapped in without
touching pipeline, retrieval, agent, or eval code.

---

## 2. ASCII Data-Flow Diagram

```
INGESTION PATH (offline, run via `docqa-ingest`)
------------------------------------------------
config/corpora/fastapi.yaml
        │  (loader type, repo, ref, globs, chunk budgets, router categories)
        ▼
SourceLoader (github_markdown | markdown_dir)
  download pinned tarball ──► extract ──► filter include/exclude globs
        ▼
markdown parse ──► heading stack ──► sections (H1 > H2 > H3 …)
        ▼
Chunker: pack sections into ≤450-tok chunks
  • merge sections < 64 tok into neighbour
  • split section > 450 tok with 60-tok overlap
  • prepend breadcrumb  "FastAPI > Tutorial > Query Parameters\n\n"
        ▼
bge-small-en-v1.5 (local, batched)  ──► 384-dim vectors
        ▼
Postgres:  INSERT INTO chunks (corpus, url, section_path, text, embedding, tsv←generated)
           HNSW(cosine) on embedding   +   GIN on tsv


QUERY PATH (online)
-------------------
Streamlit  ──► question
        ▼
LangGraph: Router agent (LLM classify → api_reference | conceptual | troubleshooting)
        │                                   │ low confidence / invalid label
        ▼                                   ▼
Sub-agent (per category)               fallback: generic retrieval, no category shaping
        │  shared state {query, category, chunks, draft, citations}
        ▼
Retrieval tool
  ├─ dense:  embedding(query w/ instruction prefix) → pgvector  <=>  top 30
  ├─ sparse: websearch_to_tsquery(query)           → ts_rank    top 30
  ├─ fuse:   Reciprocal Rank Fusion (k=60)         → top 20
  └─ rerank: bge-reranker-base cross-encoder(query, chunk) → top 5–8
        ▼
Context assembly (breadcrumb + text + source URL, token-capped)
        ▼
Groq llama-3.3-70b-versatile  (provider-abstracted)  ──► draft answer + citations
        ▼
Streamlit renders: answer  +  sources table (url, dense score, sparse rank, rerank score)
```

---

## 3. Decision Rationales

### 3.1 Section-aware chunking over fixed-size; the specific token budgets

**Decision.** Parse markdown into a heading stack, treat each leaf section as the base unit, then pack
sections into chunks with `max_tokens=450`, `min_tokens=64`, `overlap_tokens=60` (overlap applied
only when a single section exceeds the budget), and prepend a `FastAPI > Tutorial > Query Parameters`
breadcrumb to every chunk.

**Alternatives considered.**
- *Fixed-size sliding window* (e.g. 512 tokens, 15% overlap) — trivial to implement, library default.
- *Recursive character splitting* (LangChain `RecursiveCharacterTextSplitter`) — splits on `\n\n`, then
  `\n`, then space; structure-aware only by accident.
- *Semantic/embedding-based chunking* — split where adjacent sentence embeddings diverge.

**Trade-off.** Fixed-size windows routinely cut through the middle of a code block or between a claim
and its qualifying sentence, which produces chunks that embed to a blurred centroid and retrieve
poorly; they also duplicate 15% of the corpus as overlap regardless of need. Section-aware chunking
respects the author's own semantic boundaries (a `##` heading is a human-curated topic split) and
keeps fenced code blocks intact. The cost is real: it needs a markdown parser, a heading-stack state
machine, and merge/split logic for the long tail of sections that are too small or too big — roughly
150 lines of code and a class of edge cases (tables, nested lists, front-matter) that fixed-size never
hits. Semantic chunking was rejected as more expensive at ingest and non-deterministic — hard to
reproduce and debug for marginal gain over heading splits on well-structured docs.

**Why these numbers.** bge-small has a hard 512-token context; 450 leaves ~60 tokens of headroom for
the breadcrumb so it is never truncated. `min_tokens=64` prevents "orphan" chunks (a lone heading, a
one-line note) that dilute the index and waste a retrieval slot. `overlap_tokens=60` (~13% of 450) is
the standard "one or two sentences of context" bridge, applied *only* on forced splits so the index
does not carry redundant near-duplicates for the common case where a section already fits.

**At 100x scale.** Add a second, larger "parent" granularity (small-to-big / parent-document
retrieval): embed 200–300 token children for precision, but feed the enclosing section or page to the
LLM for context. Also move chunking parameters into an A/B-tunable sweep driven by the eval harness
rather than hand-set constants.

### 3.2 bge-small-en-v1.5 over bge-base / nomic-embed / OpenAI embeddings

**Decision.** `BAAI/bge-small-en-v1.5`, 384-dim, run locally via `sentence-transformers`, query-side
instruction prefix only (`Represent this sentence for searching relevant passages:`).

**Alternatives considered.**
- *bge-base-en-v1.5* (768-dim) — same family, ~2–3 MTEB points higher, 2x vector size, ~3x CPU latency.
- *nomic-embed-text-v1.5* (768-dim, 8192-token context, Matryoshka) — long-context, strong retrieval.
- *OpenAI `text-embedding-3-small`* (1536-dim) — no local compute, strong quality, ~$0.02 / 1M tokens.

**Trade-off.** The dominating constraints here are **$0 cost** and **runs on a low-tier CPU box**.
bge-small is ~130 MB, embeds a few hundred chunks/sec on CPU, and its 384-dim vectors keep the HNSW
index small (memory is the pgvector bottleneck on a small instance). bge-base buys a few retrieval
points for double the storage and triple the latency — not worth it when a cross-encoder reranker
(§3.7) recovers most of that gap more cheaply. nomic's 8192-token context is wasted because chunks are
capped at 450 tokens by design. OpenAI embeddings reintroduce a paid API dependency, network latency
on the hot path, and vendor lock-in for the one component whose output (vectors) is expensive to
re-generate if the corpus is large. The accepted downside: bge-small is English-only and mid-pack on
MTEB, so hard paraphrase/synonym matches lean on the sparse channel and the reranker.

**At 100x scale.** Re-evaluate on a current MTEB retrieval slice; likely move to a 768-dim model
(bge-base, `gte-base`, or a fine-tuned bge-small on in-domain query/chunk pairs mined from logs) and
run embedding on a GPU batch worker. Consider Matryoshka truncation to keep index size flat.

### 3.3 pgvector over a dedicated vector DB (Qdrant / Weaviate / Pinecone / Chroma)

**Decision.** Postgres 16 + `pgvector` (`pgvector/pgvector:pg16`), raw SQL via `psycopg3`, one `chunks`
table.

**Alternatives considered.**
- *Qdrant / Weaviate* — purpose-built, fast HNSW, native payload filtering, built-in hybrid (Weaviate).
- *Pinecone* — fully managed, zero ops, autoscaling.
- *Chroma* — trivial local dev, embedded, popular in tutorials.

**Trade-off.** A dedicated vector DB is a *second* stateful system to run, back up, monitor, and keep
in sync with the source-of-truth metadata. For a corpus of this size (FastAPI docs ≈ low thousands of
chunks; even 100k is trivial), pgvector's HNSW is more than fast enough (single-digit ms), and putting
vectors *next to* the text, URL, breadcrumb, and `tsvector` in one row means hybrid retrieval, corpus
scoping (`WHERE corpus = $1`), and metadata filtering are one SQL query with real transactions — no
dual-write consistency problem. It also demonstrates SQL/DB engineering rather than
API-call-through-an-SDK. Pinecone was rejected for cost and lock-in on a $0 project; Chroma for being
under-featured for hybrid and not something you'd defend as "production-grade". The accepted cost:
pgvector's index build and recall tuning are less turnkey than Qdrant's, there's no native distributed
sharding, and very large filtered searches need care (partial indexes / partitioning).

**At 100x scale (10M+ chunks, high QPS).** Partition `chunks` by `corpus`, move to `halfvec` (2-byte
floats) to halve index memory, and if recall/latency at that size regresses, migrate the vector path
to Qdrant while keeping Postgres as the metadata + full-text store.

### 3.4 HNSW vs IVFFlat for the vector index

**Decision.** HNSW with `vector_cosine_ops`.

**Alternatives considered.**
- *IVFFlat* — inverted lists over k-means centroids; smaller index, faster build, must pick `lists`.
- *No ANN index* (exact brute force) — perfect recall, `O(n)` scan.

**Trade-off.** IVFFlat needs to be built *after* representative data is loaded (its centroids are
learned from the current rows), and its recall degrades as data drifts from those centroids, so it
wants periodic rebuilds — awkward for an incrementally-upserted corpus. It also exposes two knobs
(`lists` at build, `probes` at query) that interact. HNSW builds incrementally, handles inserts
gracefully, and gives better recall-at-latency for small-to-medium datasets, with just `ef_search` to
tune at query time. The costs accepted: HNSW index build is slower and the index is larger and more
memory-hungry (the graph must be resident for good performance) — a genuine concern on a low-tier box,
but manageable at this corpus size. At very large scale IVFFlat's smaller footprint and faster builds
start to win, which is the main reason to revisit. Exact search is kept as the correctness reference
in tests.

**At 100x scale.** Benchmark HNSW (`m`, `ef_construction`) memory vs IVFFlat/IVFPQ recall on the real
distribution; product quantization becomes attractive once the raw vectors no longer fit in RAM.

### 3.5 Hybrid retrieval; dense vs sparse strengths and blind spots

**Decision.** Run dense (pgvector cosine) and sparse (Postgres full-text `ts_rank`, BM25-style) in
parallel, fuse with RRF, then rerank.

**Alternatives considered.** Dense-only (the common RAG default); sparse-only (classic search); dense
with a metadata/keyword pre-filter instead of a true second ranking.

**What each catches that the other misses.**
- *Dense* captures paraphrase and concept match: "how do I make an endpoint non-blocking" → the
  `async def` path-operation section, with no shared words. Robust to vocabulary mismatch.
- *Dense misses*: exact identifiers and rare tokens. `HTTPException`, `status_code=422`,
  `response_model_exclude_unset` — a small embedding model smears these toward a generic centroid, and
  the exact-token signal that a user typing an error message needs is gone.
- *Sparse* nails exact symbols, error strings, parameter names, and quoted phrases, and is fully
  explainable ("matched these 3 lexemes"). Zero cold-start, no model to run.
- *Sparse misses*: synonymy and phrasing ("background job" vs "background task"), and any question
  that shares no surface tokens with the answer.

**Trade-off.** Hybrid roughly doubles retrieval work (two queries + a fusion step) and adds tuning
surface. For a *documentation* corpus — dense with jargon, symbol names, and copy-pasted error text —
the sparse channel demonstrably rescues queries dense-only drops, so the cost is justified. For a
purely narrative corpus the marginal value of sparse is lower.

**At 100x scale.** Replace Postgres FTS with a real BM25 (`ParadeDB`/`pg_search`, or Elasticsearch/
OpenSearch) and add a learned sparse model (SPLADE) as a third channel; keep RRF as the fuser.

### 3.6 Reciprocal Rank Fusion over weighted score normalization

**Decision.** Fuse the two ranked lists with RRF: `score(d) = Σ 1 / (k + rank_i(d))`, `k = 60`.

**Alternatives considered.** Min-max or z-score normalize each score list then take a weighted sum;
`CombSUM`/`CombMNZ`; learning-to-rank a fusion model.

**Trade-off.** Cosine similarity (bounded ~0–1, often compressed into a narrow high band) and
`ts_rank` (unbounded, corpus- and length-dependent) are **not on comparable scales**, and their
distributions shift per query. Any weighted-sum scheme forces you to pick a normalization *and* a
weight, both brittle across queries and needing re-tuning whenever the model or corpus changes. RRF
ignores magnitudes entirely and uses only rank position, so it is scale-free, parameter-light (just
`k`, and `k=60` is a well-established default that damps low-rank items), and empirically competitive
with tuned weighted fusion. The accepted downside: RRF throws away genuine score information — a
landslide #1 in both lists is treated like a marginal #1 — which is exactly what the cross-encoder
reranker downstream is there to fix. With a labelled training set a learned fuser could beat RRF; there
isn't one, so RRF is the right default.

### 3.7 Why a cross-encoder reranker earns its latency cost, and where it sits

**Decision.** After RRF, take the top ~20 fused candidates and score each with
`BAAI/bge-reranker-base`, a cross-encoder that jointly encodes `(query, chunk)`; keep the top 5–8 for
context assembly.

**Alternatives considered.** No rerank (feed fused top-k straight to the LLM); an LLM-as-reranker
prompt; Cohere Rerank API; MMR for diversity instead of relevance reranking.

**Trade-off.** Bi-encoder retrieval (dense and, loosely, sparse) scores query and document
*independently*, so it cannot model term interaction — it retrieves things "about the same topic" but
not necessarily things that *answer the question*. A cross-encoder attends over query and candidate
together and is far more precise at "does this passage actually contain the answer", which directly
moves RAGAS context precision and faithfulness. The cost is latency: `N` forward passes of a ~300 MB
model on the hot path, ~50–200 ms on CPU for `N≈20`. That is affordable precisely because it runs on a
*short* list — hence its position: **after** cheap high-recall retrieval and fusion have cut millions
of chunks to ~20, **before** context assembly. Running it earlier (on hundreds of candidates) would
blow the latency budget; skipping it pushes the precision problem onto the LLM, which then wastes
context tokens or gets distracted by near-miss passages. LLM-as-reranker is slower and costs
generation tokens; Cohere Rerank reintroduces a paid API.

**At 100x scale.** Move the reranker to a GPU micro-service with dynamic batching; consider a
distilled/smaller reranker or ONNX/quantized inference; cache `(query_hash, chunk_id) → score`.

### 3.8 Multi-agent / LangGraph over a single RAG chain — and when it's over-engineering

**Decision.** A LangGraph graph: a router node classifies the query, then dispatches to one of three
specialized sub-agents (`api_reference`, `conceptual`, `troubleshooting`) that share one retrieval
tool and a common typed state object (`query`, `category`, `retrieved_chunks`, `draft_answer`,
`citations`).

**Alternatives considered.** A single linear RAG chain with one prompt; a single agent with tools and
a ReAct loop; a prompt-router only (branch the *prompt*, not the graph).

**Trade-off.** The three query types genuinely want different behavior: `api_reference` wants tight,
literal, signature-quoting answers at low temperature with a "quote the docs, don't paraphrase"
instruction; `conceptual` wants synthesis across several sections and tolerates longer context;
`troubleshooting` wants a symptom→cause→fix structure and often benefits from pulling the error string
verbatim into the sparse query. A single prompt doing all three compromises on each. LangGraph gives
an explicit, inspectable state machine — you can see which node ran, log the routing decision, add
loop-back edges — which matters for a portfolio piece meant to look production-grade and for
debugging. **When this is over-engineering:** if the query distribution were homogeneous, or the
corpus small and uniform, the router is pure overhead — added latency (an extra LLM call), an extra
failure mode (misrouting), more code — and a single well-written RAG chain would match it. The honest
interview framing: multi-agent here is justified by *measured* heterogeneity in query types, not
adopted because it's fashionable, and the graph is deliberately shallow (one routing hop, no
agent-to-agent chatter) to keep cost bounded.

### 3.9 Router design and failure modes

**Design.** The router is a single constrained LLM classification call. Categories, descriptions, and
few-shot examples are **not in code** — they are read from `config/corpora/<name>.yaml`
(`router_categories`), so a Pokémon corpus defines its own (`lore` / `game_mechanics` /
`competitive`) with no code change. Output is constrained to the known label set (enum / structured
output), with a confidence or top-2 signal where the provider supports it.

**Failure modes and handling.**
- *Invalid / unparseable label* → deterministic fallback to a generic sub-agent that runs the same
  hybrid retrieval with no category-specific query shaping or prompt.
- *Low-confidence or near-tie between two categories* → route to the generic path rather than guess;
  optionally union the retrieval from both candidate categories.
- *Systematic misclassification* (e.g. troubleshooting questions phrased conceptually) → caught by the
  eval set, which includes labelled `expected_category`; router accuracy is a tracked metric, and the
  fix is editing descriptions/examples in YAML, not code.
- *Router LLM outage / timeout* → skip routing entirely, go straight to the generic agent; the system
  degrades to a plain hybrid-RAG chain rather than failing.
- *Cost/latency* → the router prompt is tiny and cache-friendly; if it ever dominates latency it can
  be replaced by a local zero-shot classifier (the same bge model + a small head) with no interface
  change.

The key design property: **misrouting degrades quality, never correctness** — every category's
sub-agent can answer any question; the category only tunes retrieval shaping and prompt style.

### 3.10 Why RAGAS; what the three metrics measure and their blind spots

**Decision.** Evaluate with RAGAS on `faithfulness`, `answer_relevancy`, `context_precision` over a
curated 20–30 item set; commit baseline JSON, then commit deltas after tuning chunking and retrieval
params, under `evals/results/`.

**Alternatives considered.** Exact-match / F1 against gold answers (SQuAD-style); BLEU/ROUGE; a
hand-rolled LLM-as-judge rubric; human rating only.

**What each metric measures.**
- *Faithfulness* — are the claims in the answer entailed by the retrieved context? Decomposes the
  answer into atomic statements and checks each against the context. Targets hallucination.
  *Blind spot:* an answer can be perfectly faithful to context that is itself wrong or off-topic; says
  nothing about whether the answer is *useful*; judge-LLM-dependent.
- *Answer relevancy* — does the answer actually address the question (not evasive, not padded)?
  Computed by generating questions the answer would answer and comparing them to the real question.
  *Blind spot:* rewards on-topic answers even if factually wrong; penalizes correct answers that add
  useful caveats; insensitive to completeness.
- *Context precision* — of the retrieved chunks, how many are actually relevant, and are they ranked
  high? Directly measures the retrieval + rerank stack. *Blind spot:* precision, not recall — it does
  not tell you the answer-bearing chunk was *missing* entirely (that needs context recall, which needs
  gold contexts).

**Trade-off.** Lexical metrics (EM/F1/ROUGE) are cheap and deterministic but near-useless for
free-form RAG answers correct in many phrasings. RAGAS is reference-light (mostly needs questions +
retrieved contexts, not gold answers), decomposes the pipeline so you can see *where* a regression is
(retrieval vs generation), and is a recognized standard. The costs: every metric is an LLM call, so
scores are **noisy and judge-model-dependent** (pin the judge model and version), it costs tokens per
run, and small sets have wide confidence intervals — hence reporting *deltas* on a *fixed* set rather
than treating an absolute 0.82 as meaningful.

### 3.11 Constructing the eval set to avoid leakage and overfitting

- **Author questions from the docs, but not the chunks.** Write questions against the *concepts* and
  real user problems, then let retrieval find support. Do not write a question by paraphrasing one
  chunk — that bakes in the retrieval answer and inflates context precision.
- **Freeze it before tuning.** The 20–30 items are curated once and version-controlled. All chunking/
  retrieval tuning is measured against this frozen set; the set is never edited *because* a config
  change made an item fail.
- **Hold out a blind slice.** Keep ~30% of items unused during tuning and only run them at the end. If
  tuned and held-out deltas diverge sharply, the tuning overfit the visible items.
- **Diversity quotas.** Spread items across all three router categories, across doc areas (tutorial,
  advanced, deployment, security), and across difficulty: single-chunk factoid, multi-chunk synthesis,
  and "answer is a caveat / not in the docs" negatives to test refusal.
- **Include hard negatives.** Questions whose answer is *not* in the FastAPI docs, where the correct
  behavior is to say so — this catches a system that always confabulates.
- **Provenance, not gold prose.** Store `question`, `expected_category`, and `reference_source_urls`
  (which pages should be retrieved) rather than one gold answer string, so scoring does not overfit to
  one phrasing.
- **Refresh cadence.** Re-pin and partly regenerate the set when the docs version bumps; keep the old
  set for historical comparison.

### 3.12 Hosted LLM + provider abstraction; latency / quality / lock-in trade-offs

**Decision.** Generation goes through a thin provider interface
(`complete(messages, **params) -> str`) with three implementations behind it —
Gemini, Groq, and local Ollama — selected by one env var (`LLM_PROVIDER`). The
**default is Google Gemini `gemini-flash-lite-latest`** (free tier, hosted, fast; the `-latest`
alias survives Google's frequent model rotation, and the "lite" line has the least-restrictive
free-tier limits). Every hosted LLM call is wrapped in exponential backoff that honours the
provider's `retry-after` hint — free-tier 429s are expected traffic. The evaluation harness
(§3.10) makes ~7 LLM calls per item and is meant to run against a local Ollama model, not the
hosted free tier, whose per-day request cap a tuning loop would blow through in one sitting.

> **Note — the abstraction already paid for itself.** The original default was
> Groq `llama-3.3-70b-versatile`; it was moved to Gemini when Groq's sign-up flow
> proved unusable on the dev machine (a browser-agnostic OAuth redirect loop). The
> switch was a single new ~40-line file plus one line in the factory — no change to
> the graph, prompts, retrieval, or eval code. That is precisely the property this
> seam exists to provide, and it is a stronger interview anecdote than the original
> hypothetical: a real vendor problem, absorbed by the design in minutes.

The reasoning below applies to whichever hosted provider is default; Gemini and
Groq are near-equivalent on the axes that matter here (both free, both fast, both
well below GPT-class on hard reasoning but fine for grounded extraction).

**Alternatives considered.** OpenAI/Anthropic hosted APIs (top quality, per-token cost, hard lock-in
to their SDK + prompt quirks); Ollama-only (fully local, $0, but slow on a CPU box and heavy for a
low-tier deploy); Together/Fireworks/OpenRouter (paid OSS hosting).

**Trade-off.**
- *Cost:* Groq's free tier keeps the project at ~$0, a hard requirement.
- *Latency:* Groq's LPU inference is unusually fast (hundreds of tokens/sec), so generation is not the
  bottleneck — good for an interactive UI.
- *Quality:* Llama-3.3-70B is clearly below GPT-4-class models on hard reasoning, but for *grounded
  extraction and summarization over retrieved context* — what RAG generation is — the gap is small, and
  faithfulness depends more on retrieval quality than on the generator.
- *Lock-in:* the free tier has rate limits and no SLA and could change. The provider abstraction is
  the insurance: swapping to Ollama (offline demo, no rate limit) or to a paid endpoint (if quality
  needs to jump) is a one-file change, and prompts are kept provider-neutral. Accepted downside: a
  lowest-common-denominator interface (no provider-specific features like fine-grained tool-call
  formats or prompt caching wired in yet).

### 3.13 Deployment topology on a low-cost tier; where the bottlenecks are

**Topology.** One small always-on container on Railway/Fly.io running the Streamlit app + the
LangGraph runtime + the local models (bge-small embedder, bge-reranker-base). Postgres+pgvector is a
managed small instance (or the platform's Postgres add-on) reachable over the private network. Groq is
an outbound API call. Ingestion runs **offline** (locally or as a one-off job), writing to the same
DB; it is not part of the request-serving container's steady state.

```
[ Browser ]──HTTPS──▶[ App container: Streamlit + LangGraph + bge-small + bge-reranker ]
                              │ SQL (private net)            │ HTTPS
                              ▼                              ▼
                     [ Managed Postgres 16 + pgvector ]   [ Groq API ]
```

**Where it will hurt.**
- **Memory / cold start.** bge-small (~130 MB) + bge-reranker-base (~300 MB) + torch runtime is the
  bulk of the container's RAM and most of its cold-start time. Mitigations: load models once at
  process start (not per request), pin to CPU, consider ONNX/quantized weights, keep one instance
  warm.
- **Reranker CPU time.** ~50–200 ms per query for ~20 candidates, single-threaded on a shared vCPU —
  the largest controllable latency component. Cap candidate count; batch.
- **pgvector memory.** HNSW wants the graph resident; on a tiny Postgres instance a large index can
  cause cache thrashing. Fine at this corpus size; first thing to watch as corpora grow.
- **Groq rate limits.** Free-tier RPM/TPM caps are the throughput ceiling under concurrent users;
  handle 429s with backoff and a queue, fall back to Ollama or a paid key if sustained.
- **Single instance = no HA.** Acceptable for a portfolio project; the fix is horizontal replicas
  behind the platform LB with models baked into the image.

---

## 4. Known Limitations / What I'd Do Next

- **Faithfulness scores lower on code/instructional answers than on prose, for reasons not yet
  disambiguated.** Confirmed on the held-out set: conceptual/prose answers scored 0.94–1.00
  faithfulness; code-heavy `api_reference` answers and numbered-step `troubleshooting` answers scored
  0.26–0.6, despite retrieving the correct page 100% of the time. Either the model infers beyond the
  source when adapting a code sample or synthesising "steps" from prose, or RAGAS's claim-based
  decomposition doesn't map cleanly onto code blocks and imperative instructions in the first place.
  Next: an eval item type / metric that scores "does the code match the docs' own example" separately
  from prose faithfulness, rather than one claim-based score for both.
- **No context-recall metric.** RAGAS context precision tells me retrieved-junk ratio, not whether the
  answer-bearing chunk was missed. Next: add gold `reference_source_urls` to every eval item and
  compute retrieval recall@k.
- **Small, single-author eval set.** 20–30 items has wide confidence intervals and author bias. Next:
  grow to ~100, add a second author, mine real questions from FastAPI GitHub discussions.
- **English-only, small embedder.** bge-small is mid-pack and monolingual. Next: benchmark bge-base /
  gte-base, and fine-tune bge-small on in-domain (query, positive chunk) pairs mined from logs.
- **Postgres FTS is not true BM25.** `ts_rank` is a decent proxy but not Okapi BM25. Next: `pg_search`/
  ParadeDB or a dedicated search engine; add SPLADE as a learned-sparse third channel.
- **Router adds a failure mode for modest gain.** Next: quantify router lift with an ablation
  (multi-agent vs single chain on the eval set); if lift is small, collapse to prompt-routing.
- **No caching anywhere.** Every query re-embeds, re-retrieves, re-ranks, re-generates. Next: cache
  query embeddings, `(query, chunk)` rerank scores, and full answers keyed by normalized question +
  corpus version.
- **Ingestion is manual and full-rebuild-ish.** Next: content-hash per source file, upsert only
  changed chunks, and a scheduled job triggered by upstream doc releases.
- **No guardrails on output.** Next: a post-generation faithfulness check (reuse the reranker or a
  cheap NLI model) that flags or regenerates answers whose claims aren't supported by context.
- **Single instance, no observability stack.** Next: structured traces per graph node (LangSmith or
  OpenTelemetry), latency/error dashboards, per-stage timing surfaced in the UI.

---

## 5. Interview Talking Points (cheat-sheet)

- **One store, two retrieval modes.** Putting `vector(384)` and a generated `tsvector` in the same
  Postgres row makes hybrid search a single transactional SQL query and kills the dual-write sync
  problem a separate vector DB would create — and it's plenty fast at this corpus size.
- **Dense and sparse fail in opposite directions.** Dense handles paraphrase but smears exact symbols
  like `HTTPException` and `status_code=422`; sparse nails identifiers and pasted error strings but
  misses synonyms. Docs corpora need both.
- **RRF because the scores aren't comparable.** Cosine (0–1, compressed) and `ts_rank` (unbounded,
  length-dependent) can't be summed without brittle per-query normalization; RRF uses only rank, has
  one well-understood constant (`k=60`), and the reranker recovers the magnitude info RRF discards.
- **The reranker runs late and on a short list on purpose.** Cross-encoder precision is worth
  ~50–200 ms only after retrieval+fusion have cut the field to ~20; earlier it blows the latency
  budget, skipped it pushes the precision problem onto the LLM's context window.
- **Section-aware chunking respects the author's boundaries.** A `##` heading is a human-curated topic
  split; 450 tokens leaves headroom under bge-small's 512 limit for the breadcrumb, and overlap is
  spent only on forced splits so the index isn't full of near-duplicates.
- **Multi-agent is justified by measured query heterogeneity, not fashion.** Three query types want
  genuinely different prompts and retrieval shaping; the graph is deliberately one shallow routing
  hop, and misrouting degrades quality, never correctness — every sub-agent can answer anything.
- **Provider abstraction is lock-in insurance.** Groq gives GPT-adjacent grounded-answer quality at
  ~$0 and very low latency; when the free tier's limits bite, swapping to Ollama or a paid endpoint is
  a one-file change with provider-neutral prompts.
- **Evaluate deltas on a frozen, partly-blind set.** RAGAS metrics are LLM-judged and noisy, so an
  absolute 0.82 means little; a held-out slice never used during tuning is what proves the chunking/
  retrieval changes generalized instead of overfitting the visible items.

---

## Appendix: `chunks` table (actual schema — see `src/docqa/db/schema.sql`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           uuid PRIMARY KEY,                       -- uuid5(corpus:source_id:ordinal), stable across re-ingest
    corpus       text        NOT NULL,                   -- scoping: WHERE corpus = %s
    source_id    text        NOT NULL,                   -- e.g. tutorial/query-params
    url          text        NOT NULL DEFAULT '',        -- resolved public doc URL for citations
    title        text        NOT NULL DEFAULT '',        -- page H1
    section_path text[]      NOT NULL DEFAULT '{}',      -- breadcrumb, e.g. {Tutorial,"Query Parameters"}
    ordinal      integer     NOT NULL,                   -- chunk position within the source doc
    text         text        NOT NULL,                   -- chunk body (raw; breadcrumb prepended at embed time)
    token_count  integer     NOT NULL DEFAULT 0,
    embedding    vector(384),                            -- bge-small-en-v1.5, normalised
    tsv          tsvector,                               -- to_tsvector(title + breadcrumb + body),
                                                         -- set by upsert_chunks() at write time
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_corpus_idx ON chunks (corpus);
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (corpus, source_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx    ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embed_idx  ON chunks USING hnsw (embedding vector_cosine_ops);
```

> Note: the vector dimension (384) is hard-coded to match bge-small. Changing the embedding model is a
> schema migration, not a config flip — a deliberate constraint that keeps the index honest.
>
> `tsv` was originally a `GENERATED ... STORED` column, which is the tidier design, but the
> text-configuration form of `to_tsvector()` is `STABLE`, not `IMMUTABLE`, so Postgres rejects it in a
> generation expression. Rather than wrap it in a (dishonestly) `IMMUTABLE` SQL function, it's a plain
> column written by the single writer, `upsert_chunks()`. `connect()` also runs
> `CREATE EXTENSION IF NOT EXISTS vector` before registering the pgvector type adapter, so a fresh
> managed database works with no manual setup.
