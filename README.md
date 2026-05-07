# RAG Pipeline — Semantic Chunking, Embedding & Cluster Visualization

> End-to-end RAG preprocessing pipeline with boundary-aware chunking, OpenAI embeddings, UMAP/KMeans clustering, 2D/3D visualization, and LLM-powered cluster labeling.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-text--embedding--3--small%20%7C%20gpt--4o--mini-412991)](https://platform.openai.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans%20%7C%20PCA-F7931E)](https://scikit-learn.org/)
[![UMAP](https://img.shields.io/badge/UMAP-Dimensionality%20Reduction-brightgreen)](https://umap-learn.readthedocs.io/)

---

## What This Is

A production-oriented RAG preprocessing pipeline built for the [SDS AI Challenge](https://www.skool.com/ai-challenge). It takes a raw document, chunks it with boundary awareness, embeds it via OpenAI, validates embedding structure through clustering and 2D/3D visualization, and uses `gpt-4o-mini` to label each discovered topic cluster.

The challenge specified boundary-aware chunking, visual quality validation, and LLM cluster labeling. Independent engineering decisions cover the choice of UMAP over t-SNE, the combined use of `random_state` and `n_jobs=1` for partial UMAP determinism, output caching, and a PCA fallback for fully reproducible projections. Embeddings are drop-in compatible with Pinecone, Weaviate, and pgvector.

**Demo knowledge base:** [Netflix Culture Memo](https://jobs.netflix.com/culture)

---

## Pipeline Overview

| Step | Stage | What Happens |
|------|-------|--------------|
| 01 | **Ingest** | Load and normalize raw text |
| 02 | **Chunk** | Boundary-aware splitting with configurable overlap |
| 03 | **Embed** | 1,536-dim dense vectors via `text-embedding-3-small` |
| 04 | **Reduce** | UMAP → 2D and 3D projections |
| 05 | **Cluster** | KMeans (k=2) on reduced embeddings |
| 06 | **Label** | `gpt-4o-mini` generates a thematic summary per cluster |

---

## Implementation Notes

The SDS AI Challenge specified boundary-aware chunking at chunk *ends*, the chunking parameter values, visual quality validation via clustering, and LLM cluster labeling as requirements. The following documents how those requirements were implemented, and where independent engineering decisions were made on top of them.

---

### Chunking — Required + Extended

The challenge required boundary-aware splitting at chunk ends — snapping cuts to natural language boundaries rather than a fixed character offset. The parameter values (`max_size`, `overlap`, `min_size`) were also provided as part of the spec.

**Configurable parameters (challenge-specified):**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_size` | 500 chars | Maximum chunk window |
| `overlap` | 50 chars | Context carried across chunk boundaries |
| `min_size` | 250 chars | Minimum size before merge or carry-forward |

The implementation snaps chunk ends to paragraphs first, then sentences, then word boundaries. Chunk *starts* use a simpler word-boundary snap — this was an implementation choice rather than a spec requirement, made to avoid splitting tokens mid-word after the overlap is applied.

Sub-minimum chunks are merged forward as a prefix into the next window rather than emitted as isolated fragments, reducing short-chunk retrieval noise at query time.

**Core helper functions:**

- `find_natural_boundary` — tries `\n\n`, then `. `, then ` `; only cuts if resulting chunk meets `min_size`
- `apply_boundary` — applies the cut and advances the end pointer past any mid-word position
- `handle_small_chunk` — merges sub-minimum chunks into the previous chunk or carries them forward
- `advance_start` — computes the next window start with overlap, calls `snap_to_word_start` for clean boundaries

---

### Visual Quality Validation — Required

The challenge specified validating embedding quality visually through clustering. UMAP + KMeans provides a practical sanity check before any data reaches the vector store: if the document's topics aren't meaningfully separable in embedding space, retrieval quality will suffer regardless of prompt engineering.

Applied to the Netflix Culture Memo, the pipeline surfaces a clean two-cluster separation — organizational principles vs. people and culture themes.

**2D UMAP projection** — static matplotlib scatter with annotated chunk indices:

![UMAP 2D Projection](UMAP_2D.png)

**3D UMAP projection** — interactive Plotly view that surfaces structure obscured in 2D:

![UMAP 3D Projection](UMAP_3D.png)

---

### LLM Cluster Labeling — Required

Automated cluster labeling via LLM was a challenge requirement. `gpt-4o-mini` receives up to 10 representative chunks per cluster and returns a 2–3 sentence thematic summary.

```python
messages=[
    {"role": "system", "content": "You are a helpful assistant that analyses and summarises groups of text chunks."},
    {"role": "user",   "content": f"Here are text chunks grouped by semantic similarity:\n\n{combined_text}\n\nIn 2–3 sentences, describe what topic or theme unites them."}
]
```

---

### Reproducibility — Independent Decision

Reproducibility handling was not part of the challenge spec. UMAP's known non-determinism — from `pynndescent`'s approximate nearest-neighbor graph construction and floating-point variability through its optimization loop — makes consistent outputs across runs genuinely difficult.

**UMAP over t-SNE** was a deliberate choice: UMAP better preserves global neighborhood structure at this data scale, runs faster, and has a `random_state` parameter that at least partially constrains randomness. t-SNE offers no equivalent.

**`random_state` + `n_jobs=1`** were set together intentionally. `random_state` alone is insufficient — UMAP's multi-threaded execution introduces non-deterministic floating-point ordering. Setting `n_jobs=1` forces single-threaded execution, which eliminates that source of variance.

**Caching** — `reduce_to_2d_cached()` and `reduce_to_3d_cached()` save UMAP output on first run and reload on subsequent runs, so UMAP executes exactly once:

```python
CACHE_2D = Path("reduced_2d.npy")

def reduce_to_2d_cached(embedding_matrix: np.ndarray, seed: int = RANDOM_STATE) -> np.ndarray:
    if CACHE_2D.exists():
        return np.load(CACHE_2D)
    reduced = umap.UMAP(n_components=2, random_state=seed, n_jobs=1).fit_transform(embedding_matrix)
    np.save(CACHE_2D, reduced)
    return reduced
```

> **Note:** Delete `.npy` cache files whenever chunks or embeddings change.

**PCA fallback** — added as a strictly deterministic alternative for cases where run-to-run plot identity matters more than visual cluster separation. PCA is linear and has no stochastic components, so output is identical across every run.

```python
from sklearn.decomposition import PCA

def reduce_to_2d_pca(embedding_matrix: np.ndarray, seed: int = RANDOM_STATE) -> np.ndarray:
    return PCA(n_components=2, random_state=seed).fit_transform(embedding_matrix)
```

---

## Implementation Reference

**Embedding:**

```python
response = client.embeddings.create(
    input=chunks,
    model="text-embedding-3-small"
)
embeddings = [item.embedding for item in response.data]
# Output: N × 1,536 float vectors
```

**Dimensionality reduction:**

```python
# 2D — matplotlib scatter with per-point index labels
reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_jobs=1)
reduced_2d = reducer.fit_transform(embedding_matrix)

# 3D — Plotly Scatter3d with one trace per cluster
reducer = umap.UMAP(n_components=3, random_state=RANDOM_STATE, n_jobs=1)
reduced_3d = reducer.fit_transform(embedding_matrix)
```

`n_jobs=1` disables parallelism whose non-deterministic thread scheduling would otherwise cause floating-point differences between runs.

**KMeans clustering:**

```python
kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
labels = kmeans.fit_predict(reduced)
```

`n_init=10` runs 10 independent centroid initializations to guard against local minima.

---

## Stack

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| Embeddings | `text-embedding-3-small` | 1,536-dim, strong semantic fidelity at low cost |
| Cluster labeling | `gpt-4o-mini` | Short summarization task; low latency and cost |
| Dimensionality reduction | UMAP | Preserves local neighborhood structure better than PCA or t-SNE at this scale |
| Deterministic reduction | PCA | Strictly deterministic fallback when run-to-run plot identity matters |
| Clustering | KMeans (scikit-learn) | Interpretable and fast; appropriate for known k |
| Visualization (2D) | matplotlib | Static cluster inspection with annotated chunk indices |
| Visualization (3D) | Plotly `Scatter3d` | Interactive and rotatable — surfaces structure 2D can obscure |
| Vector DB targets | Pinecone / Weaviate / pgvector | Embeddings are drop-in compatible; no modification required |

---

## Getting Started

**1. Install dependencies**

```bash
pip install openai python-dotenv numpy umap-learn scikit-learn plotly matplotlib
```

**2. Configure API access**

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

**3. Run the notebook**

```bash
jupyter notebook rag_pipeline_reproducible_umap_kmeans_2d3d_plots.ipynb
```

Execute all cells top to bottom. UMAP output is cached to `reduced_2d.npy` and `reduced_3d.npy` on first run.

---

## UMAP Reproducibility

Even with `random_state=42` and `n_jobs=1`, UMAP can produce axially shifted or reflected outputs between runs due to:

1. `pynndescent` (UMAP's nearest-neighbor backend) has internal randomness not fully controlled by UMAP's `random_state`
2. Floating-point non-determinism compounds through UMAP's optimization loop
3. The degree of determinism varies by `pynndescent` version

**Recommended:** use `reduce_to_2d_cached()` / `reduce_to_3d_cached()` so UMAP runs once and subsequent runs reload from disk. For guaranteed determinism with zero disk I/O, use `reduce_to_2d_pca()` / `reduce_to_3d_pca()` — at the cost of less pronounced cluster separation.

---

## Relevance by Role

| Role | What this demonstrates |
|------|------------------------|
| **AI / Applied ML Engineer** | UMAP vs. t-SNE tradeoff at this data scale; diagnosing and mitigating UMAP non-determinism via `random_state` + `n_jobs=1`; caching strategy for reproducible projections; PCA as a strictly deterministic fallback |
| **AI Product Engineer** | End-to-end RAG prototype on a real knowledge base; reproducibility vs. visual fidelity tradeoffs; embedding architecture compatible with Pinecone, Weaviate, and pgvector without modification |
| **LLM / Agent Engineer** | Multi-step pipeline with LLM calls at defined stages; `gpt-4o-mini` model selection for cost/latency appropriateness on a short summarization task |
| **Forward Deployed / Generalist Engineer** | Corpus-agnostic pipeline; drop-in vector store compatibility; no customer-specific modifications required |

---

*Built for the [Super Data Science AI Challenge](https://www.skool.com/ai-challenge) · Knowledge base: [Netflix Culture Memo](https://jobs.netflix.com/culture)*
