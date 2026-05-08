# RAG Pipeline — Chunking, Embedding, Clustering & Retrieval Chat

> End-to-end RAG pipeline: boundary-aware chunking → OpenAI embeddings → UMAP/KMeans clustering → ChromaDB vector storage → Gradio chat interface with live retrieval.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-text--embedding--3--small%20%7C%20gpt--4.1--mini-412991)](https://platform.openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)](https://www.trychroma.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans%20%7C%20PCA-F7931E)](https://scikit-learn.org/)
[![UMAP](https://img.shields.io/badge/UMAP-Dimensionality%20Reduction-brightgreen)](https://umap-learn.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/Gradio-Chat%20Interface-ff7c00)](https://gradio.app/)

---

## What This Is

A production-oriented RAG pipeline built for the [SDS AI Challenge](https://www.skool.com/ai-challenge). It takes a raw document through the full retrieval-augmented generation stack: boundary-aware chunking, OpenAI embeddings, UMAP + KMeans clustering with 2D/3D visualization, LLM-powered cluster labeling, persistent vector storage in ChromaDB, and a Gradio chat interface that answers questions by retrieving live context from the vector store.

The pipeline is designed to be corpus-agnostic — swap in any document and the rest runs unchanged.

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
| 07 | **Store** | Chunks + embeddings persisted to ChromaDB |
| 08 | **Retrieve** | Query embedding → top-k similarity search |
| 09 | **Generate** | `gpt-4.1-mini` answers with retrieved context injected |
| 10 | **Chat** | Gradio `ChatInterface` for multi-turn conversation |

---

## Implementation Notes

### Chunking

The challenge required boundary-aware splitting at chunk ends — snapping cuts to natural language boundaries rather than fixed character offsets. The parameter values (`max_size`, `overlap`, `min_size`) were also provided as part of the spec.

**Configurable parameters:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_size` | 1,000 chars | Maximum chunk window |
| `overlap` | 50 chars | Context carried across chunk boundaries |
| `min_size` | 500 chars | Minimum size before merge or carry-forward |

The implementation snaps chunk ends to paragraphs first (`\n\n`), then sentences (`. `), then word boundaries (` `). A boundary only qualifies when the resulting chunk meets `min_size`. Chunk starts use a simpler word-boundary snap to avoid splitting tokens mid-word after the overlap is applied.

Sub-minimum chunks are merged forward as a prefix into the next window rather than emitted as isolated fragments, which reduces short-chunk retrieval noise at query time.

**Core helper functions:**

- `find_natural_boundary` — tries `\n\n`, then `. `, then ` `; only cuts if the resulting chunk meets `min_size`
- `apply_boundary` — applies the cut and advances the end pointer past any mid-word position
- `handle_small_chunk` — merges sub-minimum chunks into a neighbor or carries them forward as a prefix
- `advance_start` — computes the next window start with overlap, calls `snap_to_word_start` for clean boundaries

Chunking is fully deterministic — no randomness is involved. Reproducibility here comes from the algorithm itself.

---

### Embeddings

Each chunk is embedded using OpenAI's `text-embedding-3-small`, producing a 1,536-dimensional dense vector. The embedding model is deterministic for fixed input, so no random seed is required at this stage. Using a pinned model string ensures consistent dimensionality across runs.

```python
response = client.embeddings.create(input=chunks, model="text-embedding-3-small")
embeddings = [item.embedding for item in response.data]
# Output: N × 1,536 float32 vectors
```

---

### Visual Quality Validation — Cluster Visualization

UMAP + KMeans provides a practical sanity check before data reaches the vector store. If the document's topics aren't meaningfully separable in embedding space, retrieval quality will suffer regardless of prompt engineering.

Applied to the Netflix Culture Memo, the pipeline surfaces a clean two-cluster separation — organizational principles vs. people and culture themes.

- **2D projection** — static matplotlib scatter with annotated chunk indices
- **3D projection** — interactive Plotly view that surfaces structure the 2D projection can obscure

```python
# 2D
reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_jobs=1)
reduced_2d = reducer.fit_transform(embedding_matrix)

# 3D
reducer = umap.UMAP(n_components=3, random_state=RANDOM_STATE, n_jobs=1)
reduced_3d = reducer.fit_transform(embedding_matrix)
```

---

### LLM Cluster Labeling

`gpt-4o-mini` receives up to 10 representative chunks per cluster and returns a 2–3 sentence thematic summary, giving human-readable labels to what the geometry surfaced.

```python
messages=[
    {"role": "system", "content": "You are a helpful assistant that analyses and summarises groups of text chunks."},
    {"role": "user",   "content": f"Here are text chunks grouped by semantic similarity:\n\n{combined_text}\n\nIn 2–3 sentences, describe what topic or theme unites them."}
]
```

---

### Vector Storage — ChromaDB

Chunks and their embeddings are persisted to a local ChromaDB collection. Metadata (source document, chunk index) is stored alongside each record to support filtered retrieval. The collection is wiped and rebuilt on each pipeline run to maintain a clean state during development.

```python
chroma_client = chromadb.PersistentClient("./chroma_db")
collection = chroma_client.get_or_create_collection(name="netflix_culture_memo_chunks")

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=raw_embeddings,
    documents=chunks,
    metadatas=[{"source": "netflix_culture_pdf", "chunk_index": i} for i in range(len(chunks))]
)
```

ChromaDB's embedding format is directly compatible with Pinecone, Weaviate, and pgvector — swapping the vector store requires only changing the storage and query calls, not the upstream pipeline.

---

### Retrieval

Queries are embedded with the same model used for chunks to guarantee vector space compatibility. The top-k most similar chunks are retrieved from ChromaDB and stitched together as context.

```python
response = client.embeddings.create(model="text-embedding-3-small", input=[query])
query_embedding = response.data[0].embedding

results = collection.query(query_embeddings=[query_embedding], n_results=8)
context = "\n---\n".join(results["documents"][0])
```

---

### RAG Chat Interface

The final step surfaces retrieval in a multi-turn Gradio chat. Each user message is embedded, retrieves top-8 chunks, injects them as context into the system message, and passes the full conversation history to `gpt-4.1-mini` for generation.

```python
def response_ai(message, history):
    # Embed the query
    response = client.embeddings.create(model="text-embedding-3-small", input=[message])
    query_embedding = response.data[0].embedding

    # Retrieve top-8 chunks
    results = collection.query(query_embeddings=[query_embedding], n_results=8)
    context = "\n---\n".join(results["documents"][0])

    # Augment system message with retrieved context
    system_message_enhanced = system_message + "\n\nContext:\n" + context

    # Generate with full conversation history
    messages = [{"role": "system", "content": system_message_enhanced}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
    return response.choices[0].message.content

gr.ChatInterface(fn=response_ai).launch(inbrowser=True)
```

---

### Reproducibility

Every source of randomness is seeded with `RANDOM_STATE = 42` via a `seed_all()` function that covers Python's `random`, NumPy, and optionally PyTorch if installed.

**UMAP over t-SNE** was a deliberate choice: UMAP better preserves global neighborhood structure at this data scale, runs faster, and exposes a `random_state` parameter. `random_state` + `n_jobs=1` together partially constrain UMAP's output — `random_state` seeds the internal PRNG; `n_jobs=1` disables multi-threaded execution, which eliminates non-deterministic floating-point ordering from thread scheduling.

Even with both set, UMAP can still produce axially shifted or reflected projections run-to-run due to `pynndescent`'s internal randomness and version-sensitivity. Two mitigation strategies are provided:

**Caching** — run UMAP once, save to `.npy`, reload on subsequent runs:

```python
CACHE_2D = Path("reduced_2d.npy")

def reduce_to_2d_cached(embedding_matrix, seed=RANDOM_STATE):
    if CACHE_2D.exists():
        return np.load(CACHE_2D)
    reduced = umap.UMAP(n_components=2, random_state=seed, n_jobs=1).fit_transform(embedding_matrix)
    np.save(CACHE_2D, reduced)
    return reduced
```

> Delete `.npy` files whenever chunks or embeddings change.

**PCA fallback** — strictly deterministic, no disk I/O required, at the cost of less pronounced visual cluster separation:

```python
from sklearn.decomposition import PCA

def reduce_to_2d_pca(embedding_matrix, seed=RANDOM_STATE):
    return PCA(n_components=2, random_state=seed).fit_transform(embedding_matrix)
```

---

## Stack

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| Embeddings | `text-embedding-3-small` | 1,536-dim, strong semantic fidelity at low cost |
| Cluster labeling | `gpt-4o-mini` | Short summarization task; low latency and cost |
| Chat generation | `gpt-4.1-mini` | Capable and cost-efficient for RAG Q&A |
| Dimensionality reduction | UMAP | Preserves local neighborhood structure better than PCA or t-SNE at this scale |
| Deterministic reduction | PCA | Strictly deterministic fallback when run-to-run plot identity matters |
| Clustering | KMeans (scikit-learn) | Interpretable and fast; appropriate for known k |
| Vector store | ChromaDB (persistent) | Local, zero-infrastructure vector DB; embeddings are drop-in compatible with Pinecone, Weaviate, and pgvector |
| Visualization (2D) | matplotlib | Static cluster inspection with annotated chunk indices |
| Visualization (3D) | Plotly `Scatter3d` | Interactive and rotatable — surfaces structure 2D can obscure |
| Chat UI | Gradio `ChatInterface` | Minimal-overhead multi-turn interface; shareable with `share=True` |

---

## Getting Started

**1. Install dependencies**

```bash
pip install openai python-dotenv numpy umap-learn scikit-learn matplotlib plotly chromadb gradio
```

**2. Configure API access**

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

**3. Run the notebook**

```bash
jupyter notebook RAG-scratchpad3.ipynb
```

Execute all cells top to bottom. ChromaDB stores data in `./chroma_db/`. UMAP output can be cached to `reduced_2d.npy` and `reduced_3d.npy` using the cached variants to avoid recomputation during development. The Gradio chat interface launches automatically in the final cell.

---

## UMAP Reproducibility Reference

Even with `random_state=42` and `n_jobs=1`, UMAP can produce shifted or reflected outputs between runs because:

1. `pynndescent` (UMAP's nearest-neighbor backend) has internal randomness not fully controlled by UMAP's `random_state`
2. Floating-point non-determinism compounds through UMAP's optimization loop
3. The degree of determinism varies by `pynndescent` version

**Recommended:** use `reduce_to_2d_cached()` / `reduce_to_3d_cached()` so UMAP runs once per corpus and subsequent runs reload from disk. For guaranteed run-to-run identity with no disk I/O, use `reduce_to_2d_pca()` / `reduce_to_3d_pca()` — at the cost of less pronounced cluster separation.

---

## Relevance by Role

| Role | What this demonstrates |
|------|------------------------|
| **AI / Applied ML Engineer** | UMAP vs. t-SNE tradeoff at this data scale; mitigating UMAP non-determinism via `random_state` + `n_jobs=1`, caching, and PCA fallback; embedding pipeline architecture compatible with any production vector store |
| **AI Product Engineer** | End-to-end RAG prototype on a real knowledge base; visual embedding validation before production; vector store portability (ChromaDB → Pinecone / Weaviate / pgvector without upstream changes) |
| **LLM / Agent Engineer** | Multi-step pipeline with LLM calls at defined stages; context assembly from retrieved chunks; multi-turn conversation with injected retrieval context; model selection for cost/latency at each stage |
| **Forward Deployed / Generalist Engineer** | Corpus-agnostic design — swap the document, re-run the cells; drop-in vector store compatibility; Gradio interface requires no frontend work to demo to stakeholders |

---

*Built for the [Super Data Science AI Challenge](https://www.skool.com/ai-challenge) · Knowledge base: [Netflix Culture Memo](https://jobs.netflix.com/culture)*
