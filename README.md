# RAG Pipeline — Semantic Chunking, Embedding & Cluster Visualization

**End-to-end retrieval-augmented generation (RAG) preprocessing · Python · OpenAI API · scikit-learn**

A retrieval-augmented generation (RAG) preprocessing pipeline covering document ingestion, boundary-aware chunking, dense vector embedding, unsupervised clustering, and large language model (LLM)-assisted interpretation — built with an emphasis on the decisions that production deployments actually require.

**Stack:** `text-embedding-3-small` · `gpt-4o-mini` · KMeans clustering · UMAP dimensionality reduction · Plotly 3D · matplotlib · NumPy · scikit-learn · python-dotenv · vector databases (Pinecone / Weaviate / pgvector) · natural language processing (NLP) · semantic search · information retrieval

---

## Pipeline

| Step | Stage | Description |
|------|-------|-------------|
| 01 | **Ingest** | Load & normalize raw text from source document |
| 02 | **Chunk** | Boundary-aware splitting with configurable overlap |
| 03 | **Embed** | 1536-dim dense vectors via `text-embedding-3-small` |
| 04 | **Cluster** | KMeans (k=2) on high-dimensional embeddings |
| 05 | **Reduce** | UMAP → 2D and 3D for visualization & inspection |
| 06 | **Label** | `gpt-4o-mini` interprets each discovered cluster |

---

## Approach

The pipeline was designed around a few decisions that tutorial implementations tend to skip.

**Chunking** snaps to natural language boundaries — paragraphs first, then sentences, then words — rather than cutting at a fixed character count. A configurable `max_size` (default: 500 chars), `min_size` (default: 250 chars), and `overlap` (default: 50 chars) govern the process. Chunks that fall below `min_size` mid-document are merged forward as a prefix into the next window rather than emitted as isolated fragments — reducing short-chunk retrieval noise at query time. The advance logic always snaps the next start position to a word boundary, so chunks never begin mid-word. Several helper functions handle the edge cases directly:

- `find_natural_boundary` — tries paragraph breaks (`\n\n`), then sentence breaks (`. `), then word breaks (` `) within the current window, only accepting a cut if the resulting chunk meets the minimum size threshold
- `apply_boundary` — applies the cut and advances the end pointer past any mid-word position
- `handle_small_chunk` — merges sub-minimum chunks into the previous chunk if possible, or carries them forward as a prefix
- `advance_start` — computes the next window start with overlap, guarding against backward movement, and calls `snap_to_word_start` to guarantee clean boundaries

**Embedding quality validation** happens before the vector store sees any data. UMAP combined with KMeans clustering provides a practical sanity check: if the document's topics are not meaningfully separable in vector space, retrieval quality will suffer regardless of how prompts are written. Applied to the Netflix Culture Memo, the pipeline surfaces a clean two-cluster separation — organizational principles vs. people and culture themes — visible in both the 2D matplotlib scatter and the interactive 3D Plotly projection.

**Cluster labeling** calls `gpt-4o-mini` with a structured system/user prompt, passing up to 10 chunks per cluster to stay within token limits. The task is short thematic summarization; the tradeoff favors low latency and cost over additional reasoning capacity. Output is post-processed with `textwrap.fill` and a terminal-width guard that safely falls back to 88 columns in notebook environments.

The embedding structure is compatible with production vector databases — Pinecone, Weaviate, and pgvector — without modification.

---

## Implementation Details

**Chunker parameters** (`chunk_text` function):

| Parameter | Default | Role |
|-----------|---------|------|
| `max_size` | 500 chars | Maximum chunk window |
| `overlap` | 50 chars | Context carried across chunk boundaries |
| `min_size` | 250 chars | Minimum chunk size before merge/carry-forward |

**Embedding API call:**

```python
response = client.embeddings.create(
    input=chunks,
    model="text-embedding-3-small"
)
embeddings = [item.embedding for item in response.data]
# Output: N × 1536 float vectors
```

**Dimensionality reduction — 2D (inspection) and 3D (interactive):**

```python
# 2D — matplotlib scatter with per-point index labels
reduced_2d = umap.UMAP(n_components=2, n_jobs=1).fit_transform(np.array(embeddings))

# 3D — Plotly Scatter3d with one trace per cluster for a clean legend
reduced_3d = umap.UMAP(n_components=3, n_jobs=1).fit_transform(np.array(embeddings))
```

Fixed `n_jobs=1` ensures reproducible projections across runs.

**KMeans clustering:**

```python
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(reduced_embeddings)
```

`n_init=10` runs 10 independent centroid initializations to guard against local minima.

**LLM cluster labeling prompt structure:**

```python
messages=[
    {"role": "system", "content": "You are a helpful assistant that analyzes and summarizes groups of text chunks."},
    {"role": "user",   "content": f"Here are text chunks grouped by semantic similarity:\n\n{combined_text}\n\nIn 2-3 sentences, describe what topic or theme unites these chunks."}
]
```

---

## Model & Tool Selection

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| Embeddings | `text-embedding-3-small` | 1536-dim, strong semantic fidelity at low cost — well-suited for retrieval at scale |
| Cluster labeling | `gpt-4o-mini` | Short summarization task; low latency and cost without meaningful quality loss |
| Dimensionality reduction | UMAP | Preserves local neighborhood structure better than PCA or t-SNE at this scale |
| Clustering | KMeans (scikit-learn) | Interpretable and fast; appropriate for k=2 with known cluster count |
| Visualization (2D) | matplotlib | Quick inspection of cluster separation with annotated chunk indices |
| Visualization (3D) | Plotly (`Scatter3d`) | Interactive and rotatable — surfaces structure that 2D projections can obscure |

---

## Getting Started

**1. Install dependencies**

```bash
pip install openai python-dotenv numpy umap-learn scikit-learn plotly matplotlib
```

**2. Configure API access**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

**3. Run the notebook**

```bash
jupyter notebook RAG-initial.ipynb
```

Execute all cells top to bottom.

---

## Relevant to

| Role | Applicable work |
|------|----------------|
| AI / Applied ML Engineer | Custom boundary-aware chunker with merge logic and overlap; embedding pipeline; KMeans on high-dimensional vectors; 2D/3D dimensionality reduction; visual evaluation of vector space structure |
| AI Product Engineer | End-to-end RAG prototype on a real knowledge base (Netflix Culture Memo); configurable chunking parameters; tradeoff decisions on chunk size, overlap, and retrieval granularity |
| LLM / Agent Engineer | Structured multi-step pipeline with LLM calls at defined stages; prompt design for cluster interpretation; model selection (embedding vs. chat) based on task requirements and cost/latency tradeoffs |
| Solutions / Forward-Deployed | Corpus-agnostic pipeline; embedding and retrieval architecture directly compatible with Pinecone, Weaviate, and pgvector; no modifications required for customer-specific deployments |

---

*Built for the [Super Data Science AI Challenge](https://www.skool.com/ai-challenge) · Knowledge base: [Netflix Culture Memo](https://jobs.netflix.com/culture)*
