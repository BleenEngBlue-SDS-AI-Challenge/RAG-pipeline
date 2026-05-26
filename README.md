# RAG from Scratch

**A production-minded Retrieval-Augmented Generation system built from scratch in Python — chunk → embed → reduce → cluster → visualize → store → retrieve.**

Every stage is seeded with `RANDOM_STATE = 42` and built around single-responsibility functions, making each step independently testable, swappable, and extensible.

> Built for the [SuperDataScience AI Challenge](https://www.skool.com/ai-challenge/about)

---

## What This Demonstrates

This project shows deep understanding of a RAG pipeline from the inside out — with custom chunking, a reproducibility-first engineering approach, and documented tradeoffs at every stage. Built as part of the SuperDataScience AI Challenge, the pipeline covers the full stack from raw text to semantic retrieval:

- **Custom boundary-aware chunking** that respects paragraph → sentence → word structure instead of hard-splitting on character count
- **OpenAI embeddings** (`text-embedding-3-small`) with a clean abstraction that makes the embedding model swappable in one line
- **UMAP dimensionality reduction** in both 2D and 3D, with a documented understanding of *where* determinism breaks down and two explicit mitigation strategies
- **Semantic clustering** via seeded KMeans, with LLM-generated cluster descriptions from `gpt-4o-mini`
- **Interactive visualization** — Matplotlib for 2D, Plotly for interactive 3D
- **ChromaDB vector store** with persistent storage and cosine similarity retrieval

The demo corpus is the [Netflix Culture Memo](https://jobs.netflix.com/culture) — a multi-topic document that makes cluster separation easy to reason about.

---

## Pipeline

```
Raw Text
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 1  │  chunk_text()          │  Boundary-aware chunking      │
│          │                        │  max 1000 chars, 50-char overlap│
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 2  │  embed_chunks()        │  OpenAI text-embedding-3-small │
│          │                        │  → float32 vectors             │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 3  │  reduce_to_2d/3d()     │  UMAP (primary) or PCA         │
│          │                        │  (deterministic fallback)       │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 4  │  cluster_embeddings()  │  KMeans with seeded centroids   │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 5  │  describe_cluster()    │  gpt-4o-mini summarizes each   │
│          │                        │  cluster's semantic theme       │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 6  │  plot_2d / plot_3d()   │  Matplotlib 2D scatter         │
│          │                        │  Interactive Plotly 3D scatter  │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 7  │  ChromaDB              │  PersistentClient → store →    │
│          │                        │  cosine similarity top-k query  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Visualizations

### 2D UMAP — Chunk Embedding Space

![UMAP 2D Projection of Chunk Embeddings](umap-2D.png)

17 text chunks from the Netflix Culture Memo projected into 2D via UMAP and colored by KMeans cluster assignment. **Cluster 0** (red, left) groups chunks covering Netflix's core cultural values — high performance, candor, respect, and organizational philosophy. **Cluster 1** (blue, right) captures operationally-focused content: compensation, policies, and day-to-day practices. The clean separation confirms that the raw embeddings carry strong semantic signal before any clustering is applied.

---

### 3D UMAP — Cluster Geometry

![UMAP 3D Projection of Chunk Embeddings](umap-3D.png)

The 3D view adds a third dimension to reveal additional structure in the embedding manifold. The two clusters remain clearly separable across all three axes. This view is most useful for spotting borderline chunks that appear merged in 2D but split cleanly along the third axis — a practical debugging tool for evaluating chunking and clustering quality.

---

## Key Design Decisions

### Boundary-Aware Chunking

Rather than hard-splitting on character count, the chunker searches backward from the right window boundary for the highest-priority natural break — prioritized as paragraph (`\n\n`) → sentence (`. `) → word (` `) — then forward from the left window boundary to the nearest word start, ensuring every chunk opens at a clean word boundary rather than mid-word. This keeps chunk openings readable and avoids partial tokens that can confuse the embedding model. Chunks below `min_size` are merged with neighbors rather than emitted as orphans.

`max_size` is a critical tuning parameter: too large and chunks lose topical focus, hurting retrieval precision; too small and individual chunks lose enough context to become semantically ambiguous.

### Reproducibility as a First-Class Concern

All stochastic components flow through a single `seed_all(RANDOM_STATE)` call. The project explicitly documents where determinism breaks down — UMAP's `pynndescent` nearest-neighbor backend — and provides two escape hatches rather than pretending the problem doesn't exist:

| Strategy | Tradeoff |
|---|---|
| `reduce_to_2d_cached()` / `reduce_to_3d_cached()` | Runs UMAP once, caches to `.npy` — reloads on every subsequent run |
| `reduce_to_2d_pca()` / `reduce_to_3d_pca()` | Strictly deterministic; sacrifices some visual cluster separation |

### Why UMAP Over PCA or t-SNE

| | PCA | t-SNE | UMAP |
|---|---|---|---|
| **Speed** | ⚡⚡⚡ Fastest | 🐢 Slowest | ⚡⚡ Fast |
| **Global structure** | ✅ Preserved | ❌ Poor | ✅ Good |
| **Local structure** | ⚠️ Limited | ✅ Excellent | ✅ Excellent |
| **Scalability** | ✅ Large datasets | ❌ Struggles >10k | ✅ Handles large data |
| **Reproducibility** | ✅ Deterministic | ⚠️ Stochastic | ⚠️ Stochastic (seedable) |

UMAP is the primary choice because it preserves both local cluster structure and global inter-cluster relationships — giving a more truthful picture of the embedding space than either alternative. PCA's limitation is that it projects onto directions of greatest *linear* variance; non-linear structure common in text embeddings is essentially invisible to it. t-SNE optimizes for local neighborhood structure at the expense of global relationships, which can smear inter-cluster geometry across the projection — how far apart two clusters appear tells you little about how semantically different they actually are. It also doesn't scale.

**PCA as a preprocessor:** Despite its limitations for visualization, PCA is documented here as a valuable preprocessing step before UMAP for high-dimensional data (10,000+ dims). Reducing to 30–50 PCA components first concentrates signal, cuts UMAP runtime significantly, and gives the nearest-neighbor graph a more meaningful distance metric — addressing the curse of dimensionality. A typical production pipeline: **PCA to ~30–50 dims → UMAP to 2D**, with the PCA component count selected via an elbow/scree plot.

### Single-Responsibility Functions

Every function does exactly one thing. Chunking, embedding, reduction, clustering, and storage are fully decoupled — each stage is independently testable and swappable. Swapping in a different embedding model, vector store, or chunking strategy requires changing one function and nothing else.

---

## Retrieval Example

```python
# Embed query with the same model used at index time
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["days off"]
)
query_embedding = response.data[0].embedding

# Top-3 most semantically relevant chunks via cosine similarity
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
```

ChromaDB returns the most relevant chunks along with their metadata. Query `"days off"` correctly surfaces chunks about Netflix's vacation policy (`"Take vacation."`) — not chunks about performance or values.

---

## Stack

| Layer | Technology |
|---|---|
| Embeddings | `openai` — `text-embedding-3-small` |
| Dimensionality Reduction | `umap-learn`, `scikit-learn` (PCA) |
| Clustering | `scikit-learn` — KMeans |
| 2D Visualization | `matplotlib` |
| 3D Visualization | `plotly` |
| Vector Store | `chromadb` — PersistentClient |
| Cluster Description | `openai` — `gpt-4o-mini` |
| Environment | `python-dotenv` |

Python `3.14.3` · Jupyter Notebook

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/BleenEngBlue/rag-pipeline.git
cd rag-pipeline
pip install openai python-dotenv numpy umap-learn scikit-learn matplotlib plotly chromadb

# 2. Set your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Run
jupyter notebook rag-pipeline.ipynb
# or: jupyter nbconvert --to script rag-pipeline.ipynb && python rag-pipeline.py
```

---

## Project Structure

```
rag-pipeline/
├── rag-pipeline.ipynb   # Fully executable top-to-bottom
├── .env                 # OPENAI_API_KEY — not committed
├── chroma_db/           # ChromaDB persistent storage — auto-created
├── reduced_2d.npy       # UMAP 2D cache — delete to re-run UMAP
├── reduced_3d.npy       # UMAP 3D cache — delete to re-run UMAP
└── README.md
```

> Add `chroma_db/`, `reduced_2d.npy`, `reduced_3d.npy`, and `.env` to `.gitignore`.

---

## Extension Points

This architecture is intentionally modular. Common next steps:

- **Swap the document** — replace the `document` string with any plain text corpus
- **Token-aware chunker** — plug in a `tiktoken`-based chunker for production token-budget control
- **Different embedding model** — any model returning float vectors works; just use the same model for both indexing and querying
- **Scale clusters** — adjust `n_clusters` for larger, more topically diverse documents
- **Close the RAG loop** — pass retrieved chunks as context to a chat completion call to add generation
- **PCA preprocessing** — prepend PCA to 30–50 dims before UMAP for faster, more consistent results on larger corpora

---

## Attribution

Built for the **[SuperDataScience AI Challenge](https://www.skool.com/ai-challenge/about)**.
