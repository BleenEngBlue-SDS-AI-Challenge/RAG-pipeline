# RAG Pipeline — Semantic Chunking, Embedding & Cluster Visualization

**End-to-end retrieval-augmented generation (RAG) preprocessing · Python · OpenAI API · scikit-learn**

A retrieval-augmented generation (RAG) preprocessing pipeline covering document ingestion, boundary-aware chunking, dense vector embedding, unsupervised clustering, and large language model (LLM)-assisted interpretation — built with an emphasis on the decisions that production deployments actually require.

**Stack:** `text-embedding-3-small` · `gpt-4o-mini` · KMeans clustering · UMAP dimensionality reduction · Plotly 3D · vector databases (Pinecone / Weaviate / pgvector) · natural language processing (NLP) · semantic search · information retrieval

---

## Pipeline

| Step | Stage | Description |
|------|-------|-------------|
| 01 | **Ingest** | Load & normalize raw text from source document |
| 02 | **Chunk** | Boundary-aware splitting with configurable overlap |
| 03 | **Embed** | 1536-dim dense vectors via `text-embedding-3-small` |
| 04 | **Cluster** | KMeans (k=2) on high-dimensional embeddings |
| 05 | **Reduce** | UMAP → 2D/3D for visualization & inspection |
| 06 | **Label** | `gpt-4o-mini` interprets each discovered cluster |

---

## Approach

The pipeline was designed around a few decisions that tutorial implementations tend to skip. Chunking snaps to natural language boundaries — paragraphs first, then sentences, then words — rather than cutting at a fixed character count. Chunks below a minimum size are merged with adjacent ones rather than emitted standalone, which reduces short-chunk retrieval noise at query time. A 50-character overlap ensures that context straddling a boundary is represented in both adjacent chunks.

Before the vector store sees any data, Uniform Manifold Approximation and Projection (UMAP) combined with KMeans clustering provides a practical sanity check on embedding quality: if the document's topics aren't meaningfully separable in vector space, retrieval quality will suffer regardless of how the prompts are written. Applied to the Netflix Culture Memo, the pipeline surfaces a clean two-cluster separation — organizational principles vs. people and culture themes — which can be inspected in the interactive 3D projection.

Cluster labeling uses `gpt-4o-mini` rather than a larger model: the task is short summarization, and the tradeoff favors speed and cost over additional reasoning capacity. The embedding structure is compatible with production vector databases — Pinecone, Weaviate, and pgvector — without modification.

---

## Model & Tool Selection

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| Embeddings | `text-embedding-3-small` | 1536-dim, strong semantic fidelity at low cost — well-suited for retrieval at scale |
| Cluster labeling | `gpt-4o-mini` | Short summarization task; low latency and cost without meaningful quality loss |
| Dimensionality reduction | UMAP | Preserves local neighborhood structure better than PCA or t-SNE at this scale |
| Clustering | KMeans (scikit-learn) | Interpretable and fast; appropriate for k=2 with known cluster count |
| Visualization | Plotly (3D) | Interactive and rotatable — surfaces structure that 2D projections can obscure |

---

## Getting Started

**1. Install dependencies**

```bash
pip install openai python-dotenv numpy umap-learn scikit-learn plotly
```

**2. Configure API access**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

**3. Run the notebook**

```bash
jupyter notebook RAG-scratchpad.ipynb
```

Execute all cells top to bottom.

---

## Relevant to

| Role | Applicable work |
|------|----------------|
| AI / Applied ML Engineer | Embedding pipeline, chunking heuristics, clustering, dimensionality reduction, visual evaluation of vector space structure |
| AI Product Engineer | End-to-end RAG prototype on a real knowledge base; tradeoff decisions on chunk size, overlap, and retrieval granularity |
| LLM / Agent Engineer | Structured multi-step pipeline with LLM calls at defined stages; model selection based on task requirements |
| Solutions / Forward-Deployed | Corpus-agnostic pipeline; embedding and retrieval architecture adaptable to customer-specific deployments |

---

*Built for the [Super Data Science AI Challenge](https://www.skool.com/ai-challenge) · Knowledge base: [Netflix Culture Memo](https://jobs.netflix.com/culture)*
