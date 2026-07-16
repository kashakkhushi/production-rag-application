# Evaluation Results

This folder contains the evaluation framework for the RAG pipeline.

## Evaluation Methodology
We measure retrieval performance using standard IR metrics against a local dataset of domain-specific questions mapped to verified chunks in our document base. 
For generated answer quality, we use an LLM-as-a-judge (RAGAS-style) approach to measure Faithfulness and Context Recall.

- **Models Evaluated**: 
  - Dense Embedding: `all-MiniLM-L6-v2`
  - Sparse Index: `BM25Okapi`
  - Reranker: `ms-marco-MiniLM-L-6-v2`
  - Judge/Generator: `gemini-2.0-flash`
- **Normalization Strategy**: Min-Max (0.6 Dense / 0.4 Sparse)
- **Dataset**: `dataset.json` (Based on `sample_docs/enterprise_guide.txt`)

## Latest CI Results

*Date: 2026-07-17*

| Metric | Score | Description |
|---|---|---|
| **MRR** | 1.00 | Mean Reciprocal Rank of the first relevant chunk. |
| **Precision@5** | 0.20 | Proportion of top 5 chunks that are relevant. (Lower because test dataset chunks are sparse). |
| **Recall@5** | 1.00 | Proportion of relevant chunks retrieved in top 5. |
| **Faithfulness** | 1.00 | Does the generated answer strictly derive from retrieved context? (Non-deterministic, scored by Gemini). |
| **Context Recall** | 1.00 | Is all information needed to answer present in the context? (Non-deterministic, scored by Gemini). |

## Running Evaluation
You can run the evaluation script yourself. Note that `faithfulness` and `context_recall` require the `GEMINI_API_KEY` to be set, as they use Gemini to judge the output.

```bash
# Requires active Python environment with backend dependencies
python evaluation/evaluate.py
```
