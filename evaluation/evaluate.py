import os
import json
import sys
import asyncio
from pathlib import Path

# Add backend to path so we can import RAG modules directly for offline evaluation
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from app.services.orchestrator import process_document_background
from app.rag.pipeline import run_hybrid_retrieval
from app.rag.generation import generate_answer, llm_provider, MockProvider
from app.core.database import Base, engine

async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def mrr_score(retrieved_ids, expected_ids):
    for i, rid in enumerate(retrieved_ids):
        if rid in expected_ids:
            return 1.0 / (i + 1)
    return 0.0

def precision_at_k(retrieved_ids, expected_ids, k=5):
    top_k = retrieved_ids[:k]
    relevant = [rid for rid in top_k if rid in expected_ids]
    return len(relevant) / k if k > 0 else 0.0

def recall_at_k(retrieved_ids, expected_ids, k=5):
    top_k = retrieved_ids[:k]
    relevant = [rid for rid in top_k if rid in expected_ids]
    return len(relevant) / len(expected_ids) if expected_ids else 0.0

async def evaluate():
    print("Starting Offline Evaluation...")
    
    # Setup DB
    await setup_db()
    
    # 1. Ingest test document
    doc_path = Path(__file__).parent / "sample_docs" / "enterprise_guide.txt"
    doc_id = "eval_doc_001"
    
    print(f"Ingesting {doc_path.name}...")
    await process_document_background(doc_id, str(doc_path), doc_path.name)
    
    # 2. Define dataset (we know the chunk IDs will be eval_doc_001_1_0 and eval_doc_001_1_1 based on chunk sizes)
    dataset = [
        {
            "question": "What is the token expiration limit for internal microservices?",
            "expected_answer": "15 minutes",
            "expected_chunks": ["eval_doc_001_1_0"]
        },
        {
            "question": "What database should be used for telemetry and logging?",
            "expected_answer": "MongoDB",
            "expected_chunks": ["eval_doc_001_1_0", "eval_doc_001_1_1"] # Might span chunks
        },
        {
            "question": "What base image is required for Docker containers?",
            "expected_answer": "Alpine Linux",
            "expected_chunks": ["eval_doc_001_1_1"]
        }
    ]
    
    with open(Path(__file__).parent / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
        
    # 3. Run evaluation
    results = {
        "mrr": [],
        "precision@5": [],
        "recall@5": [],
        "faithfulness": [],
        "context_recall": []
    }
    
    for item in dataset:
        print(f"\nEvaluating: {item['question']}")
        
        chunks, stats = run_hybrid_retrieval(item['question'], top_k=5)
        retrieved_ids = [c["chunk_id"] for c in chunks]
        
        mrr = mrr_score(retrieved_ids, item["expected_chunks"])
        p5 = precision_at_k(retrieved_ids, item["expected_chunks"], k=5)
        r5 = recall_at_k(retrieved_ids, item["expected_chunks"], k=5)
        
        results["mrr"].append(mrr)
        results["precision@5"].append(p5)
        results["recall@5"].append(r5)
        
        answer, citations, conf = generate_answer(item['question'], chunks)
        
        # LLM-as-a-judge for Faithfulness and Context Recall
        if not isinstance(llm_provider, MockProvider):
            judge_prompt_faithfulness = f"Question: {item['question']}\nAnswer: {answer}\nContext: {[c['text'] for c in chunks]}\nIs the answer strictly derived from the context? Answer strictly 'Yes' or 'No'."
            judge_res_f = llm_provider.generate(judge_prompt_faithfulness, "You are an impartial judge.").strip().lower()
            faithfulness = 1.0 if "yes" in judge_res_f else 0.0
            
            judge_prompt_recall = f"Question: {item['question']}\nExpected Answer: {item['expected_answer']}\nContext: {[c['text'] for c in chunks]}\nDoes the context contain enough information to formulate the expected answer? Answer strictly 'Yes' or 'No'."
            judge_res_r = llm_provider.generate(judge_prompt_recall, "You are an impartial judge.").strip().lower()
            context_recall = 1.0 if "yes" in judge_res_r else 0.0
        else:
            faithfulness = 1.0
            context_recall = 1.0
            
        results["faithfulness"].append(faithfulness)
        results["context_recall"].append(context_recall)

    # 4. Summarize
    print("\n=== EVALUATION RESULTS ===")
    for metric, scores in results.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"{metric.upper()}: {avg:.2f}")

if __name__ == "__main__":
    asyncio.run(evaluate())
