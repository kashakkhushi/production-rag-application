# System Architecture Diagrams

## Chat Flow (Sequence Diagram)

Below is the verified sequence diagram for the `/chat` endpoint. It visualizes the strict layering from the frontend, down through the API orchestration, into the hybrid retrieval pipeline, and finally to the LLM verification logic.

```mermaid
sequenceDiagram
    participant Frontend as Next.js Frontend
    participant API as router.py (/chat)
    participant ChatService as chat_service.py
    participant Hybrid as pipeline.py (Hybrid Retrieval)
    participant Chroma as ChromaDB (Dense)
    participant BM25 as Sparse Index (Lexical)
    participant Reranker as Cross-Encoder
    participant Gen as generation.py
    participant LLM as Gemini API (or Mock)

    Frontend->>API: POST /chat {query: "..."}
    API->>ChatService: process_chat_query(query)
    
    rect rgb(240, 248, 255)
        Note over ChatService, Hybrid: ThreadPool (Non-blocking)
        ChatService->>Hybrid: run_hybrid_retrieval()
        par
            Hybrid->>Chroma: search(query, top_k)
            Chroma-->>Hybrid: Dense Candidates
        and
            Hybrid->>BM25: search(query, top_k)
            BM25-->>Hybrid: Sparse Candidates
        end
        Hybrid->>Hybrid: Normalize & Combine Scores
        Hybrid->>Reranker: predict(query, combined_chunks)
        Reranker-->>Hybrid: final_scores
        Hybrid-->>ChatService: Top Reranked Chunks
    end

    rect rgb(255, 245, 238)
        Note over ChatService, Gen: ThreadPool (Non-blocking)
        ChatService->>Gen: generate_answer(query, chunks)
        alt max_score < 0.3
            Gen-->>ChatService: "I could not find sufficient information..." (Fallback)
        else max_score >= 0.3
            Gen->>LLM: LLMProvider.generate(prompt)
            LLM-->>Gen: Markdown Response with [chunk_ids]
            Gen->>Gen: Validate citations exist in retrieved chunks
            Gen-->>ChatService: Validated Response + Citations
        end
    end
    
    ChatService-->>API: ChatResponse
    API-->>Frontend: JSON payload + retrieval_stats
```
