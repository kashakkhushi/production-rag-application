import pytest
from unittest.mock import patch, MagicMock
from app.rag.chunking import recursive_character_chunking
from app.rag.pipeline import min_max_normalize
from app.rag.generation import generate_answer, MockProvider

def test_recursive_character_chunking():
    text = "A" * 1000
    chunks = recursive_character_chunking(text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert len(chunks[0]) <= 500
    
def test_min_max_normalize():
    scores = [0.1, 0.5, 0.9]
    norm_scores = min_max_normalize(scores)
    assert norm_scores[0] == 0.0
    assert norm_scores[1] == 0.5
    assert norm_scores[2] == 1.0
    
    # Test identical scores
    scores = [0.5, 0.5]
    norm_scores = min_max_normalize(scores)
    assert norm_scores == [0.5, 0.5]

def test_generate_answer_hallucination_prevention():
    # Low confidence chunks should trigger refusal
    chunks = [{"chunk_id": "1", "text": "foo", "reranker_score": 0.1, "filename": "doc.txt", "page_number": 1}]
    answer, citations, conf = generate_answer("What is bar?", chunks)
    assert "could not find sufficient information" in answer.lower()
    assert citations == []

@patch('app.rag.generation.llm_provider')
def test_generate_answer_with_citations(mock_provider):
    # Setup mock to return a response with citations
    mock_provider.generate.return_value = "This is a fact [doc_1_1_0]. This is another fact [doc_1_1_1]."
    
    chunks = [
        {"chunk_id": "doc_1_1_0", "text": "Context A", "reranker_score": 0.9, "filename": "doc.txt", "page_number": 1},
        {"chunk_id": "doc_1_1_1", "text": "Context B", "reranker_score": 0.8, "filename": "doc.txt", "page_number": 1}
    ]
    
    answer, citations, conf = generate_answer("What are the facts?", chunks)
    
    assert "This is a fact" in answer
    assert len(citations) == 2
    assert citations[0].chunk_id in ["doc_1_1_0", "doc_1_1_1"]
    assert conf == "high"

@patch('app.rag.generation.llm_provider')
def test_generate_answer_hallucinated_citation_rejected(mock_provider):
    # LLM hallucinates a citation [doc_999] not in chunks
    mock_provider.generate.return_value = "This fact is made up [doc_999]."
    
    chunks = [
        {"chunk_id": "doc_1_1_0", "text": "Context A", "reranker_score": 0.9, "filename": "doc.txt", "page_number": 1},
    ]
    
    answer, citations, conf = generate_answer("What are the facts?", chunks)
    
    assert "This fact is made up" in answer
    # Citation should be verified and rejected
    assert len(citations) == 0
