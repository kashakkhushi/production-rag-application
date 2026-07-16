import re
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from google import genai
from google.genai import types
from app.core.config import settings
from app.models.schemas import Citation
from loguru import logger

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_instruction: str) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        
    def generate(self, prompt: str, system_instruction: str) -> str:
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            )
        )
        return response.text

class MockProvider(LLMProvider):
    def generate(self, prompt: str, system_instruction: str) -> str:
        return "LLM provider not configured"

# Initialize provider
if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip() != "your_gemini_api_key_here":
    try:
        llm_provider: LLMProvider = GeminiProvider(api_key=settings.GEMINI_API_KEY)
        logger.info("Initialized Gemini LLM Provider.")
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {e}. Falling back to MockProvider.")
        llm_provider = MockProvider()
else:
    logger.info("No valid GEMINI_API_KEY found. Initializing MockProvider.")
    llm_provider = MockProvider()


SYSTEM_INSTRUCTION = """You are a precise, enterprise knowledge assistant. 
Your task is to answer the user's query strictly based on the provided context chunks.

RULES:
1. ONLY use the provided context to answer the query. Do NOT use any external knowledge.
2. If the answer is not contained in the provided context, you MUST reply with exactly: "I could not find sufficient information in the uploaded documents."
3. For EVERY claim you make, you MUST cite the source chunk using its ID in brackets, e.g., [doc_1_0_0].
4. Your response must be in Markdown format.

CONTEXT PROVIDED BELOW:
"""

def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> Tuple[str, List[Citation], str]:
    """Generates an answer based strictly on the provided context chunks.

    Uses the configured LLMProvider to generate a response. Enforces a strict
    hallucination prevention threshold by checking the maximum reranker score.
    Extracts and validates inline citations from the LLM's markdown response.

    Args:
        query (str): The user's input question.
        chunks (List[Dict[str, Any]]): The retrieved and reranked context chunks.

    Returns:
        Tuple[str, List[Citation], str]: 
            - The generated markdown answer.
            - A list of validated Citation objects.
            - The confidence level ('high', 'medium', 'low').
    """
    if not chunks:
        return "I could not find sufficient information in the uploaded documents.", [], "low"
        
    # Check max confidence from reranker to enforce threshold
    max_score = max([chunk.get("reranker_score", 0.0) for chunk in chunks])
    if max_score < settings.MIN_RELEVANCE_THRESHOLD:
        return "I could not find sufficient information in the uploaded documents.", [], "low"

    context_str = ""
    chunk_map = {}
    for idx, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        chunk_map[chunk_id] = chunk
        context_str += f"--- Chunk ID: {chunk_id} ---\n{chunk['text']}\n\n"

    prompt = f"USER QUERY: {query}\n\n{context_str}"

    try:
        answer = llm_provider.generate(prompt, SYSTEM_INSTRUCTION)
        
        if answer == "LLM provider not configured":
            # Just return the dummy answer and dummy citations for the chunks retrieved
            citations = []
            for cid, c in chunk_map.items():
                citations.append(Citation(
                    document_name=c["filename"],
                    page_number=c["page_number"],
                    chunk_id=cid,
                    score=c.get("reranker_score", 0.0)
                ))
            return answer, citations, "high"
        
        # Verify citations programmatically
        citations = []
        cited_ids = re.findall(r'\[([^\]]+)\]', answer)
        cited_ids = list(set(cited_ids)) # dedup
        
        for cid in cited_ids:
            if cid in chunk_map:
                c = chunk_map[cid]
                citations.append(Citation(
                    document_name=c["filename"],
                    page_number=c["page_number"],
                    chunk_id=cid,
                    score=c.get("reranker_score", 0.0)
                ))
            else:
                logger.warning(f"Hallucinated citation found and ignored: {cid}")
                
        confidence = "high" if max_score > 0.8 else ("medium" if max_score > 0.5 else "low")
        
        return answer, citations, confidence
        
    except Exception as e:
        logger.error(f"LLM API Error: {str(e)}")
        raise e
