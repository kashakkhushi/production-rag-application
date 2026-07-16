import os
import asyncio
from typing import Optional
from sqlalchemy import update
from app.core.database import AsyncSessionLocal
from app.models.domain import DocumentDB
from app.services.document_processor import extract_and_chunk
from app.rag.embeddings import embedding_service
from app.rag.vector_store import vector_store
from app.rag.sparse_index import sparse_index
from app.core.config import settings
from loguru import logger

async def process_document_background(doc_id: str, file_path: str, filename: str):
    try:
        logger.info(f"Starting background processing for {filename} ({doc_id})")
        # 1. Extract and chunk
        chunks = extract_and_chunk(
            file_path=file_path, 
            filename=filename, 
            doc_id=doc_id, 
            chunk_size=settings.CHUNK_SIZE, 
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        if not chunks:
            raise ValueError("No text extracted from document.")
            
        # 2. Encode
        texts = [c["text"] for c in chunks]
        embeddings = embedding_service.encode(texts)
        
        # 3. Store in Chroma
        vector_store.add_chunks(chunks, embeddings)
        
        # 4. Rebuild BM25 index asynchronously
        await sparse_index.rebuild_index()
        
        # 5. Update DB status
        async with AsyncSessionLocal() as session:
            stmt = update(DocumentDB).where(DocumentDB.id == doc_id).values(
                status="indexed", 
                chunk_count=len(chunks)
            )
            await session.execute(stmt)
            await session.commit()
            
        logger.info(f"Successfully processed {filename} ({doc_id}) into {len(chunks)} chunks.")
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        async with AsyncSessionLocal() as session:
            stmt = update(DocumentDB).where(DocumentDB.id == doc_id).values(
                status="failed", 
                error_message=str(e)
            )
            await session.execute(stmt)
            await session.commit()
    finally:
        # Cleanup temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

async def delete_document(doc_id: str):
    # Remove from Chroma
    vector_store.delete_by_document_id(doc_id)
    
    # Remove from DB
    async with AsyncSessionLocal() as session:
        doc = await session.get(DocumentDB, doc_id)
        if doc:
            await session.delete(doc)
            await session.commit()
            
    # Rebuild BM25
    await sparse_index.rebuild_index()
