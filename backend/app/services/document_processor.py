import os
import fitz  # PyMuPDF
from docx import Document
from typing import List, Dict, Any
from app.rag.chunking import recursive_character_chunking

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    pages_text = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages_text.append({
                "page_number": page_num + 1,
                "text": text
            })
    return pages_text

def process_docx(file_path: str) -> List[Dict[str, Any]]:
    pages_text = []
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    if text.strip():
        # DOCX doesn't have native fixed page numbers easily accessible via python-docx
        # We assign it to page 1 for the whole document
        pages_text.append({
            "page_number": 1,
            "text": text
        })
    return pages_text

def process_txt(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    if text.strip():
        return [{
            "page_number": 1,
            "text": text
        }]
    return []

def extract_and_chunk(file_path: str, filename: str, doc_id: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".pdf":
        pages = process_pdf(file_path)
    elif ext == ".docx":
        pages = process_docx(file_path)
    elif ext == ".txt":
        pages = process_txt(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
        
    all_chunks = []
    chunk_index = 0
    
    for page in pages:
        page_text = page["text"]
        page_num = page["page_number"]
        
        text_chunks = recursive_character_chunking(page_text, chunk_size, chunk_overlap)
        
        for text_chunk in text_chunks:
            if not text_chunk.strip():
                continue
                
            chunk_id = f"{doc_id}_{page_num}_{chunk_index}"
            
            all_chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "filename": filename,
                "page_number": page_num,
                "chunk_index": chunk_index,
                "text": text_chunk
            })
            chunk_index += 1
            
    return all_chunks
