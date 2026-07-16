import re
from typing import List, Dict, Any

def recursive_character_chunking(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits text recursively based on character limits.
    Separators are applied in priority order to keep semantic units together.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(text_to_split: str, sep_index: int) -> List[str]:
        if len(text_to_split) <= chunk_size:
            return [text_to_split]
            
        separator = separators[sep_index]
        if separator:
            splits = text_to_split.split(separator)
        else:
            splits = list(text_to_split)
            
        chunks = []
        current_chunk = []
        current_len = 0
        
        for split in splits:
            split_len = len(split) + (len(separator) if separator else 0)
            
            if current_len + split_len > chunk_size and current_chunk:
                chunk_str = separator.join(current_chunk)
                chunks.append(chunk_str)
                
                # Handling overlap
                overlap_chars = 0
                overlap_chunk = []
                for prev_split in reversed(current_chunk):
                    prev_len = len(prev_split) + (len(separator) if separator else 0)
                    if overlap_chars + prev_len > chunk_overlap:
                        break
                    overlap_chunk.insert(0, prev_split)
                    overlap_chars += prev_len
                    
                current_chunk = overlap_chunk
                current_len = overlap_chars
                
            current_chunk.append(split)
            current_len += split_len
            
        if current_chunk:
            chunks.append(separator.join(current_chunk))
            
        return chunks
        
    return split_text(text, 0)
