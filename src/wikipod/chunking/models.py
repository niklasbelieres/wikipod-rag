from pydantic import BaseModel

class Chunk(BaseModel):
    article_id: int
    article_title: str
    
    section_title: str
    
    chunk_index: int
    
    word_count: int
    
    text: str
   