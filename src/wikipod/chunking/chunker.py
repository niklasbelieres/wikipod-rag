from wikipod.analysis.models import Section
from wikipod.chunking.models import Chunk
from wikipod.analysis.models import ArticleMetadata

def chunk_section(section: Section, max_words: int=250, overlap:int=40) -> list[Chunk]:
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    
    words = section.text.split()
    if len(words) <= max_words:
        return [
            Chunk(article_id=section.article_id,
                  article_title=section.article_title,
                  section_title=section.section_title,
                  chunk_index=0,
                  word_count=len(words),
                  text=" ".join(words)
            )
        ]
        
    chunks = []
    step = max_words - overlap
    start = 0
    chunk_index = 0
    
    while start < len(words):
        chunk_words = words[start:start + max_words]
        start += step
        chunk = Chunk(article_id=section.article_id,
                  article_title=section.article_title,
                  section_title=section.section_title,
                  chunk_index=chunk_index,
                  word_count=len(chunk_words),
                  text=" ".join(chunk_words)
        )
        
        chunks.append(chunk)
        chunk_index += 1
    
    return chunks
        
    

def chunk_article(article: ArticleMetadata) -> list[Chunk]:
    chunks = []
    for section in article.sections:
        section_chunks = chunk_section(
            section=section
        )
        chunks.extend(section_chunks)
    
    return chunks