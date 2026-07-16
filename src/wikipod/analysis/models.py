from pydantic import BaseModel

class Article(BaseModel):
    article_id: int
    title: str
    html: str

class Section(BaseModel):
    article_id: int
    article_title: str
    section_title: str
    text: str

class ArticleMetadata(BaseModel):
    article_id: int
    title: str

    word_count: int
    
    link_count: int
    links: list[str]
    
    section_count: int
    sections: list[Section]
    
    categories: list[str]