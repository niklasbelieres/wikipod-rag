from pydantic import BaseModel

class Article(BaseModel):
    article_id: int
    title: str
    html: str

class ArticleMetadata(BaseModel):
    article_id: int
    title: str

    word_count: int
    link_count: int
    
    categories: list[str]