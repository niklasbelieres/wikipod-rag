from models import  (Article, ArticleMetadata)

def extract_metadata(article: Article) -> ArticleMetadata:
    # yield Article(
    # article_id=123,
    # title="Albert Einstein",
    # html="<html>...</html>")
    pass

def _count_words(html:str) -> int:
    pass

def _extract_links(html: str) ->list[str]:
    pass

def _extract_categories(html: str) -> list[str]:
    pass
    