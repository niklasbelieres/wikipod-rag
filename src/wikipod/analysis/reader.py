from collections.abc import Iterator
from libzim.reader import Archive
from .models import Article

def iter_articles(zim_path: str) -> Iterator[Article]:
    archive = Archive(zim_path)
    
    for article_id in range(archive.article_count):
        try:
            entry = archive._get_entry_by_id(article_id)
            
            if entry.is_redirect:
                continue
            
            item = entry.get_item()
            
            html = item.content.tobytes().decode(
                "utf-8", 
                errors="ignore"
            )
            
            if is_html_redirect(html):
                continue
            
            yield Article(
                article_id=article_id,
                title=entry.title,
                html=html,
            )
        except Exception as ex:
            print(f"Skipping article {article_id}: {ex}")

def is_html_redirect(html: str) -> bool:
    return (
        'http-equiv="refresh"' in html
        and "URL=" in html
    )