from wikipod.analysis.models import ArticleMetadata
from wikipod.selection.scoring import score_article

def select_top_articles(
    articles,
    link_frequencies,
    limit=100
):
    return sorted(
        articles,
        key=lambda a: score_article(
            a,
            link_frequencies
        ),
        reverse=True
    )[:limit]