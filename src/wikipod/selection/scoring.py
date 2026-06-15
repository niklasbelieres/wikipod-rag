from wikipod.analysis.models import ArticleMetadata
import math

def score_article(article: ArticleMetadata, link_frequencies: dict[str, int]) -> float:
    
    word_score = math.log1p(article.word_count)

    link_score = math.log1p(article.link_count)

    importance_score = sum(
        math.log1p(link_frequencies.get(link, 0))for link in article.links
    )

    return (
        0.3 * word_score +
        0.2 * link_score +
        0.5 * importance_score
    )
    
