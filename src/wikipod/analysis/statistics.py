from wikipod.analysis.models import ArticleMetadata
from collections import Counter

def summarize_articles(articles: list[ArticleMetadata]):
    if not articles:
        return {}
    
    article_count = len(articles)
    
    word_counts = [a.word_count for a in articles]
    link_counts = [a.link_count for a in articles]
    
    avg_word_count = sum(word_counts) / article_count
    avg_link_count = sum(link_counts) / article_count
    
    max_word_count = max(word_counts)
    max_links_count = max(link_counts)
    
    return {
        "article_count": article_count,
        "avg_word_count": avg_word_count,
        "max_word_count": max_word_count,
        "avg_link_count": avg_link_count,
        "max_link_count": max_links_count
    }

def top_linked_articles(articles: list[ArticleMetadata], limit=20):
    #     [
    #     ("Physics", 4500),
    #     ("Mathematics", 3900)
    # ]
    counter = Counter()
    i = 0
    for article in articles:
        for link in article.links:
            counter.update([link])
        
    return counter.most_common(limit)

