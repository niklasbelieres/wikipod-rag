from metrics import (
    article_size,
    article_connectivity
)

def hybrid_score(article):
    return (
        article_size(article) * 0.4 +
        article_connectivity(article) * 0.6
    )

def size_score(article):
    return article_size(article)

def connectivity_score(article):
    return article_connectivity(article)