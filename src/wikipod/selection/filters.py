def keep(article):
    return (
        article["word_count"] > 100 and
        article["link_count"] > 3
    )