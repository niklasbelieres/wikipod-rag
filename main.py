# Click commands kommen hier rein und verweisen auf die module 
from src.wikipod.analysis.reader import iter_articles
from src.wikipod.chunking.extractor import extract_arcticle_text

ZIM_FILE = "/Users/lucabritten/Documents/1. Studium/4. Semester/6_Projektarbeit/wikipod-rag/test/data/climate-change-mini.zim"

count = 0
for article in iter_articles(ZIM_FILE):
    text = extract_arcticle_text(article.html)
    print(text)
    count += 1
    if count > 1:
        break
    print("\n =======\n")