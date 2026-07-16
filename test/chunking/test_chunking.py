from wikipod.analysis.models import Section
from wikipod.chunking.chunker import chunk_section, chunk_article
from wikipod.chunking.models import Chunk
from wikipod.analysis.models import ArticleMetadata

def test_small_section_returns_one_chunk():

    section = Section(
        article_id=1,
        article_title="Climate change",
        section_title="Introduction",
        text=(
            "Climate change refers to long-term shifts in temperatures and weather "
            "patterns. These shifts may be natural, but since the 1800s human "
            "activities have been the main driver."
        )
    )
    chunks = chunk_section(section)
    
    assert len(chunks) == 1
    
    
def test_large_section_is_split_into_multiple_chunks():
    section = Section(
    article_id=1,
    article_title="Test",
    section_title="Large Section",
    text=" ".join(f"word{i}" for i in range(500))
    )
    
    chunks = chunk_section(section, max_words=200, overlap=10)
    
    assert len(chunks) == 3
    
def test_overlap_is_applied_correctly():
    section = Section(
    article_id=1,
    article_title="Test",
    section_title="Large Section",
    text=" ".join(f"word{i}" for i in range(500))
    )
    
    chunks = chunk_section(section, max_words=50, overlap=10)
    
    assert chunks[0].text.split()[-2:] == ["word48", "word49"]
    assert chunks[1].text.split()[:2] == ["word40", "word41"]
    

def test_chunk_article_returns_flat_chunk_list():
    section1 = Section(
    article_id=1,
    article_title="Climate",
    section_title="Intro",
    text=" ".join(f"a{i}" for i in range(250))
    )

    section2 = Section(
        article_id=1,
        article_title="Climate",
        section_title="History",
        text=" ".join(f"b{i}" for i in range(250))
    )

    article = ArticleMetadata(
        article_id=1,
        title="Climate",
        word_count=500,
        link_count=0,
        links=[],
        section_count=2,
        sections=[section1, section2],
        categories=[]
    )
    
    chunks = chunk_article(article)

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    assert not any(isinstance(item, list) for item in chunks)