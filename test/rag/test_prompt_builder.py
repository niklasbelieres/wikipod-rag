from wikipod.rag.prompt_builder import (
    DEFAULT_MAX_CONTEXT_WORDS,
    SYSTEM_PROMPT,
    _fit_to_budget,
    build_messages,
    format_context,
)
from wikipod.chunking.chunker import Chunk

def test_one_chunk_leads_to_one_context_result():
    article_id = 1
    article_title = "Climate Change"
    section_title = "Introduction"
    text = "The Climate change is a thread for the human civilization."
    
    chunk = Chunk(
        article_id=article_id,
        article_title=article_title,
        section_title=section_title,
        chunk_index=0,
        word_count=len(text.split()),
        text=text,
    )
    
    result = format_context([chunk])
    assert result == f"[1] ({article_title} - {section_title}) {text}"

def test_multiple_chunks_are_formatted_as_expected():
    article_id1 = 1
    article_title1 = "Climate Change"
    section_title1 = "Introduction"
    text1 = "The Climate change is a thread for the human civilization."
    
    chunk1 = Chunk(
        article_id=article_id1,
        article_title=article_title1,
        section_title=section_title1,
        chunk_index=0,
        word_count=len(text1.split()),
        text=text1,
    )
    
    article_id2 = 2
    article_title2 = "Climate Change"
    section_title2 = "Summary"
    text2 = "The Climate change is a real problem."
    
    chunk2 = Chunk(
        article_id=article_id2,
        article_title=article_title2,
        section_title=section_title2,
        chunk_index=1,
        word_count=len(text2.split()),
        text=text2,
    )
    
    result = format_context([chunk1, chunk2])
    assert len(result) > 0
    assert result == (
    f"[1] ({chunk1.article_title} - {chunk1.section_title}) {chunk1.text}\n\n"
    f"[2] ({chunk2.article_title} - {chunk2.section_title}) {chunk2.text}"
    )
    
    lines = result.split("\n\n")
    assert len(lines) == 2
    
   
def test_format_context_handles_empty_list():
    result = format_context([])
    
    assert result == ""
    
# _fit_to_budget
def test_fit_to_budget_keeps_all_chunks_when_everything_fits():
    article_id = 1
    article_title = "Climate Change"
    section_title = "Introduction"
    text = "The Climate change is a thread for the human civilization."
    
    chunk = Chunk(
        article_id=article_id,
        article_title=article_title,
        section_title=section_title,
        chunk_index=0,
        word_count=len(text.split()),
        text=text,
    )
    
    trimmed = _fit_to_budget([chunk], chunk.word_count + 10)
    assert trimmed[0] == chunk
    assert trimmed[0].word_count == chunk.word_count


def test_fit_to_budget_drops_chunks_that_exceed_the_budget():
    article_id1 = 1
    article_title1 = "Climate Change"
    section_title1 = "Introduction"
    text1 = "The Climate change is a thread for the human civilization."
    
    chunk1 = Chunk(
        article_id=article_id1,
        article_title=article_title1,
        section_title=section_title1,
        chunk_index=0,
        word_count=len(text1.split()),
        text=text1,
    )
    
    article_id2 = 2
    article_title2 = "Climate Change"
    section_title2 = "Summary"
    text2 = "The Climate change is a real problem."
    
    chunk2 = Chunk(
        article_id=article_id2,
        article_title=article_title2,
        section_title=section_title2,
        chunk_index=1,
        word_count=len(text2.split()),
        text=text2,
    )
    assert chunk1.word_count > chunk2.word_count
    
    trimmed = _fit_to_budget([chunk1, chunk2], chunk2.word_count + 1)
    assert len(trimmed) == 1
    assert trimmed[0] == chunk2


def test_fit_to_budget_skips_oversized_chunk_but_keeps_smaller_one_after_it():
    # A=200 Wörter, B=150, C=50, Budget=250 -> Ergebnis muss [A, C] sein, nicht [A]
    article_id1 = 1
    article_title1 = "Climate Change"
    section_title1 = "Introduction"
    text1 = "The Climate change is a thread for the human civilization."
    
    chunk1 = Chunk(
        article_id=article_id1,
        article_title=article_title1,
        section_title=section_title1,
        chunk_index=0,
        word_count=len(text1.split()),
        text=text1,
    )
    
    article_id2 = 2
    article_title2 = "Climate Change"
    section_title2 = "Summary"
    text2 = "The Climate change is a real problem. This text is longer than the next chunk!"
    
    chunk2 = Chunk(
        article_id=article_id2,
        article_title=article_title2,
        section_title=section_title2,
        chunk_index=1,
        word_count=len(text2.split()),
        text=text2,
    )
    
    article_id3 = 3
    article_title3 = "Climate Change"
    section_title3 = "Evaluation"
    text3 = "The Climate change is a real problem."
    
    chunk3 = Chunk(
        article_id=article_id3,
        article_title=article_title3,
        section_title=section_title3,
        chunk_index=2,
        word_count=len(text3.split()),
        text=text3,
    )
    
    budget = chunk1.word_count + chunk3.word_count
    assert chunk2.word_count > chunk3.word_count
    
    trimmed = _fit_to_budget([chunk1, chunk2, chunk3], max_context_words=budget)
    
    assert len(trimmed) == 2
    assert trimmed[0] == chunk1
    assert trimmed[1] == chunk3
    


def test_fit_to_budget_handles_empty_chunk_list():
    trimmed = _fit_to_budget([], 100)
    assert len(trimmed) == 0


def test_fit_to_budget_with_zero_budget_returns_empty_list():
    article_id1 = 1
    article_title1 = "Climate Change"
    section_title1 = "Introduction"
    text1 = "The Climate change is a thread for the human civilization."
    
    chunk1 = Chunk(
        article_id=article_id1,
        article_title=article_title1,
        section_title=section_title1,
        chunk_index=0,
        word_count=len(text1.split()),
        text=text1,
    )
    
    trimmed = _fit_to_budget([chunk1], 0)
    assert len(trimmed) == 0


# build_messages

def test_build_messages_returns_system_and_user_roles_in_order():
    messages = build_messages("What causes climate change?", [])

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_build_messages_user_content_contains_query_and_formatted_context():
    chunk = Chunk(
        article_id=1,
        article_title="Climate Change",
        section_title="Introduction",
        chunk_index=0,
        word_count=10,
        text="The Climate change is a thread for the human civilization.",
    )
    query = "What causes climate change?"

    messages = build_messages(query, [chunk])
    user_content = messages[1]["content"]

    assert f"Question: {query}" in user_content
    assert format_context([chunk]) in user_content


def test_build_messages_with_no_chunks_uses_fallback_context_text():
    messages = build_messages("unrelated query", [])
    user_content = messages[1]["content"]

    assert "No relevant excerpts were found" in user_content
    assert "Question: unrelated query" in user_content


def test_build_messages_respects_max_context_words_truncation():
    # bewusst mehr/größere Chunks reingeben als das Budget erlaubt,
    # dann prüfen dass der Text der aussortierten Chunks NICHT im Ergebnis auftaucht
    small_chunk = Chunk(
        article_id=1,
        article_title="Climate Change",
        section_title="Introduction",
        chunk_index=0,
        word_count=5,
        text="Short excerpt that fits easily.",
    )
    large_chunk = Chunk(
        article_id=2,
        article_title="Climate Change",
        section_title="Details",
        chunk_index=1,
        word_count=100,
        text="This excerpt is deliberately far too long for the tiny budget given in this test.",
    )

    messages = build_messages("query", [small_chunk, large_chunk], max_context_words=10)
    user_content = messages[1]["content"]

    assert small_chunk.text in user_content
    assert large_chunk.text not in user_content


def test_build_messages_uses_default_max_context_words_when_not_given():
    chunks = [
        Chunk(
            article_id=i,
            article_title="Climate Change",
            section_title=f"Section {i}",
            chunk_index=i,
            word_count=DEFAULT_MAX_CONTEXT_WORDS,
            text=f"Filler text number {i}.",
        )
        for i in range(2)
    ]

    messages = build_messages("query", chunks)
    user_content = messages[1]["content"]

    assert chunks[0].text in user_content
    assert chunks[1].text not in user_content