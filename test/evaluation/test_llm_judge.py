from unittest.mock import MagicMock

import pytest

from wikipod.evaluation.llm_judge import JudgeResult, LLMJudge


def test_judge_result_stores_relevance_score():
    result = JudgeResult(
        relevance_score=3,
        reasoning="The retrieved text directly answers the query.",
    )

    assert result.relevance_score == 3
    assert result.reasoning == "The retrieved text directly answers the query."


def test_llm_judge_returns_judge_result():

    judge = LLMJudge()

    result = judge.judge(

        query="Who was George III?",

        retrieved_text="George III was King of Great Britain and Ireland.",

    )

    assert isinstance(result, JudgeResult)


def test_llm_judge_uses_backend_response():
    def fake_backend(query: str, retrieved_text: str) -> str:
        return '{"relevance_score": 3, "reasoning": "Directly relevant."}'

    judge = LLMJudge(backend=fake_backend)

    result = judge.judge(
        query="Who was George III?",
        retrieved_text="George III was King of Great Britain and Ireland.",
    )

    assert result.relevance_score == 3
    assert result.reasoning == "Directly relevant."





def test_llm_judge_rejects_invalid_relevance_score():
    def fake_backend(query: str, retrieved_text: str) -> str:
        return '{"relevance_score": 7, "reasoning": "Invalid score."}'

    judge = LLMJudge(backend=fake_backend)

    with pytest.raises(ValueError):
        judge.judge(
            query="Who was George III?",
            retrieved_text="George III was King of Great Britain.",
        )


def test_llm_judge_rejects_invalid_json():
    def fake_backend(query: str, retrieved_text: str) -> str:
        return "this is not json"

    judge = LLMJudge(backend=fake_backend)

    with pytest.raises(ValueError):
        judge.judge(
            query="Who was George III?",
            retrieved_text="George III was King of Great Britain.",
        )


def test_llm_judge_can_use_generator():
    generator = MagicMock()
    generator.generate.return_value = (
        '{"relevance_score": 3, "reasoning": "Directly relevant."}'
    )

    judge = LLMJudge(generator=generator)

    result = judge.judge(
        query="Who was George III?",
        retrieved_text="George III was King of Great Britain and Ireland.",
    )

    assert result.relevance_score == 3
    generator.generate.assert_called_once()


def test_llm_judge_prompt_contains_query_and_retrieved_text():
    generator = MagicMock()
    generator.generate.return_value = (
        '{"relevance_score": 3, "reasoning": "Directly relevant."}'
    )

    judge = LLMJudge(generator=generator)

    judge.judge(
        query="Who was George III?",
        retrieved_text="George III was King of Great Britain.",
    )

    messages = generator.generate.call_args.args[0]
    prompt = messages[0]["content"]

    assert "Who was George III?" in prompt
    assert "George III was King of Great Britain." in prompt
    assert "relevance_score" in prompt
    assert "reasoning" in prompt



def test_llm_judge_prompt_defines_relevance_scale():
    generator = MagicMock()
    generator.generate.return_value = (
        '{"relevance_score": 3, "reasoning": "Directly relevant."}'
    )

    judge = LLMJudge(generator=generator)

    judge.judge(
        query="Who was George III?",
        retrieved_text="George III was King of Great Britain.",
    )

    messages = generator.generate.call_args.args[0]
    prompt = messages[0]["content"]

    assert "0 = irrelevant" in prompt
    assert "1 = weakly relevant" in prompt
    assert "2 = relevant" in prompt
    assert "3 = directly relevant" in prompt