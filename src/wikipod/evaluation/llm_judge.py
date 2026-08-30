import json
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class JudgeResult:
    relevance_score: int
    reasoning: str





class LLMJudge:
    def __init__(
            self,
            backend: Callable[[str, str], str] | None = None,
            generator=None,
    ):
        self.backend = backend
        self.generator = generator

    def judge(self, query: str, retrieved_text: str) -> JudgeResult:
        if self.backend is not None:
            response = self.backend(query, retrieved_text)
        elif self.generator is not None:
            messages = [
                {
                    "role": "user",
                    "content": (
    "Evaluate whether the retrieved text is a good retrieval result "
    "for the query.\n"
    "Judge the text as a retrieval result, not merely whether it "
    "mentions the query topic.\n\n"
    "Use this relevance scale:\n"
    "0 = irrelevant: does not answer the query\n"
    "1 = weakly relevant: only mentions the topic or has a minor connection\n"
    "2 = relevant: contains useful information that helps answer the query\n"
    "3 = directly relevant: the text is primarily about the exact entity "
    "or topic requested and directly answers the query\n\n"
    "Important rules:\n"
    "- A mere mention of the requested entity is not enough for score 2 or 3.\n"
    "- If the text is primarily about a different person, entity, or topic, "
    "it must not receive score 3.\n"
    "- Distinguish similarly named entities carefully, such as "
    "George III and George IV.\n\n"
    f"Query: {query}\n"
    f"Retrieved text: {retrieved_text}\n\n"
    "Return JSON with relevance_score and reasoning."
),
                }
            ]
            response = self.generator.generate(messages)
        else:
            return JudgeResult(
                relevance_score=0,
                reasoning="Not evaluated yet.",
            )

        response = response.strip()

        if response.startswith("```json"):
            response = response.removeprefix("```json").strip()

        if response.endswith("```"):
            response = response.removesuffix("```").strip()
        data = json.loads(response)
        score = data["relevance_score"]

        if score not in {0, 1, 2, 3}:
            raise ValueError("relevance_score must be between 0 and 3")

        return JudgeResult(
            relevance_score=score,
            reasoning=data["reasoning"],
        )



