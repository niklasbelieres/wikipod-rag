from dataclasses import dataclass


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    query_type: str
    is_out_of_scope: bool = False

class QueryAnalyzer:
    def __init__(
        self,
        corrections: dict[str, str] | None = None,
        out_of_scope_queries: set[str] | None = None,
    ):
        self.corrections = corrections if corrections is not None else {
            "Geogre": "George",
            "Cathlic Chuch": "Catholic Church",
        }
        self.out_of_scope_queries = out_of_scope_queries or set()

    def analyze(self, query: str) -> QueryAnalysis:
        normalized_query = " ".join(query.split())
        query_type = "normal"

        for typo, correction in self.corrections.items():
            if typo in normalized_query:
                normalized_query = normalized_query.replace(typo, correction)
                query_type = "typo"

        is_out_of_scope = normalized_query in self.out_of_scope_queries

        if is_out_of_scope:
            query_type = "out_of_scope"

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            query_type=query_type,
            is_out_of_scope=is_out_of_scope,
        )