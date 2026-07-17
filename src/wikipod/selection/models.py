"""Result model for the selection step."""

from __future__ import annotations

from pydantic import BaseModel

from wikipod.analysis.models import ArticleMetadata


class SelectionResult(BaseModel):
    """The chosen subset plus bookkeeping useful for the evaluation report."""

    selected: list[ArticleMetadata]
    budget_mb: float
    used_mb: float
    total_candidates: int

    @property
    def utilization(self) -> float:
        """Fraction of the storage budget actually used (0-1)."""
        return self.used_mb / self.budget_mb if self.budget_mb else 0.0
