"""Chunk model -- the unit that ultimately gets embedded and indexed."""
from pydantic import BaseModel


class Chunk(BaseModel):
    article_id: int
    article_title: str

    section_title: str
    chunk_index: int

    word_count: int
    text: str

    @property
    def chunk_id(self) -> str:
        """Stable identifier used as the OpenSearch document id."""
        return f"{self.article_id}-{self.section_title}-{self.chunk_index}"
