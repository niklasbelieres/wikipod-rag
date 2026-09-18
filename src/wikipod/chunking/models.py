"""Chunk model: the unit that ultimately gets embedded and indexed."""
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    article_id: int
    article_title: str

    section_title: str
    section_index: int = Field(default=0, ge=0)
    chunk_index: int

    word_count: int
    text: str

    @property
    def chunk_id(self) -> str:
        """Identify a chunk by article, section position and position within the section."""
        return f"{self.article_id}-{self.section_index}-{self.chunk_index}"
