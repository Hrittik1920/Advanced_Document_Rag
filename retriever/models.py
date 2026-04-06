# retriever/models.py
from dataclasses import dataclass, field

@dataclass
class _Document:
    page_content: str
    metadata: dict = field(default_factory=dict)