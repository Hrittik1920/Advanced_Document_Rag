from .vector import retriever
from .knowledge_graph import GraphRetriever
from dataclasses import dataclass, field 
@dataclass
class _Document:
    page_content: str
    metadata: dict = field(default_factory=dict)