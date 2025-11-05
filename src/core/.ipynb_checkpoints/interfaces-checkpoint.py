"""
Core interfaces to keep the app portable across backends (GCP, Databricks, etc.).
Replace `pass` blocks with concrete implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 8, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Return a list[dict] of chunks with keys: policy_id, version_date, section, page, text, score, uri?, chunk_id?"""
        raise NotImplementedError

class Generator(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Return dict like {'text': str, 'raw': Any}."""
        raise NotImplementedError

class GroundingChecker(ABC):
    @abstractmethod
    def score(self, answer_text: str, evidences: List[Dict[str, Any]]) -> float:
        """Return a support score 0..1 indicating how well the answer is grounded in evidence."""
        raise NotImplementedError

class DiffClient(ABC):
    @abstractmethod
    def section_diffs(self, policy_id: str, old_version: str, new_version: str, section: str) -> List[Dict[str, str]]:
        """Return diffs for the given section between two versions (e.g., Jan vs Oct)."""
        raise NotImplementedError
