"""Research Assistant Agent: bounded, evidence-grounded, auditable."""
from .agent import AgentConfig, ResearchAgent
from .schemas import Finding, ResearchResult
from .sources import CompositeProvider, LocalCorpusProvider, WebProvider

__all__ = ["ResearchAgent", "AgentConfig", "ResearchResult", "Finding",
           "LocalCorpusProvider", "CompositeProvider", "WebProvider"]
