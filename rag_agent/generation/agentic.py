"""
Boucle de retrieval agentique : chercher, juger, chercher encore.

Un pipeline RAG classique fait une seule recherche, puis répond avec ce
qu'elle a trouvé, même si ce n'est pas suffisant. Ici, après chaque
recherche, le LLM local juge si les passages réunis suffisent à répondre ;
sinon il propose une nouvelle requête pour l'information manquante, et la
recherche recommence, au plus `max_iterations` fois au total.

La décision est prise avant de générer la réponse, pas après : juger le
contexte coûte quelques tokens, générer une réponse pour la jeter en coûte
des centaines. La réponse n'est générée qu'une fois, sur le contexte final.
"""

import re
from dataclasses import dataclass
from typing import Protocol

from rag_agent.generation.prompt import format_context
from rag_agent.generation.truncation import shrink_to_budget
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)

SUFFICIENCY_PROMPT = """You decide whether the passages below are enough to answer the question.

Passages:
{context}

Question: {question}
Search queries already tried:
{tried}

If the passages contain the information needed to answer, reply with one word: ENOUGH.
Otherwise reply with SEARCH: followed by one short search query for the missing information,
different from the queries already tried."""

_ENOUGH = re.compile(r"\bENOUGH\b", re.IGNORECASE)
_SEARCH = re.compile(r"\bSEARCH\s*:\s*[\"'«]?(.+?)[\"'»]?\s*$", re.IGNORECASE | re.MULTILINE)


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class Decision:
    """
    enough : le contexte suffit (ou la réponse du modèle est illisible :
             on arrête plutôt que de boucler sur du bruit)
    query  : prochaine requête quand le contexte ne suffit pas
    reason : réponse brute du modèle, pour le journal
    """

    enough: bool
    query: str | None
    reason: str


@dataclass(frozen=True)
class RetrievalStep:
    """Une itération de la boucle, pour l'observabilité"""

    query: str
    sources: list[str]
    n_chunks: int
    decision: str  # "enough", "search", "unparsed" ou "max_iterations"


def parse_decision(text: str, tried: list[str]) -> Decision:
    """
    ENOUGH, ou SEARCH: <requête>. Une requête déjà essayée vaut ENOUGH :
    la rejouer ramènerait les mêmes passages.
    """
    reason = " ".join(text.split())[:300]
    search = _SEARCH.search(text)
    if search:
        query = search.group(1).strip()
        if query and query.lower() not in {t.lower() for t in tried}:
            return Decision(enough=False, query=query, reason=reason)
        return Decision(enough=True, query=None, reason=reason)
    return Decision(enough=True, query=None, reason=reason)


def is_parsed(text: str) -> bool:
    return bool(_ENOUGH.search(text) or _SEARCH.search(text))


class SufficiencyJudge:
    def __init__(self, llm: TextGenerator, max_context_chars: int = 6000):
        self.llm = llm
        # Contexte tronqué pour tenir dans la fenêtre du modèle : au budget de
        # tokens du LLM s'il l'expose, sinon à `max_context_chars` caractères
        self.max_context_chars = max_context_chars

    def _prompt(self, question: str, chunks: list[dict], tried: list[str]) -> str:
        def build(kept: list[dict]) -> str:
            return SUFFICIENCY_PROMPT.format(
                context=format_context(kept)[: self.max_context_chars],
                question=question,
                tried="\n".join(f"- {q}" for q in tried),
            )

        count_tokens = getattr(self.llm, "count_tokens", None)
        budget = getattr(self.llm, "prompt_budget", None)
        if count_tokens is None or budget is None:
            return build(chunks)
        return shrink_to_budget(build, chunks, count_tokens, budget)

    def decide(self, question: str, chunks: list[dict], tried: list[str]) -> tuple[Decision, bool]:
        """
        (décision, réponse du modèle lisible). Si le modèle échoue, la boucle
        s'arrête sur le contexte déjà réuni : le jugement ne doit jamais
        empêcher de répondre.
        """
        try:
            reply = self.llm.generate(self._prompt(question, chunks, tried))
        except Exception as e:
            logger.warning("Jugement du contexte impossible, la boucle s'arrête : %s", e)
            return Decision(enough=True, query=None, reason=f"error: {e}"), False
        return parse_decision(reply, tried), is_parsed(reply)
