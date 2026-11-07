"""
Réécriture de la question en requêtes de recherche, avant le retrieval.

Une question de conversation (« et pour les vélos, ça coûte combien ? »)
est rarement la meilleure requête pour une recherche vectorielle : le LLM
local la reformule en requêtes courtes et autonomes, et découpe une
question en plusieurs parties en une requête par partie. La question
d'origine reste toujours la première requête : une réécriture ratée ne
peut qu'ajouter des passages, jamais faire perdre ceux qu'elle trouvait.
"""

import re
from typing import Protocol

from rag_agent.logging_config import get_logger

logger = get_logger(__name__)

REWRITE_PROMPT = """Rewrite the question below into search queries for a document search engine.
- Write one query per line, at most {n} lines, and nothing else.
- If the question asks several things, write one query per part.
- Keep names, numbers and technical terms; drop politeness and filler words.
- Write the queries in the language of the question.
- Each query must make sense on its own: resolve "it", "they" or "that one" from the conversation.
{history}
Question: {question}
Queries:"""

HISTORY_BLOCK = """
Conversation so far:
{turns}
"""

# Puces, numérotation et guillemets qu'un modèle ajoute volontiers
_DECORATION = re.compile(r"^\s*(?:[-*•]|\d+[.)]|query\s*\d*\s*:)?\s*[\"'«]?\s*|\s*[\"'»]?\s*$", re.IGNORECASE)


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


def parse_queries(text: str, limit: int) -> list[str]:
    """
    Requêtes de la réponse du modèle : une par ligne, sans décoration ni
    doublon (à la casse près), au plus `limit`. Une ligne qui se termine par
    « : » est un intitulé, pas une requête.
    """
    queries: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        query = _DECORATION.sub("", line).strip()
        if not query or query.endswith(":") or query.lower() in seen:
            continue
        seen.add(query.lower())
        queries.append(query)
        if len(queries) == limit:
            break
    return queries


class QueryRewriter:
    def __init__(self, llm: TextGenerator, max_queries: int = 3):
        """
        max_queries : nombre maximal de requêtes réécrites, en plus de la
        question d'origine
        """
        if max_queries < 1:
            raise ValueError(f"max_queries must be >= 1, got {max_queries}")
        self.llm = llm
        self.max_queries = max_queries

    def rewrite(self, question: str, history: list[dict] | None = None) -> list[str]:
        """
        [question d'origine, requêtes réécrites...]. L'historique permet de
        rendre autonome une question de suite (« et pour les vélos ? »). Si
        le modèle échoue, la question seule : la réécriture ne doit jamais
        empêcher de répondre.
        """
        turns = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history or [])
        prompt = REWRITE_PROMPT.format(
            n=self.max_queries, question=question, history=HISTORY_BLOCK.format(turns=turns) if turns else ""
        )
        try:
            reply = self.llm.generate(prompt)
        except Exception as e:
            logger.warning("Réécriture de la requête impossible, question d'origine seule : %s", e)
            return [question]

        rewritten = [q for q in parse_queries(reply, self.max_queries + 1) if q.lower() != question.lower()]
        queries = [question] + rewritten[: self.max_queries]
        logger.info("Requêtes de recherche : %s", queries)
        return queries
