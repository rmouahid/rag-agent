"""
Ajustement du prompt à la fenêtre de contexte du modèle.

Quand le prompt assemblé dépasse le budget, on retire dans l'ordre :
1. les chunks les moins pertinents (score le plus bas), jusqu'à n'en
   garder qu'un — le meilleur ;
2. les échanges les plus anciens de l'historique.
Si le prompt ne tient toujours pas, `ContextOverflowError` est levée :
retirer le dernier chunk donnerait une réponse sans ancrage.
"""

from typing import Callable

from rag_agent.generation.prompt import build_prompt
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class ContextOverflowError(ValueError):
    """Le prompt minimal (question + meilleur chunk) dépasse la fenêtre de contexte."""


def _least_relevant(chunks: list[dict]) -> int:
    """
    Index du chunk le moins pertinent ; sans score, l'ordre de retrieval
    fait foi et c'est le dernier
    """
    if all("score" in c for c in chunks):
        return min(range(len(chunks)), key=lambda i: (chunks[i]["score"], -i))
    return len(chunks) - 1


def fit_prompt(
    question: str,
    chunks: list[dict],
    history: list[dict],
    count_tokens: Callable[[str], int],
    budget: int,
) -> tuple[str, list[dict], list[dict]]:
    """
    Construit le plus grand prompt tenant dans `budget` tokens.

    Retourne (prompt, chunks conservés, historique conservé) ; les chunks
    conservés gardent leur ordre d'origine.
    """
    chunks, history = list(chunks), list(history)

    while True:
        prompt = build_prompt(question, chunks, history=history)
        size = count_tokens(prompt)
        if size <= budget:
            return prompt, chunks, history

        if len(chunks) > 1:
            dropped = chunks.pop(_least_relevant(chunks))
            logger.info("Prompt trop long (%d > %d tokens) : chunk retiré (%s)", size, budget, dropped["source"])
        elif history:
            # Un échange = question + réponse
            history = history[2:]
            logger.info("Prompt trop long (%d > %d tokens) : échange le plus ancien retiré", size, budget)
        else:
            raise ContextOverflowError(
                f"The prompt needs {size} tokens but only {budget} are available: "
                "shorten the question or increase LLM_CONTEXT_SIZE."
            )
