"""
Construction du prompt RAG.

Le template vit ici, seul : on peut le faire évoluer sans toucher au
retrieval ni à la génération. `build_prompt` ne fait que du formatage —
le choix des chunks (top_k, troncature) est fait en amont.
"""

PROMPT_TEMPLATE = """Tu es un assistant précis et factuel.
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.
{history}
CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""

HISTORY_TEMPLATE = """
HISTORIQUE DE LA CONVERSATION :
{turns}
"""

CHUNK_TEMPLATE = "[Source: {source}]\n{text}"
CHUNK_SEPARATOR = "\n\n---\n\n"
EMPTY_CONTEXT = "(aucun document pertinent)"
ROLE_LABELS = {"user": "Utilisateur", "assistant": "Assistant"}


def format_context(chunks: list[dict]) -> str:
    """
    Concatène les chunks dans l'ordre reçu (le plus pertinent d'abord),
    chacun précédé de sa source pour que le modèle puisse la citer
    """
    if not chunks:
        return EMPTY_CONTEXT
    return CHUNK_SEPARATOR.join(CHUNK_TEMPLATE.format(source=c["source"], text=c["text"]) for c in chunks)


def format_history(history: list[dict]) -> str:
    """
    Rend les échanges précédents ({"role", "content"}), du plus ancien au
    plus récent ; chaîne vide s'il n'y en a pas
    """
    if not history:
        return ""
    turns = "\n".join(f"{ROLE_LABELS.get(turn['role'], turn['role'])} : {turn['content']}" for turn in history)
    return HISTORY_TEMPLATE.format(turns=turns)


def build_prompt(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    """
    Assemble le prompt final : consignes, historique récent éventuel,
    contexte récupéré et question
    """
    return PROMPT_TEMPLATE.format(
        history=format_history(history or []),
        context=format_context(chunks),
        question=question,
    )
