from rag_agent.config import config
from rag_agent.generation import answer as outcome
from rag_agent.generation.answer import Answer, Source, StreamingAnswer
from rag_agent.generation.llm import LocalLLM
from rag_agent.generation.truncation import fit_prompt
from rag_agent.ingestion.md_loader import MarkdownLoader
from rag_agent.ingestion.pdf_loader import PDFLoader
from rag_agent.ingestion.web_loader import WebLoader
from rag_agent.logging_config import get_logger
from rag_agent.vectorstore.retriever import Retriever

logger = get_logger(__name__)

# Réponses fixes quand le LLM n'est pas appelé, faute de matière
REFUSALS = {
    outcome.NO_DOCUMENTS: " Aucun document indexé. Ingère d'abord des sources.",
    outcome.NO_RELEVANT_CONTEXT: (
        " Je n'ai trouvé aucun passage assez pertinent dans les documents indexés "
        "pour répondre à cette question."
    ),
}
NO_DOCUMENTS = REFUSALS[outcome.NO_DOCUMENTS]

# Extension de fichier → loader
FILE_TYPES = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown"}


class UnsupportedSourceError(ValueError):
    """Ni une URL http(s), ni un fichier d'un type pris en charge."""


def source_type(source: str) -> str:
    """
    Loader à utiliser pour `source` : "web", "pdf" ou "markdown"
    """
    if source.startswith(("http://", "https://")):
        return "web"
    for extension, kind in FILE_TYPES.items():
        if source.lower().endswith(extension):
            return kind
    raise UnsupportedSourceError(
        f"Unsupported source {source!r}: expected an http(s) URL or a "
        f"{', '.join(FILE_TYPES)} file"
    )


class RAGAgent:
    def __init__(self, llm: LocalLLM = None, history_turns: int = None, min_relevance_score: float = None):
        # Chargé dès la construction : sans poids, l'agent échoue tout de
        # suite avec un message clair plutôt qu'à la première question.
        self.llm       = llm or LocalLLM()
        self.retriever = Retriever()
        # Échanges question/réponse récents rappelés dans le prompt
        self.history_turns = config.history_turns if history_turns is None else history_turns
        self.history       = []
        # En dessous de ce score cosinus, un chunk n'est pas considéré comme
        # une source : mieux vaut ne pas répondre que répondre sans ancrage
        self.min_relevance_score = (
            config.min_relevance_score if min_relevance_score is None else min_relevance_score
        )

        self.loaders = {
            "pdf"      : PDFLoader(),
            "markdown" : MarkdownLoader(),
            "web"      : WebLoader()
        }

    def ingest(self, source: str, source_name: str = None) -> list[str]:
        """
        Détecte automatiquement le type de source et l'ingère.

        `source_name` remplace la source enregistrée avec les chunks — pour
        un fichier reçu par l'API, stocké sous un nom temporaire. Retourne
        les ids des chunks nouvellement indexés.
        """
        loader = self.loaders[source_type(source)]
        data = loader.load(source)
        if source_name:
            data.source = source_name

        return self.retriever.ingest(data)

    def _prepare(self, question: str, top_k: int, history: list[dict]) -> tuple[str, str, list[Source]]:
        """
        Question → Retrieve → filtre de pertinence → Prompt ajusté à la
        fenêtre de contexte. Retourne (issue, prompt, sources placées dans le
        prompt) ; hors `ANSWERED`, le prompt est vide et le LLM inutile
        """
        logger.info("Recherche des chunks pertinents...")
        chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            return outcome.NO_DOCUMENTS, "", []

        relevant = [c for c in chunks if c.get("score", 1.0) >= self.min_relevance_score]
        logger.debug("%d chunks récupérés, %d assez pertinents", len(chunks), len(relevant))
        if not relevant:
            return outcome.NO_RELEVANT_CONTEXT, "", []

        prompt, relevant, _ = fit_prompt(
            question, relevant, history, self.llm.count_tokens, self.llm.prompt_budget
        )
        return outcome.ANSWERED, prompt, [Source.from_chunk(c) for c in relevant]

    def ask(self, question: str, top_k: int = 5, history: list[dict] = None) -> Answer:
        """
        Pipeline complet RAG :
        Question → Retrieve → Prompt → LLM local → Réponse

        La réponse porte les chunks réellement placés dans le prompt (après
        troncature), pas tous ceux récupérés.

        Sans `history`, l'agent est conversationnel : il utilise et met à
        jour son propre historique. Avec `history` (éventuellement vide),
        l'appel est sans état — c'est le mode de l'API, partagée entre
        plusieurs clients.
        """
        status, prompt, sources = self._prepare(question, top_k, self._history_for(history))
        if status != outcome.ANSWERED:
            return Answer(text=REFUSALS[status], status=status)

        logger.info("Génération de la réponse...")
        answer = Answer(text=self.llm.generate(prompt), sources=sources)

        if history is None:
            self._remember(question, answer.text)
        return answer

    def ask_stream(self, question: str, top_k: int = 5, history: list[dict] = None) -> StreamingAnswer:
        """
        Comme `ask`, mais le texte est produit au fil de la génération.
        En mode conversationnel, l'échange n'entre dans l'historique qu'une
        fois le flux épuisé.
        """
        status, prompt, sources = self._prepare(question, top_k, self._history_for(history))
        if status != outcome.ANSWERED:
            return StreamingAnswer(tokens=iter([REFUSALS[status]]), status=status)

        def tokens():
            logger.info("Génération de la réponse (streaming)...")
            pieces = []
            for piece in self.llm.stream(prompt):
                pieces.append(piece)
                yield piece
            if history is None:
                self._remember(question, "".join(pieces))

        return StreamingAnswer(tokens=tokens(), sources=sources)

    def _history_for(self, history: list[dict] | None) -> list[dict]:
        """
        Historique à placer dans le prompt : le sien en mode
        conversationnel, sinon celui fourni, borné aux derniers échanges
        """
        if history is None:
            return self.history
        return history[-2 * self.history_turns:] if self.history_turns > 0 else []

    def _remember(self, question: str, answer: str):
        """
        Ajoute l'échange à l'historique, borné aux `history_turns` derniers
        """
        self.history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self.history = self.history[-2 * self.history_turns:] if self.history_turns > 0 else []

    def save(self, path: str = None):
        self.retriever.save(path)

    def load(self, path: str = None):
        self.retriever.load(path)
