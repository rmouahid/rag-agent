from collections.abc import Iterator

from rag_agent.config import config
from rag_agent.generation import answer as outcome
from rag_agent.generation.answer import Answer, Source, Status, StreamingAnswer
from rag_agent.generation.llm import LocalLLM
from rag_agent.generation.query_rewriter import QueryRewriter
from rag_agent.generation.truncation import fit_prompt
from rag_agent.logging_config import get_logger
from rag_agent.pipeline import UnsupportedSourceError, source_type  # noqa: F401 (re-exported)
from rag_agent.vectorstore.retriever import Retriever

logger = get_logger(__name__)

# Réponses fixes quand le LLM n'est pas appelé, faute de matière
REFUSALS = {
    outcome.NO_DOCUMENTS: " Aucun document indexé. Ingère d'abord des sources.",
    outcome.NO_RELEVANT_CONTEXT: (
        " Je n'ai trouvé aucun passage assez pertinent dans les documents indexés " "pour répondre à cette question."
    ),
}
NO_DOCUMENTS = REFUSALS[outcome.NO_DOCUMENTS]


class RAGAgent:
    def __init__(
        self,
        llm: LocalLLM | None = None,
        history_turns: int | None = None,
        min_relevance_score: float | None = None,
        retriever: Retriever | None = None,
        query_rewriting: bool | None = None,
    ):
        # Chargé dès la construction : sans poids, l'agent échoue tout de
        # suite avec un message clair plutôt qu'à la première question.
        self.llm = llm or LocalLLM()
        # Par défaut, le store persistant de VECTOR_STORE_PATH
        self.retriever = retriever or Retriever()
        # Échanges question/réponse récents rappelés dans le prompt
        self.history_turns = config.history_turns if history_turns is None else history_turns
        self.history: list[dict[str, str]] = []
        # En dessous de ce score cosinus, un chunk n'est pas considéré comme
        # une source : mieux vaut ne pas répondre que répondre sans ancrage
        self.min_relevance_score = config.min_relevance_score if min_relevance_score is None else min_relevance_score
        # Réécriture de la question en requêtes de recherche par le LLM, avant
        # le retrieval : un appel au LLM de plus par question
        query_rewriting = config.query_rewriting if query_rewriting is None else query_rewriting
        self.query_rewriter = QueryRewriter(self.llm, config.max_rewritten_queries) if query_rewriting else None

        # Ingestion : source → chunks indexés dans le store du retriever
        self.pipeline = self.retriever.pipeline

    def ingest(self, source: str, source_name: str | None = None) -> list[str]:
        """
        Ingère une source (fichier PDF/Markdown ou URL, type détecté
        automatiquement) par le pipeline d'ingestion.

        `source_name` remplace la source enregistrée avec les chunks — pour
        un fichier reçu par l'API, stocké sous un nom temporaire. Retourne
        les ids des chunks nouvellement indexés.
        """
        return self.pipeline.ingest(source, source_name=source_name).chunk_ids

    def _prepare(self, question: str, top_k: int, history: list[dict]) -> tuple[Status, str, list[Source]]:
        """
        Question → Retrieve → filtre de pertinence → Prompt ajusté à la
        fenêtre de contexte. Retourne (issue, prompt, sources placées dans le
        prompt) ; hors `ANSWERED`, le prompt est vide et le LLM inutile
        """
        logger.info("Recherche des chunks pertinents...")
        if self.query_rewriter is not None:
            chunks = self.retriever.retrieve_many(self.query_rewriter.rewrite(question, history), top_k=top_k)
        else:
            chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            return outcome.NO_DOCUMENTS, "", []

        relevant = [c for c in chunks if c.get("score", 1.0) >= self.min_relevance_score]
        logger.debug("%d chunks récupérés, %d assez pertinents", len(chunks), len(relevant))
        if not relevant:
            return outcome.NO_RELEVANT_CONTEXT, "", []

        prompt, relevant, _ = fit_prompt(question, relevant, history, self.llm.count_tokens, self.llm.prompt_budget)
        return outcome.ANSWERED, prompt, [Source.from_chunk(c) for c in relevant]

    def ask(self, question: str, top_k: int = 5, history: list[dict] | None = None) -> Answer:
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

    def ask_stream(self, question: str, top_k: int = 5, history: list[dict] | None = None) -> StreamingAnswer:
        """
        Comme `ask`, mais le texte est produit au fil de la génération.
        En mode conversationnel, l'échange n'entre dans l'historique qu'une
        fois le flux épuisé.
        """
        status, prompt, sources = self._prepare(question, top_k, self._history_for(history))
        if status != outcome.ANSWERED:
            return StreamingAnswer(tokens=iter([REFUSALS[status]]), status=status)

        def tokens() -> Iterator[str]:
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
        return history[-2 * self.history_turns :] if self.history_turns > 0 else []

    def _remember(self, question: str, answer: str) -> None:
        """
        Ajoute l'échange à l'historique, borné aux `history_turns` derniers
        """
        self.history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self.history = self.history[-2 * self.history_turns :] if self.history_turns > 0 else []

    def save(self, path: str | None = None) -> None:
        self.retriever.save(path)

    def load(self, path: str | None = None) -> None:
        self.retriever.load(path)
