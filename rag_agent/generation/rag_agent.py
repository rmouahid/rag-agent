import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field

from rag_agent.config import config
from rag_agent.generation import answer as outcome
from rag_agent.generation.agentic import RetrievalStep, SufficiencyJudge
from rag_agent.generation.answer import Answer, Source, Status, StreamingAnswer
from rag_agent.generation.llm import LocalLLM
from rag_agent.generation.query_rewriter import QueryRewriter
from rag_agent.generation.truncation import fit_prompt
from rag_agent.logging_config import get_logger
from rag_agent.pipeline import UnsupportedSourceError, source_type  # noqa: F401 (re-exported)
from rag_agent.vectorstore.reranker import CrossEncoderReranker
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


@dataclass(frozen=True)
class RetrievedContext:
    """
    Contexte trouvé pour une question : issue, chunks retenus (du plus
    pertinent au moins pertinent) et détail de chaque recherche
    """

    status: Status
    chunks: list[dict]
    steps: list[RetrievalStep] = field(default_factory=list)


def _sources(chunks: list[dict]) -> list[str]:
    return [c["source"] for c in chunks]


class RAGAgent:
    def __init__(
        self,
        llm: LocalLLM | None = None,
        history_turns: int | None = None,
        min_relevance_score: float | None = None,
        retriever: Retriever | None = None,
        query_rewriting: bool | None = None,
        reranker: CrossEncoderReranker | None = None,
        agentic: bool | None = None,
        max_iterations: int | None = None,
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
        # Reranking des candidats par un cross-encoder (RERANK) : une passe du
        # modèle par candidat, sur RERANK_CANDIDATES candidats
        self.reranker = reranker or (CrossEncoderReranker() if config.rerank else None)
        self.rerank_candidates = config.rerank_candidates
        # Boucle agentique (AGENTIC_LOOP) : le LLM juge si le contexte suffit
        # et relance la recherche sinon, au plus `max_iterations` recherches
        agentic = config.agentic_loop if agentic is None else agentic
        self.sufficiency_judge = SufficiencyJudge(self.llm) if agentic else None
        self.max_iterations = config.agentic_max_iterations if max_iterations is None else max_iterations
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")

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

    def find_context(self, question: str, top_k: int = 5, history: list[dict] | None = None) -> RetrievedContext:
        """
        Question → (réécriture) → Retrieve → (reranking) → filtre de
        pertinence, puis, avec la boucle agentique, jugement du contexte et
        nouvelles recherches tant qu'il ne suffit pas. Retourne l'issue, les
        chunks retenus et le détail de chaque recherche, sans appeler le LLM
        pour répondre.
        """
        status, chunks = self._search(question, question, top_k, history, rewrite=True)
        steps = [RetrievalStep(query=question, sources=_sources(chunks), n_chunks=len(chunks), decision="enough")]
        if self.sufficiency_judge is None or status == outcome.NO_DOCUMENTS:
            return RetrievedContext(status, chunks, steps)

        tried = [question]
        for iteration in range(1, self.max_iterations + 1):
            if iteration == self.max_iterations:
                steps[-1] = dataclasses.replace(steps[-1], decision="max_iterations")
                logger.info("Boucle agentique : %d recherches, limite atteinte", iteration)
                break
            decision, parsed = self.sufficiency_judge.decide(question, chunks, tried)
            if decision.enough or decision.query is None:
                steps[-1] = dataclasses.replace(steps[-1], decision="enough" if parsed else "unparsed")
                logger.info("Boucle agentique : contexte suffisant après %d recherche(s)", iteration)
                break
            steps[-1] = dataclasses.replace(steps[-1], decision="search")
            logger.info("Boucle agentique, recherche %d : %r (%s)", iteration + 1, decision.query, decision.reason)
            tried.append(decision.query)
            _, found = self._search(question, decision.query, top_k, history, rewrite=False)
            known = {c["id"] for c in chunks}
            new = [c for c in found if c["id"] not in known]
            chunks = chunks + new
            steps.append(
                RetrievalStep(query=decision.query, sources=_sources(new), n_chunks=len(new), decision="enough")
            )
            logger.info("Boucle agentique : %d nouveau(x) chunk(s) : %s", len(new), _sources(new))

        return RetrievedContext(outcome.ANSWERED if chunks else status, chunks, steps)

    def _search(
        self, question: str, query: str, top_k: int, history: list[dict] | None, rewrite: bool
    ) -> tuple[Status, list[dict]]:
        """
        Une recherche pour `query` : (réécriture) → Retrieve → filtre de
        pertinence → (reranking par rapport à la question d'origine) → top_k
        """
        logger.info("Recherche des chunks pertinents...")
        n_candidates = max(top_k, self.rerank_candidates) if self.reranker is not None else top_k
        if rewrite and self.query_rewriter is not None:
            queries = self.query_rewriter.rewrite(query, history)
            chunks = self.retriever.retrieve_many(queries, top_k=n_candidates)
        else:
            chunks = self.retriever.retrieve(query, top_k=n_candidates)
        if not chunks:
            return outcome.NO_DOCUMENTS, []

        relevant = [c for c in chunks if c.get("score", 1.0) >= self.min_relevance_score]
        logger.debug("%d chunks récupérés, %d assez pertinents", len(chunks), len(relevant))
        if not relevant:
            return outcome.NO_RELEVANT_CONTEXT, []
        if self.reranker is not None:
            # Le cross-encoder note la paire avec la question d'origine, pas
            # avec ses réécritures
            relevant = self.reranker.rerank(question, relevant)
        return outcome.ANSWERED, relevant[:top_k]

    def _prepare(
        self, question: str, top_k: int, history: list[dict]
    ) -> tuple[Status, str, list[Source], list[RetrievalStep]]:
        """
        Contexte retenu (find_context) → Prompt ajusté à la fenêtre de
        contexte. Retourne (issue, prompt, sources placées dans le prompt,
        recherches effectuées) ; hors `ANSWERED`, le prompt est vide et le
        LLM inutile
        """
        context = self.find_context(question, top_k, history)
        if context.status != outcome.ANSWERED:
            return context.status, "", [], context.steps

        prompt, relevant, _ = fit_prompt(
            question, context.chunks, history, self.llm.count_tokens, self.llm.prompt_budget
        )
        return outcome.ANSWERED, prompt, [Source.from_chunk(c) for c in relevant], context.steps

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
        status, prompt, sources, steps = self._prepare(question, top_k, self._history_for(history))
        if status != outcome.ANSWERED:
            return Answer(text=REFUSALS[status], status=status, steps=steps)

        logger.info("Génération de la réponse...")
        answer = Answer(text=self.llm.generate(prompt), sources=sources, steps=steps)

        if history is None:
            self._remember(question, answer.text)
        return answer

    def ask_stream(self, question: str, top_k: int = 5, history: list[dict] | None = None) -> StreamingAnswer:
        """
        Comme `ask`, mais le texte est produit au fil de la génération.
        En mode conversationnel, l'échange n'entre dans l'historique qu'une
        fois le flux épuisé.
        """
        status, prompt, sources, steps = self._prepare(question, top_k, self._history_for(history))
        if status != outcome.ANSWERED:
            return StreamingAnswer(tokens=iter([REFUSALS[status]]), status=status, steps=steps)

        def tokens() -> Iterator[str]:
            logger.info("Génération de la réponse (streaming)...")
            pieces = []
            for piece in self.llm.stream(prompt):
                pieces.append(piece)
                yield piece
            if history is None:
                self._remember(question, "".join(pieces))

        return StreamingAnswer(tokens=tokens(), sources=sources, steps=steps)

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
