from rag_agent.config import config
from rag_agent.generation.answer import Answer, Source
from rag_agent.generation.llm import LocalLLM
from rag_agent.generation.truncation import fit_prompt
from rag_agent.vectorstore.retriever import Retriever
from rag_agent.ingestion.pdf_loader import PDFLoader
from rag_agent.ingestion.md_loader import MarkdownLoader
from rag_agent.ingestion.web_loader import WebLoader
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class RAGAgent:
    def __init__(self, llm: LocalLLM = None, history_turns: int = None):
        # Chargé dès la construction : sans poids, l'agent échoue tout de
        # suite avec un message clair plutôt qu'à la première question.
        self.llm       = llm or LocalLLM()
        self.retriever = Retriever()
        # Échanges question/réponse récents rappelés dans le prompt
        self.history_turns = config.history_turns if history_turns is None else history_turns
        self.history       = []

        self.loaders = {
            "pdf"      : PDFLoader(),
            "markdown" : MarkdownLoader(),
            "web"      : WebLoader()
        }

    def ingest(self, source: str):
        """
        Détecte automatiquement le type de source et l'ingère
        """
        if source.startswith("http"):
            data = self.loaders["web"].load(source)
        elif source.endswith(".pdf"):
            data = self.loaders["pdf"].load(source)
        elif source.endswith(".md"):
            data = self.loaders["markdown"].load(source)
        else:
            raise ValueError(f" Format non supporté : {source}")

        self.retriever.ingest(data)

    def ask(self, question: str, top_k: int = 5) -> Answer:
        """
        Pipeline complet RAG :
        Question → Retrieve → Prompt → LLM local → Réponse

        La réponse porte les chunks réellement placés dans le prompt (après
        troncature), pas tous ceux récupérés.
        """
        logger.info("Recherche des chunks pertinents...")
        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            return Answer(text=" Aucun document indexé. Ingère d'abord des sources.")

        logger.debug("%d chunks récupérés", len(chunks))
        logger.info("Génération de la réponse...")

        prompt, chunks, _ = fit_prompt(
            question, chunks, self.history, self.llm.count_tokens, self.llm.prompt_budget
        )
        answer = Answer(
            text=self.llm.generate(prompt),
            sources=[Source.from_chunk(c) for c in chunks],
        )

        self._remember(question, answer.text)
        return answer

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
