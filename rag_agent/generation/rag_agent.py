from rag_agent.generation.llm import LocalLLM
from rag_agent.vectorstore.retriever import Retriever
from rag_agent.ingestion.pdf_loader import PDFLoader
from rag_agent.ingestion.md_loader import MarkdownLoader
from rag_agent.ingestion.web_loader import WebLoader
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)


class RAGAgent:
    def __init__(self, llm: LocalLLM = None):
        # Chargé dès la construction : sans poids, l'agent échoue tout de
        # suite avec un message clair plutôt qu'à la première question.
        self.llm       = llm or LocalLLM()
        self.retriever = Retriever()

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

    def ask(self, question: str, top_k: int = 5) -> str:
        """
        Pipeline complet RAG :
        Question → Retrieve → Prompt → LLM local → Réponse
        """
        logger.info("Recherche des chunks pertinents...")
        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            return " Aucun document indexé. Ingère d'abord des sources."

        # Construit le contexte à partir des chunks récupérés
        context = "\n\n---\n\n".join([
            f"[Source: {c['source']}]\n{c['text']}"
            for c in chunks
        ])

        logger.debug("%d chunks récupérés", len(chunks))
        logger.info("Génération de la réponse...")

        # Prompt RAG
        prompt = f"""Tu es un assistant précis et factuel.
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""

        return self.llm.generate(prompt)

    def save(self, path: str = None):
        self.retriever.save(path)

    def load(self, path: str = None):
        self.retriever.load(path)
