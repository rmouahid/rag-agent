from groq import Groq
from core.retriever import Retriever
from loaders.pdf_loader import PDFLoader
from loaders.md_loader import MarkdownLoader
from loaders.web_loader import WebLoader
from dotenv import load_dotenv
import os

load_dotenv()

class RAGAgent:
    def __init__(self):
        self.retriever = Retriever()
        self.client    = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model     = "llama-3.3-70b-versatile"

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

        self.retriever.ingest(data["text"], metadata=data["metadata"])

    def ask(self, question: str, top_k: int = 5) -> str:
        """
        Pipeline complet RAG :
        Question → Retrieve → Prompt → Groq → Réponse
        """
        print(f"\n🔍 Recherche des chunks pertinents...")
        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            return " Aucun document indexé. Ingère d'abord des sources."

        # Construit le contexte à partir des chunks récupérés
        context = "\n\n---\n\n".join([
            f"[Source: {c['metadata']['source']}]\n{c['text']}"
            for c in chunks
        ])

        print(f"    {len(chunks)} chunks récupérés")
        print(f"\n Génération de la réponse...\n")

        # Prompt RAG
        prompt = f"""Tu es un assistant précis et factuel.
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    def save(self, path: str = "vector_db"):
        self.retriever.save(path)

    def load(self, path: str = "vector_db"):
        self.retriever.load(path)