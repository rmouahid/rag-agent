"""
Injection de dépendances : les routes reçoivent l'agent (et, à travers
lui, le vector store et le LLM) via `Depends`, jamais un global. Les
tests remplacent l'agent par `app.dependency_overrides[get_agent]`.
"""

import threading

from fastapi import Depends, Request

from rag_agent.generation.llm import LocalLLM
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.vectorstore.vector_store import VectorStore


# Un seul appel à l'agent à la fois : llama.cpp n'accepte pas de générations
# concurrentes sur le même modèle, et le vector store (index IVF compris)
# n'est pas protégé contre une ingestion pendant une recherche.
agent_lock = threading.Lock()


def get_agent(request: Request) -> RAGAgent:
    """
    Agent unique de l'application, construit au démarrage (lifespan)
    """
    return request.app.state.agent


def get_vector_store(agent: RAGAgent = Depends(get_agent)) -> VectorStore:
    return agent.retriever.vector_store


def get_llm(agent: RAGAgent = Depends(get_agent)) -> LocalLLM:
    return agent.llm
