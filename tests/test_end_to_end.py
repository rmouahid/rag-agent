"""
End-to-end test of the RAG pipeline on a small fixture document
(tests/fixtures/knowledge.md, five unrelated topics): ingestion through
the real loader, chunker, deduplication and vector store, then questions
whose answers are written in the document, with each search backend.

The model is an extractive stand-in: it answers with the context sentence
sharing the most words with the question. The answer can therefore only
contain the expected fact if retrieval placed the right passage in the
prompt, which is what these tests check.
"""

import dataclasses
import re
from pathlib import Path

import pytest

from rag_agent.chunking import chunker as chunker_module
from rag_agent.generation import answer as outcome
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.vectorstore import vector_store as vector_store_module

KNOWLEDGE = Path(__file__).parent / "fixtures" / "knowledge.md"

BACKENDS = {
    "brute-force": {"index_type": "brute-force"},
    # Four lists for about a dozen chunks and only two probed per query: the
    # index has to route each question to the right lists
    "ivf": {"index_type": "ivf", "ivf_min_vectors": 0, "ivf_n_lists": 4, "nprobe": 2},
    # Small graph and narrow search: the answer must be reached through the links
    "hnsw": {"index_type": "hnsw", "ivf_min_vectors": 0, "hnsw_m": 2, "hnsw_ef_construction": 4, "hnsw_ef_search": 2},
}

# (question, fact the answer must contain, word of the passage it comes from)
QUESTIONS = [
    ("Who designed and built the Eiffel Tower?", "Gustave Eiffel", "Eiffel"),
    ("How tall is the wrought-iron lattice tower, in metres?", "330 metres", "lattice"),
    ("Which pigments in the chloroplasts absorb light?", "Chlorophyll", "chloroplasts"),
    ("How do worker bees communicate the direction of flowers?", "waggle dance", "bees"),
    ("Who created the Python programming language?", "Guido van Rossum", "Python"),
    ("How high above sea level does the summit of Kilimanjaro reach?", "5,895 metres", "Kibo"),
]


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}


class ExtractiveLLM:
    """Answers with the context sentence that shares the most words with the question."""

    model_path = "models/extractive.gguf"
    is_loaded = True
    prompt_budget = 3000

    def __init__(self):
        self.prompts = []

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt):
        self.prompts.append(prompt)
        context = prompt.split("CONTEXTE :", 1)[1].split("QUESTION :", 1)[0]
        question = words(prompt.split("QUESTION :", 1)[1])
        lines = [line for line in context.splitlines() if not line.startswith("[Source:")]
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(lines))
        return max(sentences, key=lambda s: len(words(s) & question))

    def stream(self, prompt):
        yield from self.generate(prompt).split(" ")


@pytest.fixture(params=list(BACKENDS))
def agent(request, monkeypatch, vector_store_dir):
    # Small chunks: about two sentences each, so each question has one right passage
    monkeypatch.setattr(
        chunker_module, "config", dataclasses.replace(chunker_module.config, chunk_size=200, chunk_overlap=20)
    )
    monkeypatch.setattr(
        vector_store_module, "config", dataclasses.replace(vector_store_module.config, **BACKENDS[request.param])
    )
    agent = RAGAgent(llm=ExtractiveLLM(), min_relevance_score=0.2)
    agent.ingest(str(KNOWLEDGE))

    # Counts the queries answered by the IVF index, to prove it is the one searched
    agent.index_queries = 0
    index = agent.retriever.vector_store.index
    if index is not None:
        search = index.search

        def counting_search(*args, **kwargs):
            agent.index_queries += 1
            return search(*args, **kwargs)

        monkeypatch.setattr(index, "search", counting_search)
    return agent


def chunk_text(agent, chunk_id):
    return next(c["text"] for c in agent.retriever.vector_store.chunks if c["id"] == chunk_id)


def test_the_document_is_indexed_with_the_selected_backend(agent):
    store = agent.retriever.vector_store

    assert len(store) >= 8
    assert {c["source"] for c in store.chunks} == {str(KNOWLEDGE)}
    if store.index_type == "ivf":
        assert store.uses_index and store.index.n_lists == 4
    elif store.index_type == "hnsw":
        assert store.uses_index and store.index.m == 2 and store.index.covers(len(store))
    else:
        assert store.index is None


@pytest.mark.parametrize("question, fact, topic", QUESTIONS)
def test_answers_come_from_the_relevant_passage(agent, question, fact, topic):
    answer = agent.ask(question, top_k=2, history=[])

    assert answer.status == outcome.ANSWERED
    assert fact in answer.text
    best = answer.sources[0]
    assert best.source == str(KNOWLEDGE)
    assert fact in chunk_text(agent, best.chunk_id)
    assert topic.lower() in chunk_text(agent, best.chunk_id).lower()
    assert agent.index_queries == (0 if agent.retriever.vector_store.index_type == "brute-force" else 1)


def test_the_streamed_answer_matches_the_answer(agent):
    question = "Who created the Python programming language?"

    streamed = agent.ask_stream(question, top_k=2, history=[])

    assert "Guido van Rossum" in " ".join(streamed)
    assert [s.chunk_id for s in streamed.sources] == [
        s.chunk_id for s in agent.ask(question, top_k=2, history=[]).sources
    ]


def test_a_question_the_document_does_not_cover_is_not_answered(agent):
    answer = agent.ask("Quel temps fera-t-il demain matin ?", history=[])

    assert answer.status == outcome.NO_RELEVANT_CONTEXT
    assert agent.llm.prompts == []
