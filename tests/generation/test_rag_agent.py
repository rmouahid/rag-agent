import pytest

from rag_agent.generation import rag_agent as rag_agent_module
from rag_agent.generation.rag_agent import RAGAgent


class FakeRetriever:
    chunks = []
    pipeline = None

    def retrieve(self, question, top_k):
        return self.chunks[:top_k]


class FakeLLM:
    def __init__(self, prompt_budget=10_000):
        self.prompts = []
        self.prompt_budget = prompt_budget

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "answer"

    def stream(self, prompt):
        self.prompts.append(prompt)
        yield from ["ans", "wer"]


@pytest.fixture(autouse=True)
def fake_retriever(monkeypatch):
    FakeRetriever.chunks = []
    monkeypatch.setattr(rag_agent_module, "Retriever", FakeRetriever)


def test_ask_sends_the_question_and_retrieved_context_to_the_local_llm():
    FakeRetriever.chunks = [
        {"source": "a.md", "text": "Paris is the capital of France."},
        {"source": "b.pdf", "text": "Berlin is the capital of Germany."},
    ]
    llm = FakeLLM()

    answer = RAGAgent(llm=llm).ask("What is the capital of France?")

    assert answer.text == "answer"
    (prompt,) = llm.prompts
    assert "What is the capital of France?" in prompt
    assert "[Source: a.md]\nParis is the capital of France." in prompt
    assert "[Source: b.pdf]\nBerlin is the capital of Germany." in prompt


def test_ask_without_indexed_documents_does_not_call_the_llm():
    llm = FakeLLM()

    answer = RAGAgent(llm=llm).ask("anything")

    assert "Aucun document" in answer.text
    assert answer.sources == []
    assert llm.prompts == []


def test_previous_exchanges_are_recalled_in_the_next_prompt():
    FakeRetriever.chunks = [{"source": "a.md", "text": "Les Misérables, by Victor Hugo."}]
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, history_turns=3)

    agent.ask("Who wrote Les Misérables?")
    agent.ask("When was it published?")

    first, second = llm.prompts
    assert "HISTORIQUE" not in first
    assert "Who wrote Les Misérables?" in second
    assert second.index("Who wrote Les Misérables?") < second.index("When was it published?")


def test_history_keeps_only_the_last_turns():
    FakeRetriever.chunks = [{"source": "a.md", "text": "text"}]
    agent = RAGAgent(llm=FakeLLM(), history_turns=1)

    agent.ask("first")
    agent.ask("second")

    assert [turn["content"] for turn in agent.history] == ["second", "answer"]


def test_zero_history_turns_disables_history():
    FakeRetriever.chunks = [{"source": "a.md", "text": "text"}]
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, history_turns=0)

    agent.ask("first")
    agent.ask("second")

    assert agent.history == []
    assert "first" not in llm.prompts[1]


def test_ask_leaves_out_the_least_relevant_chunks_when_the_prompt_overflows():
    FakeRetriever.chunks = [
        {"source": "best.md", "text": "relevant " * 50, "score": 0.9},
        {"source": "worst.md", "text": "noise " * 500, "score": 0.1},
    ]
    llm = FakeLLM(prompt_budget=200)

    RAGAgent(llm=llm).ask("question")

    (prompt,) = llm.prompts
    assert "best.md" in prompt
    assert "worst.md" not in prompt


def test_answer_lists_the_chunks_used_to_build_the_prompt():
    FakeRetriever.chunks = [
        {"id": "1", "source": "a.md", "text": "Paris is the capital of France.", "score": 0.9},
        {"id": "2", "source": "b.pdf", "text": "Berlin is the capital of Germany.", "score": 0.4},
    ]

    answer = RAGAgent(llm=FakeLLM()).ask("What is the capital of France?")

    assert [(s.chunk_id, s.source, s.excerpt) for s in answer.sources] == [
        ("1", "a.md", "Paris is the capital of France."),
        ("2", "b.pdf", "Berlin is the capital of Germany."),
    ]


def test_chunks_left_out_by_truncation_are_not_cited():
    FakeRetriever.chunks = [
        {"id": "best", "source": "best.md", "text": "relevant " * 50, "score": 0.9},
        {"id": "worst", "source": "worst.md", "text": "noise " * 500, "score": 0.1},
    ]

    answer = RAGAgent(llm=FakeLLM(prompt_budget=200)).ask("question")

    assert [s.chunk_id for s in answer.sources] == ["best"]


def test_ask_stream_yields_the_answer_piece_by_piece_with_its_sources_upfront():
    FakeRetriever.chunks = [{"id": "1", "source": "a.md", "text": "Paris.", "score": 0.9}]
    llm = FakeLLM()

    answer = RAGAgent(llm=llm).ask_stream("Capital of France?")

    assert [s.source for s in answer.sources] == ["a.md"]
    assert list(answer) == ["ans", "wer"]
    assert "Capital of France?" in llm.prompts[0]


def test_streamed_exchange_enters_history_once_the_stream_is_consumed():
    FakeRetriever.chunks = [{"source": "a.md", "text": "text"}]
    agent = RAGAgent(llm=FakeLLM())

    answer = agent.ask_stream("first")
    assert agent.history == []

    list(answer)
    assert agent.history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]


def test_ask_stream_without_indexed_documents_does_not_call_the_llm():
    llm = FakeLLM()

    answer = RAGAgent(llm=llm).ask_stream("anything")

    assert "Aucun document" in "".join(answer)
    assert answer.sources == []
    assert llm.prompts == []


def test_chunks_below_the_relevance_threshold_are_not_used():
    FakeRetriever.chunks = [
        {"source": "relevant.md", "text": "Paris.", "score": 0.6},
        {"source": "noise.md", "text": "Unrelated.", "score": 0.1},
    ]
    llm = FakeLLM()

    answer = RAGAgent(llm=llm, min_relevance_score=0.3).ask("capital?")

    assert [s.source for s in answer.sources] == ["relevant.md"]
    assert "noise.md" not in llm.prompts[0]


def test_no_relevant_chunk_gives_an_explicit_answer_without_calling_the_llm():
    FakeRetriever.chunks = [{"source": "noise.md", "text": "Unrelated.", "score": 0.1}]
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, min_relevance_score=0.3)

    answer = agent.ask("capital?")
    streamed = agent.ask_stream("capital?")

    assert answer.status == streamed.status == "no_relevant_context"
    assert "pertinent" in answer.text and "pertinent" in "".join(streamed)
    assert answer.sources == []
    assert llm.prompts == []


def test_empty_store_is_reported_as_no_documents():
    answer = RAGAgent(llm=FakeLLM()).ask("anything")

    assert answer.status == "no_documents"


def test_a_given_history_is_used_but_the_agent_history_is_left_untouched():
    FakeRetriever.chunks = [{"source": "a.md", "text": "text", "score": 0.9}]
    llm = FakeLLM()
    agent = RAGAgent(llm=llm, history_turns=3)
    given = [{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer"}]

    agent.ask("now?", history=given)
    list(agent.ask_stream("again?", history=[]))

    assert "earlier question" in llm.prompts[0]
    assert "earlier question" not in llm.prompts[1]
    assert agent.history == []


def test_a_given_history_is_capped_to_the_last_turns():
    FakeRetriever.chunks = [{"source": "a.md", "text": "text", "score": 0.9}]
    llm = FakeLLM()
    given = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]

    RAGAgent(llm=llm, history_turns=1).ask("now?", history=given)

    assert "old question" not in llm.prompts[0]
    assert "recent question" in llm.prompts[0]


# --- query rewriting ----------------------------------------------------------


class RewritingLLM(FakeLLM):
    """Answers the rewriting prompt with search queries, any other prompt with « answer »"""

    def generate(self, prompt):
        if prompt.startswith("Rewrite the question"):
            return "Amber Line last train\nbicycle rules"
        return super().generate(prompt)


def test_rewriting_searches_the_question_and_its_rewrites(monkeypatch):
    calls = []

    def retrieve_many(self, queries, top_k):
        calls.append((queries, top_k))
        return [{"source": "a.md", "text": "text", "score": 0.9}]

    monkeypatch.setattr(FakeRetriever, "retrieve_many", retrieve_many, raising=False)
    llm = RewritingLLM()

    answer = RAGAgent(llm=llm, query_rewriting=True).ask("Last train, and bikes?", top_k=4)

    assert calls == [(["Last train, and bikes?", "Amber Line last train", "bicycle rules"], 4)]
    assert answer.text == "answer"
    assert len(llm.prompts) == 1  # le prompt de réécriture ne passe pas par FakeLLM.generate


def test_rewriting_is_off_by_default():
    FakeRetriever.chunks = [{"source": "a.md", "text": "text"}]

    agent = RAGAgent(llm=RewritingLLM())

    assert agent.query_rewriter is None
    assert agent.ask("question").text == "answer"


# --- reranking ----------------------------------------------------------------


class ReverseReranker:
    """Puts the vector search results in reverse order."""

    def __init__(self):
        self.queries = []

    def rerank(self, query, chunks):
        self.queries.append(query)
        return [{**c, "rerank_score": float(i)} for i, c in enumerate(chunks)][::-1]


def test_reranking_reorders_more_candidates_then_keeps_top_k():
    FakeRetriever.chunks = [{"source": f"{i}.md", "text": f"chunk {i}", "score": 0.9 - i / 100} for i in range(30)]
    reranker = ReverseReranker()
    agent = RAGAgent(llm=FakeLLM(), reranker=reranker)
    agent.rerank_candidates = 20

    context = agent.find_context("question", top_k=3)

    assert context.status == "answered"
    assert [c["source"] for c in context.chunks] == ["19.md", "18.md", "17.md"]
    assert reranker.queries == ["question"]


def test_reranked_chunks_reach_the_prompt_in_reranked_order():
    FakeRetriever.chunks = [
        {"source": "a.md", "text": "first", "score": 0.9},
        {"source": "b.md", "text": "second", "score": 0.8},
    ]
    llm = FakeLLM()

    answer = RAGAgent(llm=llm, reranker=ReverseReranker()).ask("question", top_k=2)

    assert [s.source for s in answer.sources] == ["b.md", "a.md"]
    assert llm.prompts[0].index("second") < llm.prompts[0].index("first")


def test_chunks_below_the_relevance_threshold_are_not_reranked():
    FakeRetriever.chunks = [
        {"source": "a.md", "text": "a", "score": 0.9},
        {"source": "b.md", "text": "b", "score": 0.1},
    ]

    context = RAGAgent(llm=FakeLLM(), min_relevance_score=0.25, reranker=ReverseReranker()).find_context("q")

    assert [c["source"] for c in context.chunks] == ["a.md"]


def test_rewritten_queries_are_reranked_against_the_original_question(monkeypatch):
    monkeypatch.setattr(
        FakeRetriever,
        "retrieve_many",
        lambda self, queries, top_k: [{"source": "a.md", "text": "a", "score": 0.9}],
        raising=False,
    )
    reranker = ReverseReranker()

    RAGAgent(llm=RewritingLLM(), query_rewriting=True, reranker=reranker).find_context("Last train, and bikes?")

    assert reranker.queries == ["Last train, and bikes?"]


def test_reranking_is_off_by_default():
    assert RAGAgent(llm=FakeLLM()).reranker is None
