import pytest

from rag_agent.generation.agentic import SufficiencyJudge, is_parsed, parse_decision
from rag_agent.generation.rag_agent import RAGAgent


def chunk(name, score=0.9):
    return {"id": name, "source": f"{name}.md", "text": f"text of {name}", "score": score}


class QueryRetriever:
    """Returns the chunks registered for each query; records every search."""

    pipeline = None

    def __init__(self, results):
        self.results = results
        self.searches = []

    def _lookup(self, query, top_k):
        return self.results.get(query, self.results.get("*", []))[:top_k]

    def retrieve(self, query, top_k):
        self.searches.append(("retrieve", query))
        return self._lookup(query, top_k)

    def retrieve_many(self, queries, top_k):
        self.searches.append(("retrieve_many", tuple(queries)))
        return self._lookup(queries[0], top_k)


class ScriptedLLM:
    """
    Replies to the sufficiency prompt with the next scripted decision (the
    last one repeats), to the rewriting prompt with `rewrites`, and to the
    answer prompt with « answer ».
    """

    prompt_budget = 10_000

    def __init__(self, decisions=("ENOUGH",), rewrites="rewritten query"):
        self.decisions = list(decisions)
        self.rewrites = rewrites
        self.judge_prompts = []
        self.answer_prompts = []

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt):
        if prompt.startswith("You decide whether"):
            self.judge_prompts.append(prompt)
            return self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]
        if prompt.startswith("Rewrite the question"):
            return self.rewrites
        self.answer_prompts.append(prompt)
        return "answer"

    def stream(self, prompt):
        yield self.generate(prompt)


def agent(llm, results, **kwargs):
    retriever = QueryRetriever(results)
    return RAGAgent(llm=llm, retriever=retriever, agentic=True, min_relevance_score=0.25, **kwargs), retriever


# --- decision parsing -----------------------------------------------------------


@pytest.mark.parametrize("reply", ["ENOUGH", "enough.", "ENOUGH: the passages give the date."])
def test_enough_stops_the_search(reply):
    decision = parse_decision(reply, tried=["q"])

    assert decision.enough and decision.query is None


@pytest.mark.parametrize(
    "reply, query",
    [
        ("SEARCH: Amber Line last train", "Amber Line last train"),
        ('search: "bicycle fare"', "bicycle fare"),
        ("The passages miss the fare.\nSEARCH: bicycle fare", "bicycle fare"),
    ],
)
def test_search_gives_the_next_query(reply, query):
    decision = parse_decision(reply, tried=["question"])

    assert not decision.enough
    assert decision.query == query


def test_a_query_already_tried_ends_the_loop():
    assert parse_decision("SEARCH: Bicycle Fare", tried=["question", "bicycle fare"]).enough


def test_an_unparsable_reply_ends_the_loop_and_is_flagged():
    assert parse_decision("I am not sure what to do.", tried=["q"]).enough
    assert not is_parsed("I am not sure what to do.")
    assert is_parsed("SEARCH: x") and is_parsed("ENOUGH")


def test_the_judge_sees_the_question_the_passages_and_the_queries_tried():
    llm = ScriptedLLM()

    SufficiencyJudge(llm).decide("When is the last train?", [chunk("timetable")], ["When is the last train?", "x"])

    prompt = llm.judge_prompts[0]
    assert "Question: When is the last train?" in prompt
    assert "[Source: timetable.md]\ntext of timetable" in prompt
    assert "- When is the last train?\n- x" in prompt


def test_the_judge_context_is_truncated():
    llm = ScriptedLLM()

    SufficiencyJudge(llm, max_context_chars=20).decide("q", [chunk("a" * 100)], ["q"])

    assert "a" * 30 not in llm.judge_prompts[0]


# --- the loop ---------------------------------------------------------------------


def test_enough_after_the_first_search_means_a_single_pass():
    llm = ScriptedLLM(["ENOUGH"])
    rag, retriever = agent(llm, {"question": [chunk("a")]})

    context = rag.find_context("question")

    assert [s.query for s in context.steps] == ["question"]
    assert context.steps[0].decision == "enough"
    assert len(llm.judge_prompts) == 1
    assert retriever.searches == [("retrieve", "question")]


def test_a_new_query_adds_its_new_passages_to_the_context():
    llm = ScriptedLLM(["SEARCH: bicycle fare", "ENOUGH"])
    rag, retriever = agent(llm, {"question": [chunk("a"), chunk("b")], "bicycle fare": [chunk("b"), chunk("c")]})

    context = rag.find_context("question")

    assert [c["id"] for c in context.chunks] == ["a", "b", "c"]
    assert [(s.query, s.sources, s.decision) for s in context.steps] == [
        ("question", ["a.md", "b.md"], "search"),
        ("bicycle fare", ["c.md"], "enough"),
    ]
    assert retriever.searches == [("retrieve", "question"), ("retrieve", "bicycle fare")]
    assert "- question\n- bicycle fare" in llm.judge_prompts[1]


def test_the_loop_stops_at_the_iteration_limit():
    llm = ScriptedLLM([f"SEARCH: query {i}" for i in range(10)])
    rag, retriever = agent(llm, {"*": [chunk("a")]}, max_iterations=3)

    context = rag.find_context("question")

    assert [s.query for s in context.steps] == ["question", "query 0", "query 1"]
    assert context.steps[-1].decision == "max_iterations"
    assert len(llm.judge_prompts) == 2  # pas de jugement inutile après la dernière recherche permise
    assert len(retriever.searches) == 3


def test_a_single_iteration_never_asks_the_judge():
    llm = ScriptedLLM(["SEARCH: more"])
    rag, _ = agent(llm, {"*": [chunk("a")]}, max_iterations=1)

    context = rag.find_context("question")

    assert len(context.steps) == 1
    assert llm.judge_prompts == []


def test_a_repeated_query_ends_the_loop():
    llm = ScriptedLLM(["SEARCH: other", "SEARCH: question"])
    rag, retriever = agent(llm, {"*": [chunk("a")]}, max_iterations=5)

    context = rag.find_context("question")

    assert len(context.steps) == 2
    assert len(retriever.searches) == 2


def test_an_unparsable_decision_is_recorded():
    rag, _ = agent(ScriptedLLM(["hmm"]), {"*": [chunk("a")]})

    assert rag.find_context("question").steps[-1].decision == "unparsed"


def test_without_documents_the_loop_does_not_run():
    llm = ScriptedLLM()
    rag, _ = agent(llm, {})

    context = rag.find_context("question")

    assert context.status == "no_documents"
    assert llm.judge_prompts == []


def test_a_follow_up_search_can_rescue_a_question_without_relevant_context():
    llm = ScriptedLLM(["SEARCH: Lumen Miles", "ENOUGH"])
    rag, _ = agent(llm, {"question": [chunk("weak", score=0.1)], "Lumen Miles": [chunk("miles")]})

    context = rag.find_context("question")

    assert context.status == "answered"
    assert [c["id"] for c in context.chunks] == ["miles"]
    assert "(aucun document pertinent)" in llm.judge_prompts[0]


def test_irrelevant_follow_up_results_leave_the_question_unanswered():
    llm = ScriptedLLM(["SEARCH: other"])
    rag, _ = agent(llm, {"*": [chunk("weak", score=0.1)]}, max_iterations=2)

    assert rag.find_context("question").status == "no_relevant_context"


def test_rewriting_applies_to_the_first_search_only():
    llm = ScriptedLLM(["SEARCH: follow up", "ENOUGH"])
    rag, retriever = agent(llm, {"*": [chunk("a")]}, query_rewriting=True)

    rag.find_context("question")

    assert retriever.searches == [("retrieve_many", ("question", "rewritten query")), ("retrieve", "follow up")]


def test_follow_up_results_are_reranked_against_the_original_question():
    class RecordingReranker:
        queries = []

        def rerank(self, query, chunks):
            self.queries.append(query)
            return chunks

    reranker = RecordingReranker()
    rag, _ = agent(ScriptedLLM(["SEARCH: follow up", "ENOUGH"]), {"*": [chunk("a")]}, reranker=reranker)

    rag.find_context("question")

    assert reranker.queries == ["question", "question"]


def test_the_answer_is_generated_once_on_the_final_context():
    llm = ScriptedLLM(["SEARCH: bicycle fare", "ENOUGH"])
    rag, _ = agent(llm, {"question": [chunk("a")], "bicycle fare": [chunk("c")]})

    answer = rag.ask("question", history=[])

    assert answer.text == "answer"
    (prompt,) = llm.answer_prompts
    assert "text of a" in prompt and "text of c" in prompt
    assert [s.source for s in answer.sources] == ["a.md", "c.md"]
    assert [s.query for s in answer.steps] == ["question", "bicycle fare"]


def test_streaming_answers_carry_the_searches_too():
    rag, _ = agent(ScriptedLLM(["SEARCH: more", "ENOUGH"]), {"*": [chunk("a")], "more": [chunk("b")]})

    answer = rag.ask_stream("question", history=[])

    assert "".join(answer) == "answer"
    assert [s.query for s in answer.steps] == ["question", "more"]


def test_the_loop_is_off_by_default():
    rag = RAGAgent(llm=ScriptedLLM(), retriever=QueryRetriever({"*": [chunk("a")]}))

    context = rag.find_context("question")

    assert rag.sufficiency_judge is None
    assert len(context.steps) == 1


def test_max_iterations_must_be_positive():
    with pytest.raises(ValueError, match="max_iterations"):
        RAGAgent(llm=ScriptedLLM(), retriever=QueryRetriever({}), agentic=True, max_iterations=0)
