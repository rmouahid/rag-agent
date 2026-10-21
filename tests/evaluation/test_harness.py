import dataclasses
import json
import os

import pytest

from rag_agent.evaluation import harness
from rag_agent.evaluation.dataset import load_dataset
from rag_agent.evaluation.judge import LLMJudge
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.vectorstore.retriever import Retriever
from rag_agent.vectorstore.vector_store import VectorStore

DATASET = "eval/golden/v1"
REFUSAL = "The documents do not mention this."


class ScriptedLLM:
    """Answers the questions it knows (matched in the prompt), refuses the others."""

    model_path = "models/scripted.gguf"
    is_loaded = True
    prompt_budget = 3000

    def __init__(self, answers):
        self.answers = answers

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt):
        # La question posée figure en fin de prompt, après le contexte
        question = prompt.rsplit("Question:", 1)[-1]
        return next((a for q, a in self.answers.items() if q in question), REFUSAL)

    def stream(self, prompt):
        yield self.generate(prompt)


class JudgeLLM:
    def __init__(self, faithfulness="SUPPORTED", correctness="CORRECT"):
        self.faithfulness, self.correctness = faithfulness, correctness

    def generate(self, prompt):
        return self.correctness if "Reference answer:" in prompt else self.faithfulness


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET)


@pytest.fixture
def make_agent(offline_embeddings):
    def make(answers):
        retriever = Retriever(vector_store=VectorStore(path=None, index_type="brute-force"))
        return RAGAgent(llm=ScriptedLLM(answers), history_turns=0, min_relevance_score=0.0, retriever=retriever)

    return make


def by_id(dataset, qid):
    return next(q for q in dataset.questions if q.id == qid)


def test_the_corpus_goes_to_an_in_memory_store(make_agent, dataset, vector_store_dir):
    agent = make_agent({})

    n_chunks = harness.ingest_corpus(agent, dataset)

    assert n_chunks == len(agent.retriever.vector_store) > len(dataset.corpus)
    assert {c["source"] for c in agent.retriever.vector_store.chunks} == set(dataset.corpus)
    assert not vector_store_dir.exists()


def test_a_grounded_answer_scores_well(make_agent, dataset):
    q = by_id(dataset, "q02")
    agent = make_agent({q.question: "Lumen Rail employs 1,240 people, 410 of them drivers and conductors."})
    harness.ingest_corpus(agent, dataset)

    result = harness.evaluate_question(agent, q, top_k=5, judge=None)

    assert result.evidence_recall == 1.0
    assert result.reciprocal_rank > 0
    assert not result.refused
    assert not result.hallucination_rules
    assert result.unsupported_numbers == []
    assert result.reference_overlap == 1.0


def test_an_invented_figure_is_flagged(make_agent, dataset):
    q = by_id(dataset, "q02")
    agent = make_agent({q.question: "Lumen Rail employs 3,975 people."})
    harness.ingest_corpus(agent, dataset)

    result = harness.evaluate_question(agent, q, top_k=5, judge=None)

    assert result.hallucination_rules
    assert result.unsupported_numbers == ["3975"]


def test_answering_an_unanswerable_question_is_a_hallucination(make_agent, dataset):
    q = dataset.by_category("no_answer")[0]
    agent = make_agent({q.question: "It costs 12 euros."})
    harness.ingest_corpus(agent, dataset)

    assert harness.evaluate_question(agent, q, top_k=5, judge=None).hallucination_rules
    refusing = make_agent({})
    harness.ingest_corpus(refusing, dataset)
    result = harness.evaluate_question(refusing, q, top_k=5, judge=None)
    assert result.refused and not result.hallucination_rules


def test_the_judge_grades_answers_and_refusals(make_agent, dataset):
    answerable, unanswerable = by_id(dataset, "q02"), dataset.by_category("no_answer")[0]
    agent = make_agent({answerable.question: "Lumen Rail employs 1,240 people."})
    harness.ingest_corpus(agent, dataset)
    judge = LLMJudge(JudgeLLM(faithfulness="UNSUPPORTED", correctness="PARTIAL"))

    answered = harness.evaluate_question(agent, answerable, top_k=5, judge=judge)
    refused = harness.evaluate_question(agent, unanswerable, top_k=5, judge=judge)

    assert (answered.faithfulness, answered.correctness, answered.hallucination_judge) == (
        "unsupported",
        "partial",
        True,
    )
    assert refused.faithfulness is None and refused.correctness is None
    assert refused.hallucination_judge is False


def test_a_full_run_is_summarised_reported_and_compared(make_agent, dataset, tmp_path):
    answers = {q.question: q.reference_answer for q in dataset.questions if q.answerable}
    agent = make_agent(answers)
    harness.ingest_corpus(agent, dataset)
    seen = []

    results = harness.evaluate(
        agent, dataset, top_k=5, judge=LLMJudge(JudgeLLM()), progress=lambda i, r: seen.append(i)
    )
    summary = harness.summarize(results)
    meta = harness.run_metadata(dataset, "baseline", 5, agent.llm.model_path, "models/judge.gguf")
    run_dir = harness.write_report(str(tmp_path), meta, summary, results)

    assert seen == list(range(1, len(dataset.questions) + 1))
    assert summary["questions"] == 40
    assert summary["refusals"]["correct_refusal_rate"] == 1.0
    assert summary["judge"]["answer_correctness"] == 1.0
    assert set(summary["by_category"]) == {"factual", "multi_doc", "no_answer", "ambiguous"}
    report = json.loads((tmp_path / os.path.basename(run_dir) / harness.REPORT_JSON).read_text())
    assert report["meta"]["dataset"]["fingerprint"] == dataset.fingerprint
    assert len(report["results"]) == 40
    markdown = (tmp_path / os.path.basename(run_dir) / harness.REPORT_MD).read_text()
    assert "# RAG evaluation — baseline" in markdown
    assert "Answer correctness, judge | 100.0%" in markdown

    table = harness.compare([harness.load_report(run_dir), harness.load_report(run_dir)])
    assert table.startswith("| Metric | baseline | baseline |")
    assert "| Correct refusals | 100.0% | 100.0% |" in table


def test_runs_on_different_datasets_are_not_compared():
    report = {"meta": {"label": "a", "dataset": {"fingerprint": "aaa"}}, "summary": {}}
    other = {"meta": {"label": "b", "dataset": {"fingerprint": "bbb"}}, "summary": {}}

    with pytest.raises(ValueError, match="not comparable"):
        harness.compare([report, other])


def test_the_judge_section_is_absent_without_a_judge(make_agent, dataset):
    agent = make_agent({})
    harness.ingest_corpus(agent, dataset)

    unanswerable = dataclasses.replace(dataset, questions=tuple(dataset.by_category("no_answer")))

    summary = harness.summarize(harness.evaluate(agent, unanswerable, top_k=3))

    assert "judge" not in summary
    assert summary["refusals"]["false_refusal_rate"] is None
