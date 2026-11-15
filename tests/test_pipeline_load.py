import argparse
import json

import pytest

from benchmarks import pipeline_load
from rag_agent.generation.rag_agent import RAGAgent
from rag_agent.vectorstore.retriever import Retriever
from rag_agent.vectorstore.vector_store import VectorStore


def test_latency_distribution_keeps_tail_and_first_request():
    result = pipeline_load.latency_summary([0.1, 0.2, 0.3, 0.4, 10.0])
    assert result["p50_seconds"] == 0.3
    assert result["p95_seconds"] == pytest.approx(8.08)
    assert result["max_seconds"] == 10.0


def test_empty_or_malformed_questions_are_rejected(tmp_path):
    questions = tmp_path / "questions.jsonl"
    for text in ["", '{"question": ""}\n', '{"question": 2}\n', '"question"\n']:
        questions.write_text(text)
        with pytest.raises(ValueError):
            pipeline_load.load_questions(questions)


@pytest.mark.parametrize("minimum,expected_status", [(100_000, "failed"), (1, "completed")])
def test_real_pipeline_report_cannot_claim_an_unreached_scale(
    tmp_path, monkeypatch, offline_embeddings, stub_llm, sample_markdown, minimum, expected_status
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    sample_markdown.rename(corpus / "tower.md")
    questions = tmp_path / "questions.jsonl"
    questions.write_text('{"question": "Who built the Eiffel Tower?"}\n')
    agent = RAGAgent(
        llm=stub_llm,
        retriever=Retriever(vector_store=VectorStore(path=None)),
        agentic=False,
        query_rewriting=False,
    )
    monkeypatch.setattr(pipeline_load, "RAGAgent", lambda **kwargs: agent)
    args = argparse.Namespace(
        corpus=corpus,
        questions=questions,
        output=tmp_path / "run.json",
        min_chunks=minimum,
        repeats=2,
        top_k=3,
        no_agentic=True,
    )
    if expected_status == "failed":
        with pytest.raises(ValueError, match="distinct chunks"):
            pipeline_load.run(args)
    else:
        pipeline_load.run(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == expected_status
    assert report["indexed_chunks"] == 1
    assert report["corpus_files"][0]["sha256"]
    assert report["peak_rss_mb"] > 0
    if expected_status == "failed":
        assert report["queries"] == []
        assert "latency" not in report
    else:
        assert len(report["queries"]) == 2
        assert report["answered_requests"] == 2
        with pytest.raises(ValueError, match="preserve the previous run"):
            pipeline_load.run(args)
