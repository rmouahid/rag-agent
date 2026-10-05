import pytest

from rag_agent import cli
from rag_agent.generation.llm import ModelNotFoundError
from rag_agent.generation.rag_agent import RAGAgent


@pytest.fixture
def agent_with_stub_llm(monkeypatch, vector_store_dir, stub_llm):
    monkeypatch.setattr(cli, "RAGAgent", lambda: RAGAgent(llm=stub_llm, min_relevance_score=0.25))
    return stub_llm


def test_ingest_indexes_a_file_into_the_persisted_store(vector_store_dir, sample_markdown, capsys):
    assert cli.main(["ingest", str(sample_markdown)]) == 0

    out = capsys.readouterr().out
    assert f"{sample_markdown}: " in out
    assert "chunks indexed" in out
    assert (vector_store_dir / "chunks.json").exists()


def test_ingest_walks_a_directory_for_supported_files(vector_store_dir, sample_markdown, sample_pdf, capsys):
    (sample_markdown.parent / "notes.txt").write_text("ignored")

    assert cli.main(["ingest", str(sample_markdown.parent)]) == 0

    out = capsys.readouterr().out
    assert str(sample_markdown) in out
    assert str(sample_pdf) in out
    assert "notes.txt" not in out


def test_reingesting_reports_duplicates(vector_store_dir, sample_markdown, capsys):
    cli.main(["ingest", str(sample_markdown)])
    capsys.readouterr()

    assert cli.main(["ingest", str(sample_markdown)]) == 0
    assert "0 chunks indexed" in capsys.readouterr().out


def test_ingest_reports_failures_and_carries_on(vector_store_dir, sample_markdown, tmp_path, capsys):
    unsupported = tmp_path / "notes.docx"
    unsupported.write_text("x")

    assert cli.main(["ingest", str(unsupported), str(sample_markdown)]) == 1

    captured = capsys.readouterr()
    assert "notes.docx" in captured.err
    assert "chunks indexed" in captured.out


def test_ingest_of_a_missing_path_fails(vector_store_dir, tmp_path, capsys):
    assert cli.main(["ingest", str(tmp_path / "missing.md")]) == 1
    assert "missing.md" in capsys.readouterr().err


def test_ingest_without_a_persisted_store_is_refused(offline_embeddings, monkeypatch, sample_markdown, capsys):
    import dataclasses

    from rag_agent.vectorstore import retriever as retriever_module

    monkeypatch.setattr(retriever_module, "config", dataclasses.replace(retriever_module.config, vector_store_path=""))

    assert cli.main(["ingest", str(sample_markdown)]) == 1
    assert "VECTOR_STORE_PATH" in capsys.readouterr().err


def test_ask_streams_the_answer_and_lists_the_sources(agent_with_stub_llm, sample_markdown, capsys):
    cli.main(["ingest", str(sample_markdown)])
    capsys.readouterr()

    assert cli.main(["ask", "Who built the Eiffel Tower?"]) == 0

    out = capsys.readouterr().out
    assert "Gustave Eiffel's company." in out
    assert "Sources" in out
    assert str(sample_markdown) in out
    assert "Who built the Eiffel Tower?" in agent_with_stub_llm.prompts[0]


def test_ask_passes_top_k(agent_with_stub_llm, sample_markdown, monkeypatch):
    cli.main(["ingest", str(sample_markdown)])
    calls = []
    original = RAGAgent.ask_stream

    def spy(self, question, top_k=5, history=None):
        calls.append(top_k)
        return original(self, question, top_k=top_k, history=history)

    monkeypatch.setattr(RAGAgent, "ask_stream", spy)

    cli.main(["ask", "--top-k", "2", "Who built the Eiffel Tower?"])

    assert calls == [2]


def test_ask_without_documents_says_so(agent_with_stub_llm, capsys):
    assert cli.main(["ask", "Who built the Eiffel Tower?"]) == 0

    assert "Aucun document" in capsys.readouterr().out
    assert agent_with_stub_llm.prompts == []


def test_ask_without_model_weights_explains_how_to_get_them(monkeypatch, vector_store_dir, capsys):
    def missing():
        raise ModelNotFoundError("Model not found: download it with hf download ...")

    monkeypatch.setattr(cli, "RAGAgent", missing)

    assert cli.main(["ask", "Who built the Eiffel Tower?"]) == 1
    assert "hf download" in capsys.readouterr().err


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2


def test_a_folder_without_supported_files_is_reported(vector_store_dir, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("ignored")

    assert cli.main(["ingest", str(empty)]) == 1
    assert "no .pdf, .md, .markdown file" in capsys.readouterr().err
