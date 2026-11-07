import json
import threading

import pytest

from rag_agent.api.dependencies import agent_lock
from rag_agent.generation.answer import Source, StreamingAnswer
from rag_agent.generation.truncation import ContextOverflowError


def parse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def streamed(agent):
    """Records the calls and streams like a grounded RAGAgent."""
    calls = []

    def ask_stream(question, top_k=5, history=None):
        calls.append({"question": question, "top_k": top_k, "history": history})
        return StreamingAnswer(
            tokens=iter([" \n", "Par", "is.\n", "Second line."]),
            sources=[Source(source="france.md", excerpt="Paris is the capital.", chunk_id="c1", score=0.82)],
        )

    agent.ask_stream = ask_stream
    return calls


def test_tokens_are_streamed_as_server_sent_events(client, streamed):
    response = client.post("/ask/stream", json={"question": "Capital of France?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert parse_events(response.text) == [
        (
            "sources",
            {
                "status": "answered",
                "sources": [
                    {"source": "france.md", "excerpt": "Paris is the capital.", "chunk_id": "c1", "score": 0.82}
                ],
            },
        ),
        ("token", {"text": "Par"}),
        ("token", {"text": "is.\n"}),
        ("token", {"text": "Second line."}),
        ("done", {"answer": "Paris.\nSecond line."}),
    ]


def test_each_event_holds_on_one_data_line(client, streamed):
    body = client.post("/ask/stream", json={"question": "q"}).text

    for block in body.strip().split("\n\n"):
        assert len(block.splitlines()) == 2


def test_post_passes_question_top_k_and_history(client, streamed):
    history = [{"role": "user", "content": "Who?"}, {"role": "assistant", "content": "Hugo."}]

    client.post("/ask/stream", json={"question": " When? ", "top_k": 3, "history": history})

    assert streamed == [{"question": "When?", "top_k": 3, "history": history}]


def test_get_variant_for_event_source(client, streamed):
    response = client.get("/ask/stream", params={"question": "Capital?", "top_k": 2})

    assert response.status_code == 200
    assert parse_events(response.text)[-1] == ("done", {"answer": "Paris.\nSecond line."})
    assert streamed == [{"question": "Capital?", "top_k": 2, "history": []}]


@pytest.mark.parametrize("params", [{}, {"question": ""}, {"question": "q", "top_k": 0}])
def test_get_variant_validates_its_parameters(client, streamed, params):
    response = client.get("/ask/stream", params=params)

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert streamed == []


def test_ungrounded_questions_stream_the_fixed_message(client, agent):
    agent.ask_stream = lambda question, top_k=5, history=None: StreamingAnswer(
        tokens=iter([" Nothing relevant."]), status="no_relevant_context"
    )

    events = parse_events(client.post("/ask/stream", json={"question": "q"}).text)

    assert events == [
        ("sources", {"status": "no_relevant_context", "sources": []}),
        ("token", {"text": "Nothing relevant."}),
        ("done", {"answer": "Nothing relevant."}),
    ]


def test_prompt_overflow_ends_the_stream_with_an_error_event(client, agent):
    def ask_stream(question, top_k=5, history=None):
        raise ContextOverflowError("The prompt needs 5000 tokens")

    agent.ask_stream = ask_stream

    events = parse_events(client.post("/ask/stream", json={"question": "q"}).text)

    assert events == [("error", {"error": "context_overflow", "detail": "The prompt needs 5000 tokens"})]


def test_a_failure_mid_generation_ends_the_stream_with_an_error_event(client, agent):
    def tokens():
        yield "Par"
        raise RuntimeError("llama_decode returned -3")

    agent.ask_stream = lambda question, top_k=5, history=None: StreamingAnswer(tokens=tokens())

    events = parse_events(client.post("/ask/stream", json={"question": "q"}).text)

    assert [e for e, _ in events] == ["sources", "token", "error"]
    assert events[-1][1] == {"error": "generation_failed", "detail": "The model failed to generate an answer."}


def test_the_agent_lock_is_held_during_generation_and_released_after(client, agent):
    held = []

    def tokens():
        held.append(agent_lock.locked())
        yield "ok"

    agent.ask_stream = lambda question, top_k=5, history=None: StreamingAnswer(tokens=tokens())

    client.post("/ask/stream", json={"question": "q"})

    assert held == [True]
    assert not agent_lock.locked()


def test_the_lock_is_released_when_the_client_disconnects(client, agent):
    closed = threading.Event()

    def tokens():
        try:
            for i in range(1000):
                yield f"t{i} "
        finally:
            closed.set()

    agent.ask_stream = lambda question, top_k=5, history=None: StreamingAnswer(tokens=tokens())

    with client.stream("POST", "/ask/stream", json={"question": "q"}) as response:
        next(response.iter_lines())

    assert closed.wait(5)
    assert not agent_lock.locked()


def test_streaming_routes_are_documented(client):
    paths = client.get("/openapi.json").json()["paths"]

    for method in ("get", "post"):
        assert "text/event-stream" in paths["/ask/stream"][method]["responses"]["200"]["content"]
