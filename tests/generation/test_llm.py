import pytest

from rag_agent.generation import llm as llm_module
from rag_agent.generation.llm import LocalLLM, ModelNotFoundError


class FakeLlama:
    """Stand-in for llama_cpp.Llama: records its settings and the chat calls."""

    instances = []

    def __init__(self, model_path, n_ctx, n_threads, verbose):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.calls = []
        FakeLlama.instances.append(self)

    def tokenize(self, text, add_bos=True, special=False):
        return [1] + list(range(len(text.split())))

    def create_chat_completion(self, messages, max_tokens, temperature, stream=False):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "temperature": temperature})
        if stream:
            # Same shape as llama-cpp-python's streamed chunks: a role delta,
            # content deltas, then an empty delta with the finish reason
            return iter([
                {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": " \n"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": " Par"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "is."}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ])
        return {"choices": [{"message": {"role": "assistant", "content": "  Paris.\n"}}]}


@pytest.fixture(autouse=True)
def fake_llama(monkeypatch):
    FakeLlama.instances = []
    monkeypatch.setattr(llm_module, "Llama", FakeLlama)


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")
    return str(path)


def test_loads_the_weights_with_the_given_settings(weights):
    LocalLLM(model_path=weights, context_size=2048, threads=2)

    (model,) = FakeLlama.instances
    assert model.model_path == weights
    assert model.n_ctx == 2048
    assert model.n_threads == 2


def test_zero_threads_lets_llama_cpp_decide(weights):
    LocalLLM(model_path=weights, threads=0)

    assert FakeLlama.instances[0].n_threads is None


def test_settings_default_to_config(weights, monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "config",
        llm_module.config.__class__(
            local_model_path=weights, llm_context_size=1024, llm_max_tokens=64, llm_temperature=0.5, llm_threads=3,
            llm_context_margin=32,
        ),
    )

    llm = LocalLLM()

    assert (llm.model_path, llm.context_size, llm.max_tokens, llm.temperature) == (weights, 1024, 64, 0.5)
    assert llm.context_margin == 32
    assert FakeLlama.instances[0].n_threads == 3


def test_generate_sends_the_prompt_as_a_chat_message(weights):
    llm = LocalLLM(model_path=weights, max_tokens=128, temperature=0.0)

    answer = llm.generate("What is the capital of France?")

    assert answer == "Paris."
    (call,) = FakeLlama.instances[0].calls
    assert call == {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 128,
        "temperature": 0.0,
    }


def test_missing_weights_raise_a_clear_error(tmp_path):
    missing = str(tmp_path / "nope.gguf")

    with pytest.raises(ModelNotFoundError) as excinfo:
        LocalLLM(model_path=missing)

    message = str(excinfo.value)
    assert missing in message
    assert "hf download" in message and "LOCAL_MODEL_PATH" in message
    assert FakeLlama.instances == []


def test_unset_model_path_raises_a_clear_error():
    with pytest.raises(ModelNotFoundError, match="LOCAL_MODEL_PATH is not set"):
        LocalLLM(model_path="")


def test_a_directory_is_not_a_model(tmp_path):
    with pytest.raises(ModelNotFoundError):
        LocalLLM(model_path=str(tmp_path))


def test_prompt_budget_leaves_room_for_the_answer_and_the_chat_template(weights):
    llm = LocalLLM(model_path=weights, context_size=4096, max_tokens=512, context_margin=64)

    assert llm.prompt_budget == 4096 - 512 - 64


def test_count_tokens_uses_the_model_tokenizer(weights):
    llm = LocalLLM(model_path=weights)

    # FakeLlama: one token per word plus BOS
    assert llm.count_tokens("three little words") == 4


def test_stream_yields_the_text_as_it_is_generated_without_leading_whitespace(weights):
    llm = LocalLLM(model_path=weights, max_tokens=128, temperature=0.0)

    pieces = list(llm.stream("What is the capital of France?"))

    assert pieces == ["Par", "is."]
    (call,) = FakeLlama.instances[0].calls
    assert call["messages"] == [{"role": "user", "content": "What is the capital of France?"}]
    assert call["max_tokens"] == 128


def test_stream_is_lazy(weights):
    llm = LocalLLM(model_path=weights)

    llm.stream("prompt")

    assert FakeLlama.instances[0].calls == []


def test_is_loaded_once_the_weights_are_in_memory(weights):
    llm = LocalLLM(model_path=weights)
    assert llm.is_loaded

    llm._model = None
    assert not llm.is_loaded
