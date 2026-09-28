import os

from llama_cpp import Llama

from rag_agent.config import config
from rag_agent.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"


class ModelNotFoundError(FileNotFoundError):
    """Les poids GGUF du modèle local sont introuvables."""


def _missing_model_message(path: str) -> str:
    where = f"No model weights found at {path!r}." if path else "LOCAL_MODEL_PATH is not set."
    return (
        f"{where}\n"
        "Download the default GGUF weights (see docs/ADR/0001-local-llm-runtime.md):\n"
        f"  poetry run hf download {DEFAULT_MODEL_REPO} {DEFAULT_MODEL_FILE} --local-dir models\n"
        f"then set LOCAL_MODEL_PATH=models/{DEFAULT_MODEL_FILE} in .env."
    )


class LocalLLM:
    """
    Modèle de langage exécuté localement via llama.cpp (ADR 0001).

    Les poids sont chargés une seule fois, à la construction ; un fichier
    manquant lève `ModelNotFoundError` avec la marche à suivre pour les
    obtenir, plutôt qu'une erreur obscure de llama.cpp.
    """

    def __init__(
        self,
        model_path: str = None,
        context_size: int = None,
        max_tokens: int = None,
        temperature: float = None,
        threads: int = None,
        context_margin: int = None,
    ):
        self.model_path = config.local_model_path if model_path is None else model_path
        self.context_size = context_size or config.llm_context_size
        self.max_tokens = max_tokens or config.llm_max_tokens
        self.temperature = config.llm_temperature if temperature is None else temperature
        self.context_margin = config.llm_context_margin if context_margin is None else context_margin
        threads = config.llm_threads if threads is None else threads

        if not self.model_path or not os.path.isfile(self.model_path):
            raise ModelNotFoundError(_missing_model_message(self.model_path))

        logger.info("Chargement du modèle local %s (n_ctx=%d)", self.model_path, self.context_size)
        self._model = Llama(
            model_path=self.model_path,
            n_ctx=self.context_size,
            # 0 = laisser llama.cpp choisir selon le nombre de cœurs
            n_threads=threads or None,
            verbose=False,
        )

    @property
    def prompt_budget(self) -> int:
        """
        Tokens disponibles pour le prompt : la fenêtre de contexte moins la
        réponse attendue et une marge pour le template de chat
        """
        return self.context_size - self.max_tokens - self.context_margin

    def count_tokens(self, text: str) -> int:
        """
        Nombre de tokens de `text` selon le tokenizer du modèle chargé
        """
        return len(self._model.tokenize(text.encode("utf-8")))

    def generate(self, prompt: str) -> str:
        """
        Génère une réponse au prompt, formaté avec le template de chat
        embarqué dans le fichier GGUF.
        """
        response = self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response["choices"][0]["message"]["content"].strip()
