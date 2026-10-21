"""
Juge LLM : fait noter par un modèle ce que les règles (metrics.py) ne
voient pas, comme une paraphrase fidèle ou une réponse juste mais
formulée autrement que la référence.

Le juge ne reçoit qu'une décision fermée à rendre (un mot-clé en tête de
réponse), ce qui le rend utilisable avec un petit modèle local et
analysable sans ambiguïté. Une réponse illisible donne un verdict None,
compté à part dans le rapport plutôt que deviné.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rag_agent.generation.truncation import shrink_to_budget

FAITHFULNESS_PROMPT = """You are checking an answer produced by a question-answering system.

Context given to the system:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer: {answer}

Is every factual claim of the answer supported by the context? Saying that the context does not contain the answer
counts as supported. Reply with one word, SUPPORTED or UNSUPPORTED, then a short reason."""

CORRECTNESS_PROMPT = """You are grading an answer against a reference answer written by a human.

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Does the answer give the same information as the reference? Reply with one word, CORRECT (all of it), PARTIAL (some
of it, or with an error) or INCORRECT (none of it, or a refusal), then a short reason."""

_FAITHFULNESS = re.compile(r"\b(UNSUPPORTED|SUPPORTED)\b", re.IGNORECASE)
_CORRECTNESS = re.compile(r"\b(INCORRECT|CORRECT|PARTIAL)\b", re.IGNORECASE)
CORRECTNESS_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


def _shrink(llm: TextGenerator, build: Callable[[list[str]], str], context: list[str]) -> str:
    """Prompt réduit au budget de tokens du LLM s'il l'expose"""
    count_tokens = getattr(llm, "count_tokens", None)
    budget = getattr(llm, "prompt_budget", None)
    if count_tokens is None or budget is None:
        return build(context)
    return shrink_to_budget(build, context, count_tokens, budget)


@dataclass(frozen=True)
class Verdict:
    label: str | None
    reason: str


def _verdict(text: str, pattern: re.Pattern[str]) -> Verdict:
    """Premier mot-clé reconnu dans la réponse du juge, None s'il n'y en a pas"""
    match = pattern.search(text)
    return Verdict(label=match.group(1).lower() if match else None, reason=" ".join(text.split())[:300])


class LLMJudge:
    def __init__(self, llm: TextGenerator, max_context_chars: int = 6000):
        self.llm = llm
        # Contexte tronqué : il doit tenir dans la fenêtre du modèle juge (au
        # budget de tokens du LLM s'il l'expose, et à `max_context_chars`)
        self.max_context_chars = max_context_chars

    def faithfulness(self, question: str, answer: str, context: list[str]) -> Verdict:
        """supported / unsupported : la réponse s'en tient-elle au contexte ?"""

        def build(kept: list[str]) -> str:
            joined = "\n---\n".join(kept)[: self.max_context_chars]
            return FAITHFULNESS_PROMPT.format(context=joined, question=question, answer=answer)

        return _verdict(self.llm.generate(_shrink(self.llm, build, context)), _FAITHFULNESS)

    def correctness(self, question: str, answer: str, reference: str) -> Verdict:
        """correct / partial / incorrect : la réponse rejoint-elle la référence ?"""
        prompt = CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer)
        return _verdict(self.llm.generate(prompt), _CORRECTNESS)
