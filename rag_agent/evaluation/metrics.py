"""
Métriques à base de règles : déterministes, sans LLM, donc gratuites et
reproductibles d'un run à l'autre. Elles restent lexicales (voir
docs/evaluation.md) : le juge LLM (judge.py) les complète quand on peut
se permettre son coût.
"""

import re

from rag_agent.evaluation.dataset import GoldenQuestion

# Mots vides anglais et français : ils n'apportent rien à un recouvrement lexical
STOPWORDS = frozenset("""
    a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    may more most must no not of on or our per should so than that the their them then there these they this
    to up was we were what when where which while who why will with within would you your yes also any each
    le la les un une des du de et est en au aux pour par sur dans que qui ne pas plus ou il elle ils sont
    """.split())

# Formulations d'un refus de répondre faute d'information
REFUSAL_PATTERNS = re.compile(
    r"\b(do(es)? not (say|mention|specify|contain|provide|include|state)|no information|not (mentioned|specified"
    r"|provided|stated|available|found)|cannot (find|answer|determine)|can't (find|answer|determine)"
    r"|(don't|do not) (know|have)|there is no mention|aucun passage|ne (le )?précise pas|pas d'information)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def content_words(text: str) -> set[str]:
    """Mots porteurs de sens : sans mots vides ni mots de moins de 3 lettres"""
    return {w for w in _WORD.findall(text.lower()) if len(w) >= 3 and w not in STOPWORDS and not w.isdigit()}


def numbers(text: str) -> set[str]:
    """Nombres du texte, séparateurs de milliers retirés (« 1,240 » = « 1240 »)"""
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def is_refusal(answer: str, answered: bool = True) -> bool:
    """
    Le RAG a décliné : soit sans appeler le LLM (aucun contexte pertinent),
    soit parce que le LLM a dit que les documents ne répondent pas
    """
    return not answered or bool(REFUSAL_PATTERNS.search(answer))


def evidence_recall(question: GoldenQuestion, context: list[str]) -> float:
    """Part des passages attendus présents dans le contexte (1.0 sans passage attendu)"""
    if not question.evidence:
        return 1.0
    found = sum(any(e.found_in(chunk) for chunk in context) for e in question.evidence)
    return found / len(question.evidence)


def reciprocal_rank(question: GoldenQuestion, context: list[str]) -> float:
    """1 / rang du premier chunk qui contient un passage attendu (0 s'il n'y en a aucun)"""
    for rank, chunk in enumerate(context, start=1):
        if any(e.found_in(chunk) for e in question.evidence):
            return 1.0 / rank
    return 0.0


def lexical_support(answer: str, context: list[str]) -> float:
    """
    Fidélité approchée : part des mots porteurs de sens de la réponse qui
    figurent dans le contexte. Un paraphrase fidèle peut être pénalisée ; un
    mot inventé, lui, l'est toujours.
    """
    words = content_words(answer)
    if not words:
        return 1.0
    available = content_words(" ".join(context))
    return len(words & available) / len(words)


def unsupported_numbers(answer: str, context: list[str]) -> set[str]:
    """
    Nombres de la réponse absents du contexte : le signal d'hallucination le
    plus fiable sur un corpus de faits chiffrés inventés
    """
    return numbers(answer) - numbers(" ".join(context))


def reference_overlap(answer: str, reference: str) -> float:
    """
    Pertinence approchée : part des mots et nombres de la réponse de
    référence que la réponse reprend
    """
    expected = content_words(reference) | numbers(reference)
    if not expected:
        return 1.0
    given = content_words(answer) | numbers(answer)
    return len(expected & given) / len(expected)


def is_hallucination_by_rules(answer: str, context: list[str], min_support: float = 0.5) -> bool:
    """
    Réponse non ancrée selon les règles : un nombre absent du contexte, ou
    moins de `min_support` de ses mots porteurs de sens présents dans le contexte
    """
    return bool(unsupported_numbers(answer, context)) or lexical_support(answer, context) < min_support
