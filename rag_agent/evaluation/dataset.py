"""
Jeu de questions de référence (golden dataset) pour évaluer le RAG.

Un jeu est un dossier versionné :

    manifest.json     nom, version, liste des documents du corpus
    questions.jsonl   une question par ligne (voir GoldenQuestion)
    corpus/           les documents à ingérer avant de poser les questions

Les sources attendues sont désignées par des citations exactes de leur
document (`Evidence.quote`), pas par des ids de chunks : elles restent
valables quand on change le découpage ou le modèle d'embedding, ce qu'une
évaluation doit justement permettre de comparer.

Toute modification des questions ou du corpus doit créer une nouvelle
version (nouveau dossier) : l'empreinte `fingerprint` figure dans chaque
rapport d'évaluation, et deux runs ne sont comparables que si elle est
identique.
"""

import hashlib
import json
import os
from dataclasses import dataclass

CATEGORIES = ("factual", "multi_doc", "no_answer", "ambiguous")
MANIFEST_FILE = "manifest.json"
CORPUS_DIR = "corpus"


class DatasetError(ValueError):
    """Jeu de référence mal formé"""


def normalize_text(text: str) -> str:
    """Espaces normalisés et minuscules, pour chercher une citation dans un chunk"""
    return " ".join(text.split()).lower()


@dataclass(frozen=True)
class Evidence:
    """Passage qui justifie la réponse : `quote` figure mot pour mot dans `source`"""

    source: str
    quote: str

    def found_in(self, text: str) -> bool:
        return normalize_text(self.quote) in normalize_text(text)


@dataclass(frozen=True)
class GoldenQuestion:
    """
    id               : identifiant stable, repris dans les rapports
    category         : factual (un passage), multi_doc (plusieurs documents),
                       no_answer (absente du corpus : le RAG doit le dire),
                       ambiguous (plusieurs lectures : la réponse doit les couvrir)
    reference_answer : réponse attendue, rédigée à la main
    evidence         : passages attendus parmi les sources ; vide pour no_answer
    """

    id: str
    category: str
    question: str
    reference_answer: str
    evidence: tuple[Evidence, ...]

    @property
    def answerable(self) -> bool:
        return self.category != "no_answer"


@dataclass(frozen=True)
class GoldenDataset:
    name: str
    version: str
    path: str
    corpus: tuple[str, ...]
    questions: tuple[GoldenQuestion, ...]
    fingerprint: str

    @property
    def corpus_paths(self) -> list[str]:
        return [os.path.join(self.path, CORPUS_DIR, name) for name in self.corpus]

    def by_category(self, category: str) -> list[GoldenQuestion]:
        return [q for q in self.questions if q.category == category]


def _parse_question(line: str, number: int) -> GoldenQuestion:
    try:
        raw = json.loads(line)
        return GoldenQuestion(
            id=raw["id"],
            category=raw["category"],
            question=raw["question"],
            reference_answer=raw["reference_answer"],
            evidence=tuple(Evidence(source=e["source"], quote=e["quote"]) for e in raw["evidence"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise DatasetError(f"questions line {number}: {e!r}") from e


def _validate(questions: list[GoldenQuestion], corpus_text: dict[str, str]) -> None:
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise DatasetError(f"duplicate question id {q.id!r}")
        seen.add(q.id)
        if q.category not in CATEGORIES:
            raise DatasetError(f"{q.id}: unknown category {q.category!r} (expected one of {CATEGORIES})")
        if q.answerable and not q.evidence:
            raise DatasetError(f"{q.id}: an answerable question needs at least one evidence passage")
        if not q.answerable and q.evidence:
            raise DatasetError(f"{q.id}: a no_answer question cannot have evidence")
        for e in q.evidence:
            if e.source not in corpus_text:
                raise DatasetError(f"{q.id}: evidence source {e.source!r} is not in the corpus")
            if not e.found_in(corpus_text[e.source]):
                raise DatasetError(f"{q.id}: quote not found in {e.source}: {e.quote!r}")


def load_dataset(path: str) -> GoldenDataset:
    """
    Charge et valide un jeu : ids uniques, catégories connues, et chaque
    citation présente dans le document qu'elle désigne
    """
    with open(os.path.join(path, MANIFEST_FILE), encoding="utf-8") as f:
        manifest = json.load(f)

    digest = hashlib.sha256()
    corpus_text = {}
    for name in manifest["corpus"]:
        with open(os.path.join(path, CORPUS_DIR, name), "rb") as f:
            content = f.read()
        digest.update(name.encode() + b"\0" + content)
        corpus_text[name] = content.decode("utf-8")

    with open(os.path.join(path, manifest["questions"]), "rb") as f:
        raw_questions = f.read()
    digest.update(raw_questions)
    lines = raw_questions.decode("utf-8").splitlines()
    questions = [_parse_question(line, i) for i, line in enumerate(lines, start=1) if line.strip()]
    _validate(questions, corpus_text)

    return GoldenDataset(
        name=manifest["name"],
        version=manifest["version"],
        path=path,
        corpus=tuple(manifest["corpus"]),
        questions=tuple(questions),
        fingerprint=digest.hexdigest()[:12],
    )
