import json
import os
import shutil

import pytest

from rag_agent.chunking.factory import get_chunker
from rag_agent.evaluation.dataset import CATEGORIES, DatasetError, load_dataset
from rag_agent.ingestion.document import Document

# Questions whose evidence is in lumen_miles.md
MILES = ("q23", "q24", "q27", "q39")
GOLDEN = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "eval", "golden", "v1")


@pytest.fixture
def dataset_copy(tmp_path):
    """Writable copy of the golden dataset, to corrupt it"""
    path = tmp_path / "golden"
    shutil.copytree(GOLDEN, path)
    return path


def rewrite_questions(path, edit):
    questions_file = path / "questions.jsonl"
    questions = [json.loads(line) for line in questions_file.read_text(encoding="utf-8").splitlines()]
    edit(questions)
    questions_file.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")


# --- the published dataset --------------------------------------------------


def test_golden_dataset_loads_and_is_valid():
    dataset = load_dataset(GOLDEN)

    assert dataset.name == "lumen-rail"
    assert dataset.version == "1.0.0"
    assert len(dataset.questions) == 40
    assert all(os.path.isfile(p) for p in dataset.corpus_paths)


def test_golden_dataset_covers_every_category():
    dataset = load_dataset(GOLDEN)

    counts = {c: len(dataset.by_category(c)) for c in CATEGORIES}

    assert counts == {"factual": 24, "multi_doc": 6, "no_answer": 5, "ambiguous": 5}


def test_multi_doc_questions_need_several_documents():
    for q in load_dataset(GOLDEN).by_category("multi_doc"):
        assert len({e.source for e in q.evidence}) >= 2, q.id


@pytest.mark.parametrize("strategy", ["fixed", "sentence"])
def test_every_quote_fits_in_one_chunk(strategy):
    """Otherwise retrieval could never be credited with finding it"""
    dataset = load_dataset(GOLDEN)
    chunker = get_chunker(strategy=strategy, chunk_size=500, overlap=50)
    chunks = {}
    for name, path in zip(dataset.corpus, dataset.corpus_paths):
        with open(path, encoding="utf-8") as f:
            chunks[name] = [c.text for c in chunker.chunk(Document(text=f.read(), source=name, metadata={}))]

    for q in dataset.questions:
        for e in q.evidence:
            assert any(e.found_in(text) for text in chunks[e.source]), f"{q.id}: {e.quote!r} spans two chunks"


def test_fingerprint_is_stable_and_tracks_changes(dataset_copy):
    original = load_dataset(GOLDEN).fingerprint
    assert load_dataset(str(dataset_copy)).fingerprint == original

    (dataset_copy / "corpus" / "lumen_miles.md").write_text("# Changed\n", encoding="utf-8")

    def drop_miles_questions(questions):
        questions[:] = [q for q in questions if q["id"] not in MILES]

    rewrite_questions(dataset_copy, drop_miles_questions)

    assert load_dataset(str(dataset_copy)).fingerprint != original


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda qs: qs.append(dict(qs[0])), "duplicate question id"),
        (lambda qs: qs[0].update(category="trivia"), "unknown category"),
        (lambda qs: qs[0].update(evidence=[]), "needs at least one evidence"),
        (lambda qs: qs[30].update(evidence=qs[0]["evidence"]), "cannot have evidence"),
        (lambda qs: qs[0]["evidence"][0].update(source="missing.md"), "not in the corpus"),
        (lambda qs: qs[0]["evidence"][0].update(quote="founded in 1999"), "quote not found"),
        (lambda qs: qs[0].pop("question"), "line 1"),
    ],
)
def test_invalid_datasets_are_rejected(dataset_copy, edit, message):
    rewrite_questions(dataset_copy, edit)

    with pytest.raises(DatasetError, match=message):
        load_dataset(str(dataset_copy))
