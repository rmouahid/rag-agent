import pytest

from rag_agent.ingestion.document import Document
from rag_agent.vectorstore.retriever import Retriever
from rag_agent.vectorstore.vector_store import VectorStore

TOPICS = {
    "trains": "The last Amber Line train leaves Castelmont at 23:10 every evening.",
    "bikes": "Bicycles travel free on regional trains outside peak hours.",
    "dogs": "Dogs travel with a reduced ticket costing three euros.",
    "refunds": "Refunds are paid within fourteen days to the original card.",
    "miles": "Lumen Miles members earn one mile per euro spent on tickets.",
}


@pytest.fixture
def retriever(offline_embeddings):
    retriever = Retriever(chunk_size=500, overlap=0, vector_store=VectorStore(path=None, index_type="brute-force"))
    for name, text in TOPICS.items():
        retriever.ingest(Document(text=text, source=f"{name}.md", metadata={"type": "markdown"}))
    return retriever


def sources(chunks):
    return [c["source"] for c in chunks]


def test_a_single_query_is_a_plain_retrieval(retriever):
    assert retriever.retrieve_many(["last train Castelmont"], top_k=2) == retriever.retrieve(
        "last train Castelmont", top_k=2
    )


def test_each_query_contributes_its_best_chunk(retriever):
    chunks = retriever.retrieve_many(["last Amber Line train", "bicycles regional trains", "dogs ticket"], top_k=3)

    assert sources(chunks) == ["trains.md", "bikes.md", "dogs.md"]


def test_results_are_capped_at_top_k_and_deduplicated(retriever):
    chunks = retriever.retrieve_many(["last Amber Line train", "Amber Line train evening", "refunds card"], top_k=4)

    assert len(chunks) == 4
    assert len(set(sources(chunks))) == 4
    assert sources(chunks)[:2] == ["trains.md", "refunds.md"]


def test_a_chunk_found_by_several_queries_keeps_its_best_score(retriever):
    weak = retriever.retrieve("Castelmont", top_k=1)[0]["score"]
    strong = retriever.retrieve("last Amber Line train leaves Castelmont", top_k=1)[0]["score"]

    (chunk,) = retriever.retrieve_many(["Castelmont", "last Amber Line train leaves Castelmont"], top_k=1)

    assert chunk["source"] == "trains.md"
    assert chunk["score"] == pytest.approx(max(weak, strong))


def test_no_query_retrieves_nothing(retriever):
    assert retriever.retrieve_many([], top_k=3) == []
