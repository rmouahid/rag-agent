import numpy as np

from rag_agent.vectorstore.reranker import CrossEncoderReranker


class WordOverlapScorer:
    """Stand-in cross-encoder: counts query words found in the chunk."""

    def __init__(self):
        self.calls = []

    def predict(self, pairs, batch_size=32):
        self.calls.append((pairs, batch_size))
        return np.array([len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs], dtype=float)


def chunk(text, score):
    return {"id": text, "source": "doc.md", "text": text, "score": score}


def test_chunks_are_sorted_by_cross_encoder_score():
    reranker = CrossEncoderReranker(model=WordOverlapScorer())
    chunks = [chunk("trains run late", 0.9), chunk("the last train leaves at night", 0.5), chunk("bikes", 0.7)]

    reranked = reranker.rerank("when does the last train leave", chunks)

    assert [c["text"] for c in reranked] == ["the last train leaves at night", "trains run late", "bikes"]
    assert [c["rerank_score"] for c in reranked] == [3.0, 0.0, 0.0]


def test_the_cosine_score_is_kept_for_the_relevance_filter():
    reranked = CrossEncoderReranker(model=WordOverlapScorer()).rerank("last train", [chunk("last train", 0.42)])

    assert reranked[0]["score"] == 0.42


def test_ties_keep_the_vector_search_order():
    chunks = [chunk("alpha", 0.9), chunk("beta", 0.8), chunk("gamma", 0.7)]

    reranked = CrossEncoderReranker(model=WordOverlapScorer()).rerank("unrelated", chunks)

    assert [c["text"] for c in reranked] == ["alpha", "beta", "gamma"]


def test_pairs_are_scored_in_batches_with_the_query_first():
    scorer = WordOverlapScorer()
    CrossEncoderReranker(model=scorer, batch_size=4).rerank("q", [chunk("a", 0.1), chunk("b", 0.2)])

    assert scorer.calls == [([("q", "a"), ("q", "b")], 4)]


def test_no_candidate_needs_no_model_call():
    scorer = WordOverlapScorer()

    assert CrossEncoderReranker(model=scorer).rerank("q", []) == []
    assert scorer.calls == []


def test_the_input_chunks_are_not_modified():
    chunks = [chunk("last train", 0.5)]
    CrossEncoderReranker(model=WordOverlapScorer()).rerank("last train", chunks)

    assert "rerank_score" not in chunks[0]
