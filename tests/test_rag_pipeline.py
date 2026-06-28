import pickle
from unittest.mock import patch

import faiss
import pytest

from rag_pipeline import RagPipeline
from tests.conftest import MockEmbeddings


@pytest.fixture
def pipeline():
    return RagPipeline(embedding_model=MockEmbeddings())


def test_add_documents_keeps_index_aligned(pipeline):
    count = pipeline.add_documents(["chunk one", "chunk two", ""], source_name="doc.pdf")
    assert count == 2
    assert len(pipeline.documents) == 2
    assert pipeline.index.ntotal == 2


def test_retrieve_returns_matching_docs(pipeline):
    pipeline.add_documents(
        ["مبلغ قرارداد یک میلیون تومان", "تاریخ امضا ۱۴۰۳"],
        source_name="contract.pdf",
    )
    result, sources = pipeline.retrieve("مبلغ", return_sources=True)
    assert "مبلغ" in result
    assert "contract.pdf" in sources


def test_delete_document_rebuilds_index(pipeline):
    pipeline.add_documents(["a", "b"], source_name="keep.pdf")
    pipeline.add_documents(["c"], source_name="remove.pdf")
    assert pipeline.delete_document("remove.pdf") is True
    assert len(pipeline.documents) == 2
    assert pipeline.index.ntotal == 2
    assert "remove.pdf" not in pipeline.document_sources


def test_load_rebuilds_mismatched_legacy_index(pipeline, tmp_path):
    pipeline.add_documents(["only doc"], source_name="legacy.pdf")
    path = tmp_path / "vector_db.pkl"

    # Simulate legacy bug: index has 2 rows but documents has 1
    bad_index = faiss.IndexFlatL2(8)
    bad_index.add(
        __import__("numpy").array([[0.1] * 8, [0.2] * 8], dtype="float32")
    )
    with open(path, "wb") as f:
        pickle.dump({
            "index": faiss.serialize_index(bad_index),
            "documents": ["only doc"],
            "document_sources": ["legacy.pdf"],
        }, f)

    loaded = RagPipeline(embedding_model=MockEmbeddings())
    with patch.object(loaded, "_rebuild_index", wraps=loaded._rebuild_index) as rebuild:
        loaded.load(str(path))
        rebuild.assert_called_once()
    assert loaded.index.ntotal == len(loaded.documents)
