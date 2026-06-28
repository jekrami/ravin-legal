from concurrent.futures import ThreadPoolExecutor

from services.document_service import DocumentService
from services.session_manager import SessionManager
from tests.conftest import MockEmbeddings


def _upload(session_id, doc_service, label):
    class FakeFile:
        name = f"/tmp/{label}.pdf"

    with __import__("unittest").mock.patch(
        "services.document_service.process_document",
        return_value=[f"chunk for {label}"],
    ):
        doc_service.process_uploaded_file(session_id, FakeFile())
    return doc_service.get_documents_list(session_id)


def test_parallel_uploads_do_not_cross_contaminate(tmp_path):
    sessions_dir = tmp_path / "sessions"
    manager = SessionManager(sessions_dir=sessions_dir)
    doc_service = DocumentService(manager)

    session_a = manager.create_session()
    session_b = manager.create_session()

    # Inject mock embeddings to avoid Ollama calls during concurrent uploads
    with manager.session_lock(session_a) as state_a:
        state_a.rag_pipeline.embedding_model = MockEmbeddings()
    with manager.session_lock(session_b) as state_b:
        state_b.rag_pipeline.embedding_model = MockEmbeddings()

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(_upload, session_a, doc_service, "alpha")
        future_b = pool.submit(_upload, session_b, doc_service, "beta")
        list_a = future_a.result()
        list_b = future_b.result()

    assert "alpha.pdf" in list_a
    assert "alpha.pdf" not in list_b
    assert "beta.pdf" in list_b
    assert "beta.pdf" not in list_a
