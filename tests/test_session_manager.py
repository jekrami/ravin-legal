from services.chat_service import ChatService
from services.document_service import DocumentService
from services.session_manager import SessionManager
from tests.conftest import MockEmbeddings


def test_session_creation_is_unique(tmp_path):
    manager = SessionManager(sessions_dir=tmp_path / "sessions")
    id_a = manager.create_session()
    id_b = manager.create_session()
    assert id_a != id_b
    assert (tmp_path / "sessions" / id_a).is_dir()
    assert (tmp_path / "sessions" / id_b).is_dir()


def test_sessions_are_isolated(tmp_path):
    sessions_dir = tmp_path / "sessions"
    manager = SessionManager(sessions_dir=sessions_dir)
    doc_service = DocumentService(manager)
    chat_service = ChatService(manager)

    session_a = manager.create_session()
    session_b = manager.create_session()

    with manager.session_lock(session_a) as state_a:
        state_a.rag_pipeline.embedding_model = MockEmbeddings()
        state_a.rag_pipeline.add_documents(["سند الف"], source_name="a.pdf")
        state_a.rag_pipeline.save(str(state_a.workspace.vector_db_path))

    with manager.session_lock(session_b) as state_b:
        state_b.rag_pipeline.embedding_model = MockEmbeddings()
        state_b.rag_pipeline.add_documents(["سند ب"], source_name="b.pdf")
        state_b.rag_pipeline.save(str(state_b.workspace.vector_db_path))

    chat_service.append_turn(session_a, "سلام", "پاسخ الف")
    chat_service.append_turn(session_b, "سلام", "پاسخ ب")

    list_a = doc_service.get_documents_list(session_a)
    list_b = doc_service.get_documents_list(session_b)
    history_a = chat_service.load_history(session_a)
    history_b = chat_service.load_history(session_b)

    assert "a.pdf" in list_a
    assert "a.pdf" not in list_b
    assert "b.pdf" in list_b
    assert history_a[-1]["content"] == "پاسخ الف"
    assert history_b[-1]["content"] == "پاسخ ب"
