import json
import logging
import os
from datetime import datetime

from document_processor import process_document
from rag_pipeline import RagPipeline

from .session_manager import SessionManager, get_session_manager

logger = logging.getLogger(__name__)


class DocumentService:
    """Per-session document upload, listing, and deletion."""

    def __init__(self, session_manager: SessionManager | None = None):
        self.session_manager = session_manager or get_session_manager()

    def get_documents_list(self, session_id: str) -> str:
        with self.session_manager.session_lock(session_id) as state:
            stats = state.rag_pipeline.get_document_stats()
            if not stats:
                return "هیچ سندی بارگذاری نشده است."

            doc_list = "📚 **اسناد بارگذاری شده:**\n\n"
            for i, (doc_name, chunk_count) in enumerate(stats.items(), 1):
                doc_list += f"{i}. **{doc_name}** ({chunk_count} بخش)\n"
            return doc_list

    def process_uploaded_file(self, session_id: str, file) -> tuple[str, str]:
        if file is None:
            return "لطفاً یک فایل را برای پردازش آپلود کنید.", self.get_documents_list(session_id)

        try:
            text_chunks = process_document(file.name)
            if not text_chunks:
                return (
                    "خطا: نتوانستم هیچ متنی از فایل استخراج کنم.",
                    self.get_documents_list(session_id),
                )

            source_name = os.path.basename(file.name)

            with self.session_manager.session_lock(session_id) as state:
                added_count = state.rag_pipeline.add_documents(
                    text_chunks, source_name=source_name
                )
                if added_count == 0:
                    return (
                        "خطا: هیچ بخش معتبری از فایل استخراج نشد.",
                        self.get_documents_list(session_id),
                    )

                self._save_session(state)
                state.documents_metadata[source_name] = {
                    "processed": True,
                    "chunks": added_count,
                    "timestamp": datetime.now().isoformat(),
                }
                self._save_metadata(state)

            return (
                f"✅ فایل `{source_name}` با موفقیت به پایگاه دانش اضافه شد.\n"
                f"📄 تعداد بخش‌ها: {added_count}",
                self.get_documents_list(session_id),
            )
        except Exception as e:
            logger.exception("File upload processing failed for session %s", session_id)
            return f"یک خطای غیرمنتظره رخ داد: {e}", self.get_documents_list(session_id)

    def delete_document(self, session_id: str, doc_name: str) -> tuple[str, str]:
        if not doc_name or not doc_name.strip():
            return "لطفاً نام سند را وارد کنید.", self.get_documents_list(session_id)

        doc_name = doc_name.strip()
        with self.session_manager.session_lock(session_id) as state:
            success = state.rag_pipeline.delete_document(doc_name)
            if not success:
                return f"❌ سند `{doc_name}` یافت نشد.", self.get_documents_list(session_id)

            state.documents_metadata.pop(doc_name, None)
            self._save_metadata(state)
            self._save_session(state)

        return f"✅ سند `{doc_name}` با موفقیت حذف شد.", self.get_documents_list(session_id)

    def clear_all_documents(self, session_id: str) -> str:
        with self.session_manager.session_lock(session_id) as state:
            state.rag_pipeline = RagPipeline()
            state.documents_metadata.clear()

            for path in (
                state.workspace.vector_db_path,
                state.workspace.metadata_path,
            ):
                if path.exists():
                    path.unlink()

        return self.get_documents_list(session_id)

    def get_rag_pipeline(self, session_id: str) -> RagPipeline:
        with self.session_manager.session_lock(session_id) as state:
            return state.rag_pipeline

    def _save_session(self, state) -> None:
        state.rag_pipeline.save(str(state.workspace.vector_db_path))

    def _save_metadata(self, state) -> None:
        with open(state.workspace.metadata_path, "w", encoding="utf-8") as f:
            json.dump(state.documents_metadata, f, ensure_ascii=False, indent=4)
