import json
import logging
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from config import MIGRATE_LEGACY, SESSIONS_DIR
from rag_pipeline import RagPipeline

logger = logging.getLogger(__name__)

VECTOR_DB_FILENAME = "vector_db.pkl"
METADATA_FILENAME = "documents_metadata.json"
CHAT_HISTORY_FILENAME = "rag_chat_history.json"


@dataclass
class SessionWorkspace:
    session_id: str
    path: Path

    @property
    def vector_db_path(self) -> Path:
        return self.path / VECTOR_DB_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.path / METADATA_FILENAME

    @property
    def chat_history_path(self) -> Path:
        return self.path / CHAT_HISTORY_FILENAME

    @property
    def analysis_dir(self) -> Path:
        return self.path / "analysis"


@dataclass
class SessionState:
    workspace: SessionWorkspace
    lock: threading.RLock = field(default_factory=threading.RLock)
    rag_pipeline: Optional[RagPipeline] = None
    documents_metadata: Dict = field(default_factory=dict)
    _loaded: bool = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return

        self.rag_pipeline = RagPipeline()
        if self.workspace.vector_db_path.exists():
            self.rag_pipeline.load(str(self.workspace.vector_db_path))

        if self.workspace.metadata_path.exists():
            with open(self.workspace.metadata_path, "r", encoding="utf-8") as f:
                self.documents_metadata = json.load(f)

        self.workspace.analysis_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = True


class SessionManager:
    """Manages per-session workspaces with thread-safe access."""

    def __init__(self, sessions_dir: Path = SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._states: Dict[str, SessionState] = {}
        self._registry_lock = threading.Lock()
        self._legacy_migrated = False

    def get_or_create_session(self, session_id: Optional[str]) -> str:
        if session_id and self._session_exists(session_id):
            return session_id
        return self.create_session()

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        workspace = SessionWorkspace(session_id=session_id, path=self.sessions_dir / session_id)
        workspace.path.mkdir(parents=True, exist_ok=True)
        workspace.analysis_dir.mkdir(parents=True, exist_ok=True)

        with self._registry_lock:
            self._states[session_id] = SessionState(workspace=workspace)

        if MIGRATE_LEGACY and not self._legacy_migrated:
            self._migrate_legacy_files(workspace)
            self._legacy_migrated = True

        logger.info("Created session %s", session_id)
        return session_id

    def get_state(self, session_id: str) -> SessionState:
        with self._registry_lock:
            if session_id not in self._states:
                if not self._session_exists(session_id):
                    raise KeyError(f"Unknown session: {session_id}")
                workspace = SessionWorkspace(
                    session_id=session_id,
                    path=self.sessions_dir / session_id,
                )
                self._states[session_id] = SessionState(workspace=workspace)
            state = self._states[session_id]

        state.ensure_loaded()
        return state

    @contextmanager
    def session_lock(self, session_id: str):
        state = self.get_state(session_id)
        state.lock.acquire()
        try:
            yield state
        finally:
            state.lock.release()

    def _session_exists(self, session_id: str) -> bool:
        return (self.sessions_dir / session_id).is_dir()

    def _migrate_legacy_files(self, workspace: SessionWorkspace) -> None:
        legacy_files = {
            "vector_db.pkl": workspace.vector_db_path,
            "documents_metadata.json": workspace.metadata_path,
            "rag_chat_history.json": workspace.chat_history_path,
        }
        for src_name, dest in legacy_files.items():
            src = Path(src_name)
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)
                logger.info("Migrated legacy %s to session %s", src_name, workspace.session_id)


_manager: Optional[SessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = SessionManager()
        return _manager
