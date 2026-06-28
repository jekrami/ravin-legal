import json
import logging

from .session_manager import SessionManager, get_session_manager

logger = logging.getLogger(__name__)


class ChatService:
    """Per-session chat history persistence."""

    def __init__(self, session_manager: SessionManager | None = None):
        self.session_manager = session_manager or get_session_manager()

    def load_history(self, session_id: str) -> list:
        with self.session_manager.session_lock(session_id) as state:
            path = state.workspace.chat_history_path
            if not path.exists():
                return []

            try:
                with open(path, "r", encoding="utf-8") as f:
                    stored = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load chat history for %s: %s", session_id, e)
                return []

        messages = []
        for entry in stored:
            if "user" in entry:
                messages.append({"role": "user", "content": entry["user"]})
            if "assistant" in entry:
                messages.append({"role": "assistant", "content": entry["assistant"]})
        return messages

    def append_turn(self, session_id: str, user_query: str, assistant_answer: str) -> None:
        with self.session_manager.session_lock(session_id) as state:
            path = state.workspace.chat_history_path
            full_history = []

            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        full_history = json.load(f)
                    except json.JSONDecodeError:
                        pass

            full_history.append({"user": user_query, "assistant": assistant_answer})

            with open(path, "w", encoding="utf-8") as f:
                json.dump(full_history, f, ensure_ascii=False, indent=4)

    def clear_history(self, session_id: str) -> None:
        with self.session_manager.session_lock(session_id) as state:
            path = state.workspace.chat_history_path
            if path.exists():
                path.unlink()
