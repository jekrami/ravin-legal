import json
import logging
from datetime import datetime
from typing import Dict, Iterator, List, Optional

from document_processor import extract_full_text
from legal_analyzer.orchestrator import run_deep_analysis
from legal_analyzer.presentation import generate_persian_presentation

from .session_manager import SessionManager, get_session_manager

logger = logging.getLogger(__name__)


class AnalysisService:
    """Per-session deep legal analysis with isolated output paths."""

    def __init__(self, session_manager: SessionManager | None = None):
        self.session_manager = session_manager or get_session_manager()

    def run_analysis(
        self,
        session_id: str,
        pdf_path: str,
        selected_pass_ids: List[int],
        title_fa_map: Dict[int, str],
    ) -> Iterator[dict]:
        contract_text = extract_full_text(pdf_path)
        if not contract_text.strip():
            raise ValueError("Could not extract text from PDF")

        yield from run_deep_analysis(
            contract_text,
            selected_pass_ids=selected_pass_ids,
            title_fa_map=title_fa_map,
        )

    def save_results(
        self,
        session_id: str,
        results: dict,
    ) -> tuple[str, str]:
        """Save analysis JSON and presentation under the session analysis dir."""
        with self.session_manager.session_lock(session_id) as state:
            analysis_dir = state.workspace.analysis_dir
            analysis_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = analysis_dir / f"deep_analysis_{timestamp}.json"
            presentation_path = analysis_dir / "presentation_fa.txt"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return str(json_path), str(presentation_path)

    def generate_presentation(
        self,
        analysis_json_path: str,
        output_path: str,
    ) -> str:
        return generate_persian_presentation(
            analysis_json_path=analysis_json_path,
            output_path=output_path,
        )
