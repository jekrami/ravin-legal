import logging
from typing import Dict, Iterator, List, Optional

from .ollama_client import OllamaError, ollama_chat
from .passes import ANALYSIS_PASSES
from .prompts import MASTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _build_user_prompt(contract_text: str, pass_config: dict, accumulated_context: str) -> str:
    allowed_actions = "\n".join([f"  • {item}" for item in pass_config["allowed"]])
    forbidden_actions = "\n".join([f"  • {item}" for item in pass_config["forbidden"]])

    return f"""
Original contract text (Persian, RTL):

----------------
{contract_text}
----------------

Current analysis focus:
{pass_config['title']}

Objective:
{pass_config['objective']}

You are ALLOWED to:
{allowed_actions}

You are FORBIDDEN from:
{forbidden_actions}

Quote Policy: {pass_config['quote_policy']}
Silence Policy: {pass_config['silence_policy']}

Required Output Format:
{pass_config['output_format']}

Previous findings (for context only, may be challenged):
{accumulated_context}
"""


def run_deep_analysis(
    contract_text: str,
    selected_pass_ids: Optional[List[int]] = None,
    title_fa_map: Optional[Dict[int, str]] = None,
) -> Iterator[dict]:
    """
    Run legal analysis passes sequentially.

    Yields progress events:
      {"event": "progress", "step": int, "total": int, "pass_id": int, "title": str}
    Then a final event:
      {"event": "complete", "results": dict}
    """
    if not contract_text.strip():
        raise ValueError("Contract text is empty")

    passes_to_run = ANALYSIS_PASSES
    if selected_pass_ids is not None:
        passes_to_run = [p for p in ANALYSIS_PASSES if p["id"] in selected_pass_ids]

    if not passes_to_run:
        raise ValueError("No analysis passes selected")

    results = {}
    accumulated_context = ""
    total = len(passes_to_run)

    for step_num, pass_config in enumerate(passes_to_run, 1):
        yield {
            "event": "progress",
            "step": step_num,
            "total": total,
            "pass_id": pass_config["id"],
            "title": pass_config["title"],
        }

        logger.info("Running %s ...", pass_config["title"])

        user_prompt = _build_user_prompt(contract_text, pass_config, accumulated_context)
        output = ollama_chat(
            system_prompt=MASTER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        result_entry = {
            "title": pass_config["title"],
            "output": output.strip(),
        }
        if title_fa_map and pass_config["id"] in title_fa_map:
            result_entry["title_fa"] = title_fa_map[pass_config["id"]]

        results[pass_config["id"]] = result_entry
        accumulated_context += f"\n\n{pass_config['title']}:\n{output.strip()}"

    yield {"event": "complete", "results": results}
