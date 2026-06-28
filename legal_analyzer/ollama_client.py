import logging
from typing import Dict, List, Optional

import requests

from config import (
    DEEP_ANALYSIS_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    OLLAMA_CHAT_ENDPOINT,
    OLLAMA_CHAT_TIMEOUT,
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_GENERATE_TIMEOUT,
    OLLAMA_PRESENTATION_TIMEOUT,
    PRESENTATION_MODEL,
    PRESENTATION_TEMPERATURE,
    PRESENTATION_TOP_P,
)

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when an Ollama API request fails."""


def ollama_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    timeout: Optional[int] = None,
) -> str:
    """Send a chat-style request to Ollama (/api/chat)."""
    payload: Dict = {
        "model": model or DEEP_ANALYSIS_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
            "top_p": top_p if top_p is not None else LLM_TOP_P,
        },
        "stream": False,
    }

    try:
        response = requests.post(
            OLLAMA_CHAT_ENDPOINT,
            json=payload,
            timeout=timeout or OLLAMA_CHAT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.RequestException as e:
        logger.exception("Ollama chat request failed for model %s", payload["model"])
        raise OllamaError(
            f"خطا در ارتباط با مدل {payload['model']}. لطفاً مطمئن شوید Ollama در حال اجراست."
        ) from e


def ollama_chat_messages(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    timeout: Optional[int] = None,
) -> str:
    """Send a multi-message chat request to Ollama."""
    payload: Dict = {
        "model": model or PRESENTATION_MODEL,
        "messages": messages,
        "options": {
            "temperature": (
                temperature if temperature is not None else PRESENTATION_TEMPERATURE
            ),
            "top_p": top_p if top_p is not None else PRESENTATION_TOP_P,
        },
        "stream": False,
    }

    try:
        response = requests.post(
            OLLAMA_CHAT_ENDPOINT,
            json=payload,
            timeout=timeout or OLLAMA_PRESENTATION_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.RequestException as e:
        logger.exception(
            "Ollama chat-messages request failed for model %s", payload["model"]
        )
        raise OllamaError(
            f"خطا در ارتباط با مدل {payload['model']}. لطفاً مطمئن شوید Ollama در حال اجراست."
        ) from e


def ollama_generate(
    model: str,
    prompt: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    timeout: Optional[int] = None,
) -> str:
    """Send a generate-style request to Ollama (/api/generate)."""
    payload: Dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
            "top_p": top_p if top_p is not None else LLM_TOP_P,
        },
    }

    try:
        response = requests.post(
            OLLAMA_GENERATE_ENDPOINT,
            json=payload,
            timeout=timeout or OLLAMA_GENERATE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.RequestException as e:
        logger.exception("Ollama generate request failed for model %s", model)
        raise OllamaError(
            f"خطا در ارتباط با مدل {model}. لطفاً مطمئن شوید Ollama در حال اجراست."
        ) from e
