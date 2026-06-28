__version__ = "2.0.0"

import os
from pathlib import Path

# Storage paths
DATA_DIR = Path(os.environ.get("RAVIN_DATA_DIR", "data"))
SESSIONS_DIR = DATA_DIR / "sessions"
SESSION_LOCK_TIMEOUT = 30

# Legacy v1 flat files (migrated only when MIGRATE_LEGACY=1)
MIGRATE_LEGACY = os.environ.get("MIGRATE_LEGACY", "").strip() in ("1", "true", "yes")

# Ollama models
PRIMARY_LEGAL_ANALYST_MODEL = "qwen2.5:14b-instruct"
SECONDARY_VERIFICATION_MODEL = "ravin-gemma3"
SYNTHESIZER_MODEL = "llama3"
EMBEDDING_MODEL = "paraphrase-multilingual:278m-mpnet-base-v2-fp16"
DEEP_ANALYSIS_MODEL = PRIMARY_LEGAL_ANALYST_MODEL
PRESENTATION_MODEL = PRIMARY_LEGAL_ANALYST_MODEL

# Ollama API endpoints
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"

# Backward-compatible alias used by llm_handler
OLLAMA_API_ENDPOINT = OLLAMA_GENERATE_ENDPOINT

# Timeouts (seconds)
OLLAMA_GENERATE_TIMEOUT = 180
OLLAMA_CHAT_TIMEOUT = 600
OLLAMA_PRESENTATION_TIMEOUT = 300

# RAG configuration
CHUNK_SIZE = 350
CHUNK_OVERLAP = 100
TOP_K = 10

# Keyword boosting
KEYWORD_BOOST_CONFIG = {
    "مبلغ": 1.5,
    "تاریخ": 1.5,
    "ماده": 1.2,
    "تبصره": 1.2,
    "فسخ": 1.5,
    "جریمه": 1.5,
    "تعهد": 1.3,
}

# LLM parameters
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.8
PRESENTATION_TEMPERATURE = 0.2
PRESENTATION_TOP_P = 0.9
