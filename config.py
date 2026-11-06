import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1/chat/completions",
).strip()
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3.5-super-vl:free",
).strip()

MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "30"))
MAX_ROWS_FOR_AI_SAMPLE = int(os.getenv("MAX_ROWS_FOR_AI_SAMPLE", "20"))
