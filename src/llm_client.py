"""
Shared Gemini client — single place to configure the LLM.
All agents import from here so switching models is a one-line change.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError

load_dotenv()

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    model: str | None = None,
    max_retries: int = 5,
) -> str:
    """
    Send a prompt to Gemini and return the response text.
    Retries on 503 with exponential backoff.
    """
    client     = get_client()
    model_name = model or os.getenv("GEMINI_CHAT_MODEL", GEMINI_CHAT_MODEL)

    config = types.GenerateContentConfig(
        system_instruction=system if system else None,
        temperature=temperature,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return response.text
        except ServerError as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f"  [retry {attempt+1}/{max_retries}] Gemini 503, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception:
            raise
