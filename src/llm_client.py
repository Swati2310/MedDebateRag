"""
Shared Gemini client — single place to configure the LLM.
All agents import from here so switching models is a one-line change.
"""

import os
import random
import threading
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

load_dotenv()

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")

_client = None
_client_lock = threading.Lock()

# Global semaphore: cap concurrent Gemini calls to avoid rate-limit hammering
_MAX_CONCURRENT = int(os.getenv("GEMINI_MAX_CONCURRENT", "8"))
_api_semaphore  = threading.Semaphore(_MAX_CONCURRENT)

# Errors that are transient and safe to retry
_RETRYABLE_PHRASES = (
    "429", "RESOURCE_EXHAUSTED",          # rate limit
    "503", "500", "overloaded",           # server overload
    "server disconnected",                # connection drop
    "remote protocol error",              # httpx disconnect
    "connection", "timeout", "timed out", # network issues
)


def get_client() -> genai.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _RETRYABLE_PHRASES)


def generate(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    model: str | None = None,
    max_retries: int = 8,
) -> str:
    """
    Send a prompt to Gemini and return the response text.
    Retries all transient errors (rate limits, server drops, connection resets)
    with exponential backoff + jitter to avoid thundering-herd retries.
    """
    client     = get_client()
    model_name = model or os.getenv("GEMINI_CHAT_MODEL", GEMINI_CHAT_MODEL)

    config = types.GenerateContentConfig(
        system_instruction=system if system else None,
        temperature=temperature,
    )

    for attempt in range(max_retries):
        try:
            with _api_semaphore:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1 and _is_retryable(e):
                # Exponential backoff + random jitter so workers don't retry in sync
                base_wait = min(2 ** attempt, 60)
                wait      = base_wait + random.uniform(0, min(base_wait, 10))
                is_rate   = "429" in str(e) or "resource_exhausted" in str(e).lower()
                label     = "rate limit" if is_rate else "server error"
                print(f"  [retry {attempt+1}/{max_retries}] Gemini {label}, waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
