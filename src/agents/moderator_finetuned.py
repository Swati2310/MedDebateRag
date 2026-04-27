"""
Fine-tuned moderator stub.

Since we are using Gemini (no local GPU required), this module
simply delegates to the standard Gemini moderator. The QLoRA
fine-tuning path on Llama is available in src/finetuning/ if
you later want to run it on a Colab A100.
"""

from src.agents.moderator import run_moderator


def run_finetuned_moderator(prompt: str, **kwargs) -> dict:
    """Delegates to Gemini moderator — no local Llama required."""
    # We re-use the Gemini moderator with the raw prompt.
    # The prompt is already formatted by moderator.py so we just
    # call generate directly.
    from src.llm_client import generate
    from src.agents.doctor_a import parse_agent_response
    from src.agents.moderator import MODERATOR_SYSTEM

    text = generate(prompt, system=MODERATOR_SYSTEM, temperature=0.3)
    return parse_agent_response(text)
