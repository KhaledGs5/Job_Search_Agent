"""OpenAI-compatible client factory for NVIDIA's API."""
from openai import OpenAI

from config import get_settings


def get_openai_client() -> OpenAI:
    """Build an OpenAI-compatible client configured for NVIDIA."""
    settings = get_settings()
    return OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
    )
