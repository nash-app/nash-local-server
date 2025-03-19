import litellm
from typing import Optional


def configure_llm(api_key: Optional[str] = None, api_base_url: str = None, model: str = None):
    """Configure LiteLLM with API keys and settings."""
    # Set API key if provided (optional for providers like Ollama)
    if api_key:
        litellm.api_key = api_key
    else:
        litellm.api_key = None

    # Set API base URL if provided
    if api_base_url:
        litellm.api_base = api_base_url

    # Initialize headers
    litellm.headers = {}


async def stream_llm_response(
    messages: list, model: str, api_key: Optional[str] = None, api_base_url: str = None, tools: list = None
):
    """Stream responses from the LLM.

    Args:
        messages: List of message dictionaries with 'role' and 'content' (required)
        model: Model to use for completion (required)
        api_key: API key for the LLM provider (optional for Ollama)
        api_base_url: API base URL for the LLM provider (required)

    Yields:
        Direct chunks from the LiteLLM API
    """
    try:
        # Validate required parameters
        if not messages:
            raise ValueError("Messages list is required")
        if not model:
            raise ValueError("Model parameter is required")
        if not api_base_url:
            raise ValueError("API base URL is required")

        # Configure LLM with provided credentials
        configure_llm(api_key, api_base_url, model)

        # Create the response stream with stop sequence
        response = await litellm.acompletion(
            model=model, messages=messages, stream=True, temperature=0.3, extra_headers=litellm.headers, tools=tools
        )

        # Simply yield each chunk directly
        async for chunk in response:
            yield chunk

    except Exception as e:
        # Simple error handling - just print the error
        print(f"\nError in stream_llm_response: {type(e).__name__}: {str(e)}")
        print(f"Messages: {len(messages)} items")

        # Re-raise to let the caller handle it
        raise
