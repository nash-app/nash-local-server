import datetime
from typing import Optional, Dict

import litellm
from litellm import get_max_tokens, token_counter

from .prompts import get_system_prompt


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
    litellm.return_response_headers = True


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
    return await litellm.acompletion(
        model=model, messages=messages, stream=True, temperature=0.7, extra_headers=litellm.headers, tools=tools
    )


def clean_up_tool_results_inline(messages: list):
    """Clean up tool results that have been responded to already.
    Because we call this when the LLM hasn't fully responded to the tool call yet,
    we shouldn't touch messages the LLM hasn't responded to.
    We can guarantee it has responded to them if it is not the last message.
    Note: This is why we do messages[:-1].
    """
    for index, message in enumerate(messages[:-1]):
        if message["role"] == "tool":
            if len(message["content"]) > 5000:
                truncate_msg = "... (Truncated. Rerun tool to operate on full results again.)"
                message["content"] = message["content"][:5000] + truncate_msg
    return messages


def get_conversation_token_info(messages: list, model: str) -> Dict[str, int]:
    """Get token usage information for a conversation.

    Args:
        messages: List of message dictionaries with 'role' and 'content' (required)
        model: Model identifier string (required)

    Returns:
        Dictionary containing token counts and limits

    Raises:
        ValueError: If messages or model is not provided
    """
    if not messages:
        raise ValueError("Messages list is required")
    if not model:
        raise ValueError("Model identifier is required")

    try:
        max_tokens = get_max_tokens(model)
        used_tokens = token_counter(
            model=model, messages=[{"role": "system", "content": get_system_prompt()}] + messages
        )
        remaining_tokens = max_tokens - used_tokens

        return {
            "max_tokens": max_tokens,
            "used_tokens": used_tokens,
            "remaining_tokens": remaining_tokens,
            "model": model,
        }
    except Exception as e:
        print(f"Error getting token info: {str(e)}")
        return {"max_tokens": 0, "used_tokens": 0, "remaining_tokens": 0, "model": model, "error": str(e)}


def get_response_metadata_from_headers(headers: dict) -> dict:
    """Get response metadata from headers.

    Args:
        headers: Headers from the response
    """

    metadata = {"limits": {}, "sleep_seconds": 0, "response_cost": 0}

    if all(
        key in headers
        for key in [
            "llm_provider-anthropic-ratelimit-input-tokens-limit",
            "llm_provider-anthropic-ratelimit-input-tokens-remaining",
            "llm_provider-anthropic-ratelimit-input-tokens-reset",
            "llm_provider-anthropic-ratelimit-output-tokens-limit",
            "llm_provider-anthropic-ratelimit-output-tokens-remaining",
            "llm_provider-anthropic-ratelimit-output-tokens-reset",
            "llm_provider-anthropic-ratelimit-requests-limit",
            "llm_provider-anthropic-ratelimit-requests-remaining",
            "llm_provider-anthropic-ratelimit-requests-reset",
        ]
    ):
        # Get remaining tokens and determine if we should sleep until the reset time - NOTE: Assumption is these fields exist in the response headers
        metadata["limits"] = {
            "input_tokens": {
                "limit": int(headers.get("llm_provider-anthropic-ratelimit-input-tokens-limit", 0)),
                "remaining": int(headers.get("llm_provider-anthropic-ratelimit-input-tokens-remaining", 0)),
                "reset": headers.get("llm_provider-anthropic-ratelimit-input-tokens-reset", ""),
            },
            "output_tokens": {
                "limit": int(headers.get("llm_provider-anthropic-ratelimit-output-tokens-limit", 0)),
                "remaining": int(headers.get("llm_provider-anthropic-ratelimit-output-tokens-remaining", 0)),
                "reset": headers.get("llm_provider-anthropic-ratelimit-output-tokens-reset", ""),
            },
            "requests": {
                "limit": int(headers.get("llm_provider-anthropic-ratelimit-requests-limit", 0)),
                "remaining": int(headers.get("llm_provider-anthropic-ratelimit-requests-remaining", 0)),
                "reset": headers.get("llm_provider-anthropic-ratelimit-requests-reset", ""),
            },
        }

        # If any of the limits are at 10% or less, we should sleep until the reset time
        if metadata["limits"]["input_tokens"]["remaining"] <= metadata["limits"]["input_tokens"]["limit"] * 0.1:
            reset_time = datetime.datetime.fromisoformat(
                metadata["limits"]["input_tokens"]["reset"].replace("Z", "+00:00")
            )
            # Get current time in UTC
            current_time = datetime.datetime.now(datetime.timezone.utc)
            # Calculate the difference in seconds
            metadata["sleep_seconds"] = (reset_time - current_time).total_seconds()
        elif metadata["limits"]["output_tokens"]["remaining"] <= metadata["limits"]["output_tokens"]["limit"] * 0.1:
            reset_time = datetime.datetime.fromisoformat(
                metadata["limits"]["output_tokens"]["reset"].replace("Z", "+00:00")
            )
            # Get current time in UTC
            current_time = datetime.datetime.now(datetime.timezone.utc)
            # Calculate the difference in seconds
            metadata["sleep_seconds"] = (reset_time - current_time).total_seconds()
        elif metadata["limits"]["requests"]["remaining"] <= metadata["limits"]["requests"]["limit"] * 0.1:
            reset_time = datetime.datetime.fromisoformat(metadata["limits"]["requests"]["reset"].replace("Z", "+00:00"))
            # Get current time in UTC
            current_time = datetime.datetime.now(datetime.timezone.utc)
            # Calculate the difference in seconds
            metadata["sleep_seconds"] = (reset_time - current_time).total_seconds()

    if "response_cost" in headers:
        metadata["response_cost"] = headers["response_cost"]

    return metadata
