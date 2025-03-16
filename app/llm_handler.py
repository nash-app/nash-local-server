import litellm
from dotenv import load_dotenv
from typing import Optional


class InvalidAPIKeyError(Exception):
    """Raised when an API key is invalid or missing."""
    pass


def validate_api_key(api_key: Optional[str] = None, model: str = None) -> None:
    """Validate that an API key is present and has the correct format.

    Args:
        api_key: The API key to validate
        model: The model being used, to determine validation requirements

    Raises:
        InvalidAPIKeyError: If the API key is missing or invalid
    """
    # Skip validation for Ollama models
    if model and model.startswith("ollama/"):
        return
        
    if not api_key and not litellm.api_key:
        raise InvalidAPIKeyError(
            "No API key provided. Please set a valid API key."
        )

    key_to_check = api_key or litellm.api_key
    if not isinstance(key_to_check, str):
        raise InvalidAPIKeyError("API key must be a string.")

    # Basic format validation for common API key formats
    if not (key_to_check.startswith('sk-') and len(key_to_check) > 20):
        raise InvalidAPIKeyError(
            "Invalid API key format. API keys should start with 'sk-' "
            "and be at least 20 characters long."
        )


def configure_llm(api_key: str = None, api_base_url: str = None, model: str = None):
    """Configure LiteLLM with API keys and settings."""
    load_dotenv()

    if api_key:
        litellm.api_key = api_key
    if api_base_url:
        litellm.api_base = api_base_url

    # Validate API key
    validate_api_key(api_key, model)

    # Initialize headers
    litellm.headers = {}


async def stream_llm_response(
    messages: list = None,
    model: str = None,
    api_key: str = None,
    api_base_url: str = None,
    tools: list = None,
    tool_choice: str = "auto",
    temperature: float = 0.3,
):
    """Stream responses from the LLM.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Optional model override
        api_key: Optional API key override
        api_base_url: Optional API base URL override
        tools: Optional list of tools to make available to the model
        tool_choice: Optional specification for tool choice behavior
        temperature: Optional temperature setting

    Yields:
        Direct chunks from the LiteLLM API
    """
    try:
        if not messages:
            messages = []

        # Configure LLM with provided credentials
        configure_llm(api_key, api_base_url, model)

        # Debug the model
        print(f"Using model: {model}")
        
        # Determine if it's an Anthropic model 
        is_anthropic = model and ("claude" in model.lower() or model.lower().startswith("anthropic/"))
        
        # Prepare completion parameters
        completion_params = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "extra_headers": litellm.headers
        }
        
        # For Anthropic models add API version header
        if is_anthropic:
            print("Using Anthropic model, adding API version header")
            if "extra_headers" not in completion_params or not completion_params["extra_headers"]:
                completion_params["extra_headers"] = {}
            completion_params["extra_headers"]["anthropic-version"] = "2023-06-01"
        
        # Add tools if provided
        if tools:
            print(f"Adding {len(tools)} tools to completion params")
            
            # Ensure tools are properly formatted
            valid_tools = []
            for tool in tools:
                if not isinstance(tool, dict):
                    print(f"Warning: Skipping non-dict tool: {tool}")
                    continue
                    
                if "type" not in tool or "function" not in tool:
                    print(f"Warning: Tool missing required fields: {tool}")
                    continue
                    
                if not isinstance(tool["function"], dict):
                    print(f"Warning: Tool function is not a dict: {tool}")
                    continue
                    
                if "name" not in tool["function"] or not tool["function"]["name"]:
                    print(f"Warning: Tool missing function name: {tool}")
                    continue
                    
                valid_tools.append(tool)
            
            # Only use tools if we have valid ones
            if valid_tools:
                completion_params["tools"] = valid_tools
                completion_params["tool_choice"] = tool_choice
                print(f"Using {len(valid_tools)} validated tools with model {model}")
                # Debug first tool
                if valid_tools:
                    print(f"First tool: {valid_tools[0]['function']['name']}")
            else:
                # Fallback to stop token if no valid tools
                print("No valid tools found, falling back to stop token")
                completion_params["stop"] = ["</tool_call>"]  # Legacy stop sequence
        else:
            # Only use stop token if not using tools
            completion_params["stop"] = ["</tool_call>"]  # Legacy stop sequence

        # Debug params (excluding messages which could be long)
        debug_params = completion_params.copy()
        debug_params.pop("messages", None)
        print(f"Completion params: {debug_params}")

        # Create the response stream
        response = await litellm.acompletion(**completion_params)

        # Simply yield each chunk directly
        async for chunk in response:
            yield chunk

    except Exception as e:
        # More detailed error handling
        print(f"\nError in stream_llm_response: {type(e).__name__}: {str(e)}")
        print(f"Messages: {len(messages)} items")
        
        # If it's a provider-specific error, provide more context
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"API Error details: {e.response.text}")
        
        # Re-raise to let the caller handle it
        raise
