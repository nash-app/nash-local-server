import os
from dotenv import load_dotenv


def get_api_credentials():
    """Get API key and base URL from environment variables."""
    load_dotenv()

    # Get provider model first to determine if API key is required
    model = os.getenv("PROVIDER_MODEL")
    if not model:
        raise ValueError("No model specified. Please set PROVIDER_MODEL in .env file.")

    # Get provider base URL
    api_base_url = os.getenv("PROVIDER_API_BASE")
    if not api_base_url:
        raise ValueError("No API base URL found. Please set PROVIDER_API_BASE in .env file.")

    # Get provider API key (optional for Ollama models)
    api_key = os.getenv("PROVIDER_API_KEY")
    if not api_key and not model.startswith("ollama/"):
        raise ValueError("No API key found. Please set PROVIDER_API_KEY in .env file.")

    return api_key, api_base_url, model


def print_credentials_info(api_key, api_base_url, model):
    """Print credential information in a user-friendly way."""
    print("\nUsing configuration:")
    if api_key:
        print(f"- API Key: {api_key[:8]}...")
    else:
        print("- API Key: None (not required for this model)")
    print(f"- API Base URL: {api_base_url}")
    print(f"- Model: {model}")
