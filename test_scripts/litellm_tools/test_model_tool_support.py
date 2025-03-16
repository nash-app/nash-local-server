#!/usr/bin/env python3
import litellm
from dotenv import load_dotenv
from test_scripts.api_credentials import get_api_credentials


def test_model_tool_support():
    """
    Test if the currently configured model supports tool/function calling.
    """
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from environment
    api_key, api_base_url, model = get_api_credentials()
    
    # Print current configuration
    print("\nCurrent configuration:")
    print(f"- API Base URL: {api_base_url}")
    print(f"- Model: {model}")
    
    # Set up litellm with the credentials
    litellm.api_key = api_key
    litellm.api_base = api_base_url
    
    # Check tool/function calling support
    print("\nTesting tool/function calling support:")
    supports_function_calling = litellm.supports_function_calling(model)
    print(f"- {model} supports function calling: {supports_function_calling}")


if __name__ == "__main__":
    test_model_tool_support()
