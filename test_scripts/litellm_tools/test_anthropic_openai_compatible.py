#!/usr/bin/env python3
"""
Test script for using Anthropic models with OpenAI-compatible tool calling format.
This approach demonstrates the recommended method for using Anthropic with LiteLLM.
"""
import asyncio
import json
import litellm
from dotenv import load_dotenv
from test_scripts.api_credentials import get_api_credentials, print_credentials_info

async def test_anthropic_openai_compatible():
    """
    Test Anthropic with OpenAI-compatible tool calling format through litellm.
    """
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from environment
    api_key, api_base_url, model = get_api_credentials()
    
    # Print credentials info
    print_credentials_info(api_key, api_base_url, model)
    
    # Check if it's an Anthropic model
    if not (model.startswith("anthropic/") or "claude" in model.lower()):
        print(f"\nWarning: {model} does not appear to be an Anthropic model")
        proceed = input("Do you want to proceed anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Exiting...")
            return
    
    # Set up litellm with the credentials
    litellm.api_key = api_key
    litellm.api_base = api_base_url
    
    # Add Anthropic API version header
    anthropic_headers = {"anthropic-version": "2023-06-01"}
    
    # Define a simple weather tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    
    try:
        # Step 1: Initial Request with Tool
        print("\nSending initial request to model with tool definition...")
        
        messages = [
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": "What's the weather like in Boston today?"}
        ]
        
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            extra_headers=anthropic_headers
        )
        
        print("\nResponse:")
        print(f"- ID: {response.id}")
        print(f"- Created: {response.created}")
        print(f"- Model: {response.model}")
        
        # Check if there are tool calls in the response
        if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            print("\nTool calls detected!")
            
            # Show all tool calls
            for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                print(f"\nTool Call {i+1}:")
                print(f"- ID: {tool_call.id}")
                print(f"- Type: {tool_call.type}")
                if hasattr(tool_call, "function"):
                    print(f"- Function Name: {tool_call.function.name}")
                    print(f"- Arguments: {tool_call.function.arguments}")
                    
                    # Parse the arguments
                    try:
                        args = json.loads(tool_call.function.arguments)
                        location = args.get('location', 'unknown')
                        unit = args.get('unit', 'celsius')
                        print(f"  * Location: {location}")
                        print(f"  * Unit: {unit}")
                    except json.JSONDecodeError:
                        print(f"  * Could not parse arguments as JSON")
            
            # Step 2: Add the tool result to the messages
            print("\nAdding tool result...")
            
            # Create a new messages array with the previous messages
            new_messages = messages.copy()
            
            # Add the assistant's message with tool calls
            new_messages.append(response.choices[0].message.model_dump())
            
            # Simulate getting the weather and add the result
            weather_result = {"temperature": 72, "condition": "sunny"}
            weather_text = json.dumps(weather_result)
            
            tool_call = response.choices[0].message.tool_calls[0]
            new_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": weather_text
            })
            
            # Step 3: Get the model's response to the tool result
            print("\nSending tool result back to model...")
            
            # IMPORTANT: Include the tools parameter again
            final_response = await litellm.acompletion(
                model=model,
                messages=new_messages,
                tools=tools,  # Keep the tools parameter
                extra_headers=anthropic_headers
            )
            
            print("\nFinal response:")
            print(final_response.choices[0].message.content)
            
        else:
            print("\nNo tool calls detected in the response.")
            print("\nModel's response:")
            print(response.choices[0].message.content)
    
    except Exception as e:
        print(f"\nError during test: {type(e).__name__}: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"API Error details: {e.response.text}")

if __name__ == "__main__":
    asyncio.run(test_anthropic_openai_compatible())