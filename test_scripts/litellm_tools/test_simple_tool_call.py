#!/usr/bin/env python3
"""
Test script for making a simple tool call with litellm using the current model.
"""
import asyncio
import json
import litellm
from dotenv import load_dotenv
from test_scripts.api_credentials import get_api_credentials, print_credentials_info

async def test_simple_tool_call():
    """
    Test making a simple tool call with the current model.
    """
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from environment
    api_key, api_base_url, model = get_api_credentials()
    
    # Print credentials info
    print_credentials_info(api_key, api_base_url, model)
    
    # Set up litellm with the credentials
    litellm.api_key = api_key
    litellm.api_base = api_base_url
    
    # Check if model supports function calling
    if not litellm.supports_function_calling(model):
        print(f"\nWarning: Model {model} may not support function calling according to litellm")
        proceed = input("Do you want to proceed anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Exiting...")
            return
    
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
    
    # Call the model with a weather question
    try:
        print("\nSending request to model...")
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like in Boston today?"}
            ],
            tools=tools,
            tool_choice="auto",
        )
        
        print("\nResponse:")
        print(f"- ID: {response.id}")
        print(f"- Created: {response.created}")
        print(f"- Model: {response.model}")
        
        # Check if there are tool calls in the response
        if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            print("\nTool calls detected!")
            for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                print(f"\nTool Call {i+1}:")
                print(f"- ID: {tool_call.id}")
                print(f"- Type: {tool_call.type}")
                if hasattr(tool_call, "function"):
                    print(f"- Function Name: {tool_call.function.name}")
                    print(f"- Function Arguments: {tool_call.function.arguments}")
                    
                    # Parse the arguments
                    try:
                        args = json.loads(tool_call.function.arguments)
                        print(f"  * Location: {args.get('location', 'not provided')}")
                        print(f"  * Unit: {args.get('unit', 'not provided')}")
                    except json.JSONDecodeError:
                        print(f"  * Could not parse arguments as JSON")
            
            # Simulate getting the weather and continue the conversation
            weather_result = json.dumps({"temperature": 72, "condition": "sunny"})
            print("\nSimulating weather tool execution...")
            print(f"- Result: {weather_result}")
            
            # Add the tool result to the messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like in Boston today?"},
                response.choices[0].message.model_dump()  # Add assistant's message with tool calls
            ]
            
            # Add tool result
            tool_call = response.choices[0].message.tool_calls[0]
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": weather_result
            })
            
            # Get LLM's response to the tool result
            print("\nSending tool result back to model...")
            final_response = await litellm.acompletion(
                model=model,
                messages=messages
            )
            
            print("\nFinal response:")
            print(final_response.choices[0].message.content)
            
        else:
            print("\nNo tool calls in the response.")
            print("\nModel's response:")
            print(response.choices[0].message.content)
    
    except Exception as e:
        print(f"\nError during tool call: {type(e).__name__}: {str(e)}")
        
        # If there's a related response attribute, print it
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"API Error details: {e.response.text}")

if __name__ == "__main__":
    asyncio.run(test_simple_tool_call())