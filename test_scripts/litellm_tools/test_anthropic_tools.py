#!/usr/bin/env python3
"""
Test script for Anthropic tool calling with a simple weather function.
"""
import asyncio
import json
import litellm
from dotenv import load_dotenv
from test_scripts.api_credentials import get_api_credentials, print_credentials_info

# Simple fake weather function that always returns the same data
def get_current_weather(location, unit="celsius"):
    """
    Simple fake weather function that always returns sunny and 72 degrees.
    """
    temp = 72 if unit == "fahrenheit" else 22
    return {
        "location": location,
        "temperature": temp,
        "condition": "sunny",
        "humidity": "low",
        "wind": "5 mph",
        "unit": unit
    }

async def test_anthropic_tools():
    """
    Test Anthropic's tool calling capabilities through litellm.
    Uses a simple fake weather function.
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
        # Get user input for the weather query
        print("\nWhat would you like to ask Claude about the weather?")
        print("(Default: What's the weather like in Boston today?)")
        user_input = input("> ").strip()
        
        if not user_input:
            user_input = "What's the weather like in Boston today?"
        
        # Step 1: Initial Request with Tool
        print("\nSending request to Anthropic model...")
        
        # Format messages in standard format
        messages = [
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": user_input}
        ]
        
        # Make the initial API call
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            extra_headers=anthropic_headers
        )
        
        print("\nResponse:")
        print(f"- ID: {response.id}")
        print(f"- Model: {response.model}")
        
        # Show any text content from the model
        if response.choices[0].message.content:
            print(f"\nText response: {response.choices[0].message.content}")
        
        # Check if there are tool calls in the response
        if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            print("\nTool calls detected! Processing...")
            
            # Process each tool call (but we expect just one)
            for tool_call in response.choices[0].message.tool_calls:
                print(f"\nTool call:")
                print(f"- ID: {tool_call.id}")
                print(f"- Function: {tool_call.function.name}")
                print(f"- Arguments: {tool_call.function.arguments}")
                
                # Only process weather tool calls
                if tool_call.function.name == "get_current_weather":
                    # Parse the arguments
                    try:
                        args = json.loads(tool_call.function.arguments)
                        location = args.get('location', 'Boston, MA')
                        unit = args.get('unit', 'celsius')
                        
                        print(f"\nGetting weather for: {location}")
                        
                        # Call our fake weather function
                        weather_data = get_current_weather(location, unit)
                        print(f"Weather result: {json.dumps(weather_data, indent=2)}")
                        
                        # Step 2: Add the tool result to messages
                        # Create a new message array for the follow-up
                        new_messages = [
                            {"role": "system", "content": "You are a helpful weather assistant."},
                            {"role": "user", "content": user_input}
                        ]
                        
                        # Add the assistant's message with the tool call
                        new_messages.append(response.choices[0].message.model_dump())
                        
                        # Add the tool result in OpenAI-compatible format
                        new_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": json.dumps(weather_data)
                        })
                        
                        # Step 3: Get response to the tool result
                        print("\nSending tool result back to model...")
                        
                        # Make another API call with the tool result
                        # IMPORTANT: Include tools parameter again for Anthropic
                        final_response = await litellm.acompletion(
                            model=model,
                            messages=new_messages,
                            tools=tools,  # Include tools again!
                            extra_headers=anthropic_headers
                        )
                        
                        # Show the final response
                        print("\nFinal response:")
                        print(final_response.choices[0].message.content)
                        
                    except json.JSONDecodeError:
                        print("Error: Could not parse tool arguments as JSON")
                    except Exception as e:
                        print(f"Error executing tool: {str(e)}")
                else:
                    print(f"Unknown tool called: {tool_call.function.name}")
        else:
            print("\nNo tool calls detected. The model responded directly:")
            print(response.choices[0].message.content)
    
    except Exception as e:
        print(f"\nError during tool call: {type(e).__name__}: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"API Error details: {e.response.text}")

if __name__ == "__main__":
    asyncio.run(test_anthropic_tools())