#!/usr/bin/env python3
"""
Test script for Anthropic tool calling with streaming completions.
This script shows how to handle tool calls with streaming responses.
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

async def test_anthropic_streaming_tools():
    """
    Test Anthropic's tool calling capabilities with streaming responses.
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
        
        # Format messages in standard format
        messages = [
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": user_input}
        ]
        
        # -----------------------------------------------------------------
        # PART 1: STREAMING INITIAL REQUEST
        # -----------------------------------------------------------------
        print("\nSending streaming request to Anthropic model...")
        
        # Variables to collect the response
        assistant_message = ""
        tool_calls = []
        current_tool_call = None
        
        # Stream the initial request with tools
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            stream=True,
            tools=tools,
            tool_choice="auto",
            extra_headers=anthropic_headers
        )
        
        async for chunk in response:
            # Process each chunk from the stream
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                
                # Handle text content if any
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    print(content, end="", flush=True)
                    assistant_message += content
                
                # Handle tool calls in streaming
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        # Get the index to identify which tool call this chunk belongs to
                        idx = tool_call_delta.index
                        
                        # Find or create the tool call object for this index
                        if idx >= len(tool_calls):
                            # This is a new tool call, so initialize it
                            new_tool_call = {
                                'index': idx,
                                'id': getattr(tool_call_delta, 'id', f"call_{idx}"),
                                'type': 'function',
                                'function': {
                                    'name': '',
                                    'arguments': ''
                                }
                            }
                            tool_calls.append(new_tool_call)
                        
                        # Reference the current tool call
                        current_tool_call = tool_calls[idx]
                        
                        # Update the tool call information
                        if hasattr(tool_call_delta, 'function'):
                            function_delta = tool_call_delta.function
                            
                            # Update name if provided in this chunk
                            if hasattr(function_delta, 'name') and function_delta.name:
                                current_tool_call['function']['name'] = function_delta.name
                                # Print when we first get the function name
                                if len(current_tool_call['function']['arguments']) == 0:
                                    print(f"\n\nTool call detected: {function_delta.name}")
                            
                            # Append to arguments if provided in this chunk
                            if hasattr(function_delta, 'arguments') and function_delta.arguments:
                                current_tool_call['function']['arguments'] += function_delta.arguments
                                print(".", end="", flush=True)  # Show progress as arguments stream in
        
        # Print a summary of what we received
        print("\n\nComplete response received:")
        if assistant_message:
            print(f"Text content: {assistant_message}")
        
        if tool_calls:
            print(f"\nTool calls detected:")
            for i, tool_call in enumerate(tool_calls):
                print(f"Tool Call {i+1}:")
                print(f"- Function: {tool_call['function']['name']}")
                
                # Try to pretty print the arguments
                try:
                    args = json.loads(tool_call['function']['arguments'])
                    print(f"- Arguments: {json.dumps(args, indent=2)}")
                except:
                    print(f"- Arguments: {tool_call['function']['arguments']}")
            
            # Process the tool call (we'll just handle the first one)
            if tool_calls and tool_calls[0]['function']['name'] == 'get_current_weather':
                tool_call = tool_calls[0]
                
                try:
                    args = json.loads(tool_call['function']['arguments'])
                    location = args.get('location', 'Boston, MA')
                    unit = args.get('unit', 'celsius')
                    
                    print(f"\nExecuting weather function for: {location}")
                    weather_data = get_current_weather(location, unit)
                    print(f"Weather result: {json.dumps(weather_data, indent=2)}")
                    
                    # Prepare messages for the follow-up request
                    new_messages = messages.copy()
                    
                    # Create a properly formatted assistant message with the tool call
                    assistant_response = {
                        "role": "assistant",
                        "content": assistant_message,
                        "tool_calls": tool_calls
                    }
                    new_messages.append(assistant_response)
                    
                    # Add the tool result
                    tool_result = {
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "name": tool_call['function']['name'],
                        "content": json.dumps(weather_data)
                    }
                    new_messages.append(tool_result)
                    
                    # -----------------------------------------------------------------
                    # PART 2: STREAMING FOLLOW-UP RESPONSE WITH TOOL RESULTS
                    # -----------------------------------------------------------------
                    print("\nSending tool result back to model (streaming)...")
                    
                    # Variable to collect the final response
                    final_response = ""
                    
                    # Stream the follow-up request
                    print("\nFinal response: ", end="")
                    final_response_stream = await litellm.acompletion(
                        model=model,
                        messages=new_messages,
                        stream=True,
                        tools=tools,  # Include tools again!
                        extra_headers=anthropic_headers
                    )
                    
                    async for chunk in final_response_stream:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            
                            # Handle content
                            if hasattr(delta, 'content') and delta.content:
                                content = delta.content
                                print(content, end="", flush=True)
                                final_response += content
                    
                    print("\n\nStreaming complete!")
                    
                except json.JSONDecodeError:
                    print("Error: Could not parse tool arguments as JSON")
                except Exception as e:
                    print(f"Error executing tool: {str(e)}")
        else:
            print("\nNo tool calls detected in the response.")
    
    except Exception as e:
        print(f"\nError during streaming tool call: {type(e).__name__}: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"API Error details: {e.response.text}")

if __name__ == "__main__":
    asyncio.run(test_anthropic_streaming_tools())