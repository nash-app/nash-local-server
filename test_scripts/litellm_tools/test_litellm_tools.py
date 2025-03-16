import asyncio
import json
from app.llm_handler import configure_llm, stream_llm_response
from app.mcp_handler import MCPHandler
from test_scripts.api_credentials import get_api_credentials, print_credentials_info
from test_scripts.message_display import (
    print_messages, print_user_prompt, print_assistant_header,
    print_tool_header, print_tool_details
)

def convert_mcp_tools_to_litellm_format(mcp_tools):
    """
    Convert MCP tools to the format expected by litellm/OpenAI API.
    """
    litellm_tools = []
    
    for tool in mcp_tools.tools:
        if not tool.name:
            print(f"Warning: Found tool without a name, skipping")
            continue
            
        # Extract the schema (could be a string or already parsed)
        schema = tool.inputSchema
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse schema for tool {tool.name}")
                continue
        
        # Make sure we have a proper schema with type object
        if not schema or not isinstance(schema, dict):
            print(f"Warning: Invalid schema for tool {tool.name}, setting default")
            schema = {"type": "object", "properties": {}, "required": []}
        
        # Ensure schema has required type field
        if "type" not in schema:
            schema["type"] = "object"
            
        # Ensure properties exists if not present
        if "properties" not in schema:
            schema["properties"] = {}
            
        litellm_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description.split("\n")[0] if tool.description else f"Tool for {tool.name}",
                "parameters": schema
            }
        }
        litellm_tools.append(litellm_tool)
        
        # Debug output
        print(f"Converted tool: {tool.name}")
    
    return litellm_tools

async def chat():
    # Get API credentials from environment
    api_key, api_base_url, model = get_api_credentials()
    
    # Configure LLM with credentials
    configure_llm(api_key=api_key, api_base_url=api_base_url, model=model)
    
    # Print credentials info
    print_credentials_info(api_key, api_base_url, model)
    
    messages = []
    
    # Initialize MCP and get tool definitions
    mcp = MCPHandler.get_instance()
    await mcp.initialize()
    mcp_tools = await mcp.list_tools()
    
    # Convert MCP tools to litellm format
    litellm_tools = convert_mcp_tools_to_litellm_format(mcp_tools)
    print(f"\nConverted {len(litellm_tools)} tools to litellm format")
    
    # System message to handle tools - using OpenAI format for litellm compatibility
    messages.append({
        "role": "system",
        "content": "You are Nash, an AI assistant that can use tools to help users. Use tools when appropriate to complete user requests."
    })
    
    try:
        while True:
            # Get user input
            print_user_prompt()
            user_input = input("").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            if user_input.lower() in ['messages']:
                print_messages(messages)
                continue
                
            # Add user message to history in OpenAI format for litellm compatibility
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Stream AI response
            print_assistant_header()
            assistant_message = ""
            tool_calls = []
            
            print("\nSending request with tools to LLM...")
            print(f"Model: {model}")
            print(f"Number of tools: {len(litellm_tools)}")
            print(f"Number of messages: {len(messages)}")
            
            try:
                async for chunk in stream_llm_response(
                    messages=messages,
                    model=model,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    tools=litellm_tools  # Pass the tools to litellm
                ):
                    # Debug raw chunk if needed
                    # print(f"\nDebug raw chunk: {chunk}")
                    
                    # Process the raw LiteLLM chunk
                    if hasattr(chunk, 'choices') and chunk.choices:
                        # Extract the delta from the choices
                        delta = chunk.choices[0].delta
                        
                        # Debug delta structure
                        if chunk.choices[0].delta != {}:
                            delta_attrs = {attr: getattr(delta, attr) for attr in dir(delta) 
                                        if not attr.startswith('_') and not callable(getattr(delta, attr))}
                            # print(f"\nDebug - Delta attributes: {delta_attrs}")
                        
                        # Handle content updates
                        if hasattr(delta, 'content') and delta.content:
                            content = delta.content
                            print(content, end="", flush=True)
                            assistant_message += content
                        
                        # Handle tool call updates
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            # For the first chunk with tool calls, print a header
                            if not tool_calls:
                                print("\n", flush=True)
                                print_tool_header()
                                print(f"Debug - Found tool calls in response!")
                            
                            # Debug tool calls in delta
                            # print(f"\nDebug - Tool calls in delta: {delta.tool_calls}")
                            
                            # Process each tool call in this chunk
                            for tool_call in delta.tool_calls:
                                # Debug tool call
                                tool_call_attrs = {attr: getattr(tool_call, attr) for attr in dir(tool_call) 
                                                if not attr.startswith('_') and not callable(getattr(tool_call, attr))}
                                print(f"\nDebug - Tool call attributes: {tool_call_attrs}")
                                
                                # Get tool call index
                                index = getattr(tool_call, 'index', 0)
                                
                                # Find existing tool call or add a new one
                                existing_call = next((t for t in tool_calls if t.get('index') == index), None)
                                
                                if existing_call is None:
                                    # Initialize a new tool call
                                    new_call = {
                                        'index': index,
                                        'id': getattr(tool_call, 'id', f"call_{len(tool_calls)}"),
                                        'type': 'function',
                                        'function': {
                                            'name': "",
                                            'arguments': ""
                                        }
                                    }
                                    tool_calls.append(new_call)
                                    existing_call = new_call
                                    print(f"Debug - Created new tool call: {new_call}")
                                
                                # Update the tool call with new information
                                if hasattr(tool_call, 'function'):
                                    function_attrs = {attr: getattr(tool_call.function, attr) for attr in dir(tool_call.function) 
                                                    if not attr.startswith('_') and not callable(getattr(tool_call.function, attr))}
                                    print(f"Debug - Function attributes: {function_attrs}")
                                    
                                    if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                        existing_call['function']['name'] = tool_call.function.name
                                        print(f"Debug - Updated function name: {tool_call.function.name}")
                                    
                                    if hasattr(tool_call.function, 'arguments'):
                                        if tool_call.function.arguments:
                                            existing_call['function']['arguments'] += tool_call.function.arguments
                                            print(f"Debug - Updated function arguments: +{tool_call.function.arguments}")
            except Exception as e:
                print(f"\nError during streaming: {type(e).__name__}: {str(e)}")
                # Continue with empty message to avoid breaking the whole chat loop
            
            # Add assistant response to history 
            if assistant_message or tool_calls:
                # Create the assistant message in OpenAI format for litellm compatibility
                message = {
                    "role": "assistant",
                    "content": assistant_message
                }
                
                # Add tool_calls to the message - OpenAI style
                if tool_calls:
                    # Make a copy of tool_calls that litellm will understand
                    message["tool_calls"] = tool_calls
                    print(f"Created message with tool_calls in OpenAI format")
                    
                    # Save the Anthropic format for reference but don't use it
                    anthropic_format = {
                        "role": "assistant", 
                        "content": []
                    }
                    
                    # Add text content if any
                    if assistant_message:
                        anthropic_format["content"].append({
                            "type": "text",
                            "text": assistant_message
                        })
                    
                    # Add tool_use blocks
                    for tc in tool_calls:
                        input_json = tc.get('function', {}).get('arguments', "{}")
                        if isinstance(input_json, str):
                            try:
                                input_data = json.loads(input_json)
                            except json.JSONDecodeError:
                                input_data = input_json
                        else:
                            input_data = input_json
                            
                        anthropic_format["content"].append({
                            "type": "tool_use",
                            "id": tc.get('id', ""),
                            "name": tc.get('function', {}).get('name', ""),
                            "input": input_data
                        })
                    
                    # Print tool call details
                    for tool_call in tool_calls:
                        # Debug the tool call structure
                        print(f"\nDebug - Tool call structure: {json.dumps(tool_call, indent=2)}")
                        
                        if 'function' not in tool_call or not tool_call['function']:
                            print(f"Error: Invalid tool call structure, missing function field")
                            continue
                            
                        function_data = tool_call['function']
                        if 'name' not in function_data or not function_data['name']:
                            print(f"Error: Tool call missing function name")
                            continue
                            
                        function_name = function_data['name']
                        if not isinstance(function_name, str):
                            print(f"Error: Tool function name is not a string: {type(function_name)}")
                            function_name = str(function_name)  # Convert to string as fallback
                        
                        # Process arguments
                        if 'arguments' not in function_data:
                            print(f"Warning: No arguments found for tool {function_name}")
                            arguments = {}
                        else:
                            try:
                                arg_str = function_data['arguments']
                                if isinstance(arg_str, str) and arg_str.strip():
                                    arguments = json.loads(arg_str)
                                elif isinstance(arg_str, dict):
                                    arguments = arg_str
                                else:
                                    arguments = {}
                            except json.JSONDecodeError:
                                print(f"Warning: Could not parse arguments as JSON: {function_data['arguments']}")
                                arguments = {}
                        
                        print_tool_details(function_name, arguments)
                        
                        # Execute the tool and get results
                        try:
                            print(f"\nCalling tool: {function_name} with arguments: {arguments}")
                            tool_result = await mcp.call_tool(function_name, arguments=arguments)
                            
                            # Format the tool result
                            result_text = ""
                            if hasattr(tool_result, 'content') and tool_result.content:
                                # If it's a list of content items
                                if isinstance(tool_result.content, list):
                                    for content_item in tool_result.content:
                                        if hasattr(content_item, 'text'):
                                            result_text += content_item.text
                                # If it's a single content item
                                elif hasattr(tool_result.content, 'text'):
                                    result_text = tool_result.content.text
                            
                            # If we couldn't extract text content, fall back to string representation
                            if not result_text:
                                result_text = str(tool_result)
                            
                            # Format the response in Anthropic's format - using the assistant's message with a tool_result content block
                            # First, find the assistant's message with the tool_use block
                            if message["role"] == "assistant":
                                # Convert OpenAI-style message to Anthropic-style message with content blocks
                                if "content" in message and isinstance(message["content"], str):
                                    # Store original text content
                                    text_content = message["content"]
                                    
                                    # Create content blocks array
                                    content_blocks = []
                                    
                                    # Add text content if any
                                    if text_content:
                                        content_blocks.append({
                                            "type": "text",
                                            "text": text_content
                                        })
                                    
                                    # Add tool_use blocks for each tool call
                                    for tc in tool_calls:
                                        content_blocks.append({
                                            "type": "tool_use",
                                            "id": tc.get('id', ""),
                                            "name": tc.get('function', {}).get('name', ""),
                                            "input": tc.get('function', {}).get('arguments', "{}")
                                        })
                                    
                                    # Replace the content with the structured content blocks
                                    message["content"] = content_blocks
                                
                                        # Add the result to the message history in OpenAI format 
                            # This is flattened format compatible with litellm validation
                            tool_response = {
                                "role": "tool", 
                                "tool_call_id": tool_call.get('id', ""),
                                "name": function_name,
                                "content": result_text
                            }
                            
                            # Store original Anthropic format for reference but don't use it
                            anthropic_format = {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_call.get('id', ""),
                                        "content": result_text
                                    }
                                ]
                            }
                            
                            # Add the OpenAI-compatible format to messages
                            messages.append(tool_response)
                            
                            print(f"\nAdded tool result message in OpenAI format for tool ID: {tool_call.get('id', '')}")
                            
                            # Show the tool result to the user
                            print(f"\nTool result: {result_text}")
                        except Exception as e:
                            error_msg = f"Error executing tool {function_name}: {str(e)}"
                            print(f"\nError: {error_msg}")
                            
                            # Add error result in OpenAI format compatible with litellm
                            tool_error_response = {
                                "role": "tool", 
                                "tool_call_id": tool_call.get('id', ""),
                                "name": function_name,
                                "content": f"Error: {error_msg}"
                            }
                            messages.append(tool_error_response)
                            print(f"\nAdded tool error message in OpenAI format for tool ID: {tool_call.get('id', '')}")
                
                # Add the assistant's message to history (already done above when we converted it)
                # If we've executed tools, get the assistant's response to the tool results
                if tool_calls:
                    print_assistant_header(responding_to_tool=True)
                    assistant_response = ""
                    
                    try:
                        print("\nSending follow-up request with tool results to the model...")
                        print(f"Message count: {len(messages)}")
                        print("Last few messages:")
                        for i, msg in enumerate(messages[-3:]):
                            print(f"Message {len(messages)-3+i}: role={msg['role']}, content_type={type(msg['content'])}")
                            if isinstance(msg['content'], list):
                                print(f"  Content blocks: {[block.get('type') for block in msg['content']]}")
                        
                        async for chunk in stream_llm_response(
                            messages=messages,
                            model=model,
                            api_key=api_key,
                            api_base_url=api_base_url,
                            tools=litellm_tools
                        ):
                            if hasattr(chunk, 'choices') and chunk.choices:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    content = delta.content
                                    print(content, end="", flush=True)
                                    assistant_response += content
                        
                        # Add the assistant's response to the tool results to the message history
                        # Using OpenAI format for litellm compatibility
                        if assistant_response:
                            messages.append({
                                "role": "assistant",
                                "content": assistant_response
                            })
                    except Exception as e:
                        print(f"\nError getting follow-up response: {str(e)}")
                        print("Continuing without follow-up response")
            
    except Exception as e:
        print(f"\nError during chat: {e}")
    finally:
        # Clean up MCP
        await mcp.close()
    
    print("\nChat ended.")


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\nStopped by user")
        asyncio.run(MCPHandler.get_instance().close())
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        asyncio.run(MCPHandler.get_instance().close())