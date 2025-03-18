import asyncio
import json
from app.llm_handler import configure_llm, stream_llm_response
from app.mcp_handler import MCPHandler
from app.prompts import get_system_prompt
from test_scripts.api_credentials import get_api_credentials, print_credentials_info
from test_scripts.message_display import (
    print_messages, print_user_prompt, print_assistant_header
)


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
    tools = await mcp.list_tools_litellm()

    system_prompt = get_system_prompt()

    messages.append({
        "role": "system",
        "content": system_prompt
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
                
            # Add user message to history
            messages.append({
                "role": "user",
                "content": user_input
            })

            while True:
                # Stream AI response
                print_assistant_header()
                assistant_message = ""
                
                # Tool call tracking variables
                tool_call_chunks = []
                collecting_tool_call = False
                tool_call_id = None
                tool_name = None
                tool_args = ""
                
                # STREAMING RESPONSE PROCESSING
                # This async loop processes the streamed response from the LLM via litellm.
                # 
                # How streaming works:
                # 1. The response comes in chunks, each chunk containing a delta (new content)
                # 2. For regular text responses, delta.content contains text fragments to display
                # 3. For tool calls, delta.tool_calls contains structured JSON objects
                #
                # Tool call streaming format:
                # - Each chunk contains part of a tool call, delivered as structured data
                # - First chunk typically has the tool call ID and function name
                # - Subsequent chunks contain fragments of the arguments JSON
                # - The final chunk includes a finish_reason indicating completion
                #
                # We accumulate these fragments to rebuild the complete tool call, then
                # use the MCPHandler to directly execute the tool with the extracted information.
            
                async for chunk in stream_llm_response(
                    messages=messages,
                    model=model,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    tools=tools
                ):
                    # Process the raw LiteLLM chunk
                    if hasattr(chunk, 'choices') and chunk.choices:
                        # Extract the delta from the choices (delta = what's new in this chunk)
                        delta = chunk.choices[0].delta
                        # The finish_reason will only appear in the final chunk when generation is complete
                        finish_reason = chunk.choices[0].finish_reason if hasattr(chunk.choices[0], 'finish_reason') else None
                        
                        # HANDLING TOOL CALLS
                        # Tool calls come through delta.tool_calls rather than delta.content
                        # delta.tool_calls is an array of structured objects, not plain text
                        # Each call has: id, type, function{name, arguments}
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            # We're receiving a tool call
                            collecting_tool_call = True
                            
                            # Store the raw tool call data
                            tool_call_chunks.append(delta.tool_calls)
                            
                            # Process the tool call
                            # We iterate through each tool_call in the array (usually just one)
                            for tool_call in delta.tool_calls:
                                # TOOL CALL ID PROCESSING
                                # The ID typically comes in the first chunk only
                                # We use 'if not tool_call_id' to ensure we only capture it once
                                if not tool_call_id and hasattr(tool_call, 'id'):
                                    tool_call_id = tool_call.id
                                    print(f"\nTOOL CALL ID: {tool_call_id}")
                                
                                # FUNCTION INFORMATION PROCESSING
                                if hasattr(tool_call, 'function'):
                                    # TOOL NAME PROCESSING
                                    # Tool name typically comes in the first chunk with the ID
                                    # We only store it once using 'if not tool_name'
                                    if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                        if not tool_name:
                                            tool_name = tool_call.function.name
                                            print(f"\nTOOL NAME: {tool_name}")
                                    
                                    # ARGUMENTS PROCESSING
                                    # Arguments often come split across multiple chunks
                                    # We concatenate each piece to build the complete JSON
                                    if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                        tool_args += tool_call.function.arguments
                        
                        # REGULAR CONTENT PROCESSING
                        # If this isn't a tool call, it's regular content in delta.content
                        elif hasattr(delta, 'content') and delta.content:
                            content = delta.content
                            # Print immediately for streaming effect
                            print(content, end="", flush=True)
                            # Accumulate to build the complete message
                            assistant_message += content
                        
                        # TOOL CALL COMPLETION DETECTION
                        # The final chunk of a tool call includes finish_reason
                        # This signals we have received the complete tool call
                        if collecting_tool_call and finish_reason:
                            print(f"\nTOOL CALL COMPLETE - Finish reason: {finish_reason}")
                            print(f"ACCUMULATED ARGS: {tool_args}")
                            
                            # Format the tool call as a structured format in the assistant's message
                            assistant_message = f"<tool_call>{{\"function\": {{\"name\": \"{tool_name}\", \"arguments\": {tool_args}}}}}</tool_call>"
                            collecting_tool_call = False

                # Add assistant response to history 
                if assistant_message:
                    # Add the original assistant message to the history first
                    message = {
                        "role": "assistant",
                        "content": assistant_message
                    }
                    messages.append(message)

                    # Check if we have a tool call to process
                    if tool_name and tool_args and collecting_tool_call == False:
                        print("\nPROCESSING DETECTED TOOL CALL:")
                        print(f"Tool Name: {tool_name}")
                        print(f"Tool Args: {tool_args}")
                        
                        try:
                            # Parse the arguments JSON if needed
                            # Arguments should already be a valid JSON string from the stream
                            tool_arguments = json.loads(tool_args)
                            
                            print("EXECUTING TOOL DIRECTLY VIA MCP_HANDLER:")
                            # Execute the tool directly using MCPHandler
                            tool_result = await mcp.call_tool(tool_name, arguments=tool_arguments)
                            
                            # Extract the text content from the tool result
                            result_text = ""
                            
                            # Try to extract text from the content field if available
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
                            
                            # Format for display
                            formatted_result = f"<tool_results>\n{result_text}\n</tool_results>"
                            
                            # Store the original tool call in the message for history
                            formatted_tool_call = f"<tool_call>{{\"function\": {{\"name\": \"{tool_name}\", \"arguments\": {tool_args}}}}}</tool_call>"
                            message['content'] = formatted_tool_call
                            
                            print("TOOL CALL RESULT --------------------------------------------------------")
                            print(formatted_result)
                            print("END RESULT --------------------------------------------------------")
                            
                            # Add the tool result to the conversation
                            messages.append({
                                "role": "assistant",
                                "content": f"Tool result: {formatted_result}",
                            })
                        except Exception as e:
                            print(f"\nError executing tool directly: {str(e)}")
                            break
                    else:
                        # No tool call detected, nothing to process
                        break
                else:
                    break
            
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
