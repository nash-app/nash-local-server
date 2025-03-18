import asyncio
import json
from app.llm_handler import configure_llm, stream_llm_response
from app.mcp_handler import MCPHandler
from app.prompts import get_system_prompt
from test_scripts.api_credentials import get_api_credentials, print_credentials_info
from test_scripts.tool_processor import process_tool_call
from test_scripts.message_display import (
    print_messages, print_user_prompt, print_assistant_header,
    print_tool_header, print_tool_details
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
            # Print current message history
            
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
            
                async for chunk in stream_llm_response(
                    messages=messages,
                    model=model,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    tools=tools
                ):
                    # Process the raw LiteLLM chunk
                    if hasattr(chunk, 'choices') and chunk.choices:
                        # Extract the delta from the choices
                        delta = chunk.choices[0].delta
                        finish_reason = chunk.choices[0].finish_reason if hasattr(chunk.choices[0], 'finish_reason') else None
                        
                        # Check for tool calls in the delta
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            # We're receiving a tool call
                            collecting_tool_call = True
                            
                            # Store the raw tool call data
                            tool_call_chunks.append(delta.tool_calls)
                            
                            # Process the tool call
                            for tool_call in delta.tool_calls:
                                # Get the tool call ID if this is the first chunk
                                if not tool_call_id and hasattr(tool_call, 'id'):
                                    tool_call_id = tool_call.id
                                    print(f"\nTOOL CALL ID: {tool_call_id}")
                                
                                # Get the function information
                                if hasattr(tool_call, 'function'):
                                    # Extract tool name
                                    if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                        if not tool_name:
                                            tool_name = tool_call.function.name
                                            print(f"\nTOOL NAME: {tool_name}")
                                    
                                    # Extract and accumulate arguments
                                    if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                        tool_args += tool_call.function.arguments
                        
                        # Check for regular content
                        elif hasattr(delta, 'content') and delta.content:
                            content = delta.content
                            print(content, end="", flush=True)
                            assistant_message += content
                        
                        # Check if we've reached the end of a tool call
                        if collecting_tool_call and finish_reason:
                            print(f"\nTOOL CALL COMPLETE - Finish reason: {finish_reason}")
                            print(f"ACCUMULATED ARGS: {tool_args}")
                            
                            # Format the tool call into the expected format
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
                        
                        # Ensure the tool call format is correct for process_tool_call
                        # The tool call parser expects the </tool_call> tag
                        formatted_tool_call = assistant_message + "</tool_call>"
                        
                        # Process the tool call
                        tool_call_result = await process_tool_call(formatted_tool_call, mcp)
                        
                        if tool_call_result['tool_call_made']:
                            # Update the message to persist the properly formatted tool call
                            message['content'] = formatted_tool_call
                            
                            print("TOOL CALL --------------------------------------------------------")
                            print(tool_call_result['formatted_result'])
                            print("END CALL --------------------------------------------------------")
                            
                            # Add the tool result to the conversation
                            messages.append({
                                "role": "assistant",
                                "content": f"Tool result: {tool_call_result['formatted_result']}",
                            })
                        else:
                            print(f"\nError processing tool call: {tool_call_result.get('error', 'Unknown error')}")
                            break
                    else:
                        # Process any traditional tool calls in the message format (for backward compatibility)
                        tool_call_result = await process_tool_call(assistant_message + "</tool_call>", mcp)
                        if tool_call_result['tool_call_made']:
                            message['content'] += "</tool_call></tool_call></tool_call></tool_call></tool_call></tool_call>"  # persist the end tag
                            print("LEGACY TOOL CALL --------------------------------------------------------")
                            print(tool_call_result['formatted_result'])
                            print("END LEGACY CALL --------------------------------------------------------")
                            messages.append({
                                "role": "assistant",
                                "content": f"Tool result: {tool_call_result['formatted_result']}",
                            })
                        else:
                            break
                    
                    #if tool_call_result['tool_call_made']:
                    #    # Print tool call details
                    #    print_tool_header()
                    #    print_tool_details(
                    #        tool_call_result['tool_name'],
                    #        tool_call_result['arguments'],
                    #        tool_call_result['result']
                    #    )
                    #    
                    #    # Add the tool result as a system message
                    #    messages.append({
                    #        "role": "system", 
                    #        "content": tool_call_result['formatted_result']
                    #    })
                    #    
                    #    # Add a system message instructing the assistant to continue solving the problem
                    #    messages.append({
                    #        "role": "system", 
                    #        "content": "Continue solving the user's request autonomously based on these tool results. If the results indicate an error or unexpected outcome, fix your approach and try again."
                    #    })
                    #    
                    #    # Get LLM's response to the tool result
                    #    print_assistant_header(responding_to_tool=True)
                    #    assistant_response = ""
                    #    async for chunk in stream_llm_response(
                    #        messages=messages,
                    #        model=model,
                    #        api_key=api_key,
                    #        api_base_url=api_base_url
                    #    ):
                    #        # Process the raw LiteLLM chunk
                    #        if hasattr(chunk, 'choices') and chunk.choices:
                    #            # Extract the content from the choices
                    #            delta = chunk.choices[0].delta
                    #            if hasattr(delta, 'content') and delta.content:
                    #                content = delta.content
                    #                # Skip printing if assistant is repeating function results
                    #                skip_print = False
                    #                if content.strip().startswith("<tool_results>"):
                    #                    skip_print = True
                    #                    
                    #                if not skip_print:
                    #                    print(content, end="", flush=True)
                    #                assistant_response += content
                    #    
                    #    # Add the tool response to message history
                    #    if assistant_response:
                    #        # Update the assistant's response in the message history
                    #        messages.append({
                    #            "role": "assistant", 
                    #            "content": assistant_response
                    #        })
                    #elif 'error' in tool_call_result:
                    #    # There was an error processing the tool call
                    #    print(f"\nError: {tool_call_result['error']}")
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
