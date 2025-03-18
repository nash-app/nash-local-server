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
            
                async for chunk in stream_llm_response(
                    messages=messages,
                    model=model,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    tools=tools
                ):
                    # Process the raw LiteLLM chunk
                    if hasattr(chunk, 'choices') and chunk.choices:
                        # Extract the content from the choices
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            content = delta.content
                            print(content, end="", flush=True)
                            assistant_message += content
            
                # Add assistant response to history 
                if assistant_message:
                    # Add the original assistant message to the history first
                    message = {
                        "role": "assistant",
                        "content": assistant_message
                    }
                    messages.append(message)

                    # Process any tool calls in the message
                    tool_call_result = await process_tool_call(assistant_message + "</tool_call>", mcp)
                    if tool_call_result['tool_call_made']:
                        message['content'] += "</tool_call></tool_call></tool_call></tool_call></tool_call></tool_call>"  # persist the end tag in the assistant message because the termination string isn't included and this is a case where the termination string was hit
                        print("TOOL CALL --------------------------------------------------------")
                        print(tool_call_result['formatted_result'])
                        print("END CALL --------------------------------------------------------")
                        # message['content'] += f"\n{tool_call_result['formatted_result']}"
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
