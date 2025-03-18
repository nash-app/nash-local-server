import asyncio
import json
from app.llm_handler import configure_llm, stream_llm_response
from app.mcp_handler import MCPHandler
from app.prompts import get_system_prompt
from app.streaming import StreamProcessor
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
                
                # Initialize the stream processor
                processor = StreamProcessor()
                
                # Process the streaming response
                async for chunk in stream_llm_response(
                    messages=messages,
                    model=model,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    tools=tools
                ):
                    # Process each chunk and get any displayable content
                    display_text = processor.process_chunk(chunk)
                    
                    # If there's text to display, print it
                    if display_text:
                        print(display_text, end="", flush=True)
                
                # Get the message to add to history
                message = processor.get_message_for_history()
                
                # Add the message if we have one
                if message:
                    messages.append(message)
                    
                    # If a tool call was detected, execute it
                    if processor.is_tool_call_detected():
                        print("\nEXECUTING TOOL CALL:")
                        print(f"Tool: {processor.tool_use_info['name']}")
                        print(f"Arguments: {json.dumps(processor.tool_use_info['input'], indent=2)}")
                        
                        # Execute the tool and get the result
                        tool_result = await processor.execute_tool(mcp)
                        
                        # Print the result
                        print("TOOL RESULT --------------------------------------------------------")
                        print(tool_result['result_text'])
                        print("END RESULT --------------------------------------------------------")
                        
                        # Add the result message to history if successful
                        if tool_result['success'] and tool_result['result_message']:
                            messages.append(tool_result['result_message'])
                    else:
                        # No tool call, just break out of the loop for next user input
                        break
                else:
                    # No message to add
                    print("No valid assistant message or tool use detected.")
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