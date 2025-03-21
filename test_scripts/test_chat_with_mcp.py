import asyncio
from app.llm_handler import configure_llm, stream_llm_response
from app.mcp_handler import MCPHandler
from app.prompts import get_system_prompt
from app.stream_processor import StreamProcessor
from test_scripts.api_credentials import get_api_credentials, print_credentials_info
from test_scripts.message_display import print_messages, print_user_prompt, print_assistant_header


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

    messages.append({"role": "system", "content": system_prompt})

    try:
        while True:
            print("Messages")
            import pprint

            pprint.pprint(messages[1:])
            print()
            # Get user input
            print_user_prompt()
            user_input = input("").strip()
            if user_input.lower() in ["quit", "exit", "bye"]:
                break
            if user_input.lower() in ["messages"]:
                print_messages(messages)
                continue

            # Add user message to history
            messages.append({"role": "user", "content": user_input})

            while True:
                # Stream AI response
                print_assistant_header()

                # Initialize the stream processor
                processor = StreamProcessor(mcp)

                # Handle state for what part of the stream we're in
                in_content_stream = False
                in_tool_name_stream = False
                in_tool_args_stream = False

                # Process the streaming response
                async for chunk in stream_llm_response(
                    messages=messages, model=model, api_key=api_key, api_base_url=api_base_url, tools=tools
                ):
                    # Process each chunk and get displayable content and tool call data
                    streamable_content = processor.process_chunk(chunk)

                    # Handle content stream
                    if streamable_content["content"]:
                        # Flip these bits if we were previously in a tool name or tool args stream
                        if in_tool_name_stream or in_tool_args_stream:
                            in_tool_name_stream = False
                            in_tool_args_stream = False

                        if not in_content_stream:
                            print("\n[CONTENT]")
                            in_content_stream = True

                        print(streamable_content["content"], end="", flush=True)

                    # Handle tool name stream
                    if streamable_content["tool_name"]:
                        if in_content_stream or in_tool_args_stream:
                            in_content_stream = False
                            in_tool_args_stream = False

                        if not in_tool_name_stream:
                            print("\n[TOOL_NAME]")
                            in_tool_name_stream = True

                        print(streamable_content["tool_name"], end="", flush=True)

                    # Handle tool args stream
                    if streamable_content["tool_args"]:
                        if in_content_stream or in_tool_name_stream:
                            in_content_stream = False
                            in_tool_name_stream = False

                        if not in_tool_args_stream:
                            print("\n[TOOL_ARGS]")
                            in_tool_args_stream = True

                        print(streamable_content["tool_args"], end="", flush=True)

                assistant_message = processor.get_assistant_message()
                messages.append(assistant_message)
                if processor.tool_calls:
                    messages_for_tool_call_results = await processor.execute_tool_calls_and_get_user_message()
                    messages.append(messages_for_tool_call_results[0])  # TODO: Handle multiple tool call results
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
