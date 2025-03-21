import asyncio
import json
import aiohttp
import os
from dotenv import load_dotenv
from test_scripts.api_credentials import get_api_credentials, print_credentials_info
from test_scripts.message_display import print_messages, print_user_prompt, print_assistant_header

# Server URL
SERVER_URL = "http://localhost:6274"


class HttpClient:
    """HTTP client for communicating with the Nash LLM Server."""

    def __init__(self, api_key, api_base_url, model):
        """Initialize the HTTP client with API credentials."""
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.model = model
        self.messages = []

    async def stream_chat(self, user_message):
        """Send a message to the server and stream the response."""
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})

        # Prepare request payload
        payload = {
            "messages": self.messages,
            "model": self.model,
            "api_key": self.api_key,
            "api_base_url": self.api_base_url,
        }

        # Handle state for what part of the stream we're in
        in_content_stream = False
        in_tool_name_stream = False
        in_tool_args_stream = False

        # Send the request and process the streaming response
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SERVER_URL}/v1/chat/completions/stream",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                # Check response status
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: HTTP {response.status} - {error_text}")
                    return

                # Process streaming response
                async for line in response.content:
                    line = line.decode("utf-8").strip()

                    # Skip empty lines
                    if not line:
                        continue

                    # Check for SSE prefix and extract data
                    if line.startswith("data: "):
                        data = line[6:]

                        # Check for end of stream marker
                        if data == "[DONE]":
                            break

                        try:
                            event = json.loads(data)

                            # Handle error
                            if "error" in event:
                                print(f"\n[Error: {event['error']}]")
                                continue

                            # Handle different event types
                            event_type = event.get("type")

                            if event_type == "stream":
                                # Handle content stream
                                if event["content"]:
                                    # Flip these bits if we were previously in a tool name or tool args stream
                                    if in_tool_name_stream or in_tool_args_stream:
                                        in_tool_name_stream = False
                                        in_tool_args_stream = False

                                    if not in_content_stream:
                                        print("\n[CONTENT]")
                                        in_content_stream = True

                                    print(event["content"], end="", flush=True)

                                # Handle tool name stream
                                if event["tool_name"]:
                                    if in_content_stream or in_tool_args_stream:
                                        in_content_stream = False
                                        in_tool_args_stream = False

                                    if not in_tool_name_stream:
                                        print("\n[TOOL_NAME]")
                                        in_tool_name_stream = True

                                    print(event["tool_name"], end="", flush=True)

                                # Handle tool args stream
                                if event["tool_args"]:
                                    if in_content_stream or in_tool_name_stream:
                                        in_content_stream = False
                                        in_tool_name_stream = False

                                    if not in_tool_args_stream:
                                        print("\n[TOOL_ARGS]")
                                        in_tool_args_stream = True

                                    print(event["tool_args"], end="", flush=True)

                            elif event_type == "tool_result":
                                # Handle tool result
                                if event["tool_result"]:
                                    print(f"\n[TOOL_RESULT] {event['tool_result']}")

                            elif event_type == "new_raw_llm_messages":
                                # Add new messages to our history
                                if event.get("new_raw_llm_messages"):
                                    for msg in event["new_raw_llm_messages"]:
                                        self.messages.append(msg)

                        except json.JSONDecodeError:
                            print(f"Error parsing JSON: {data}")


async def main():
    """Main function for the HTTP client test script."""
    # Get API credentials from environment
    try:
        api_key, api_base_url, model = get_api_credentials()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set the required environment variables in a .env file:")
        print("PROVIDER_API_KEY=your_api_key")
        print("PROVIDER_API_BASE=your_api_base_url")
        print("PROVIDER_MODEL=your_model")
        return

    # Print credentials info
    print_credentials_info(api_key, api_base_url, model)
    print(f"\nConnecting to server at: {SERVER_URL}")

    # Check server health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SERVER_URL}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"Server health: {health['status']}")
                else:
                    print(f"Server health check failed: HTTP {response.status}")
                    return
    except aiohttp.ClientError as e:
        print(f"Could not connect to server: {e}")
        print("Make sure the server is running at the specified URL.")
        return

    # Create the HTTP client
    client = HttpClient(api_key, api_base_url, model)

    # Main chat loop
    try:
        while True:
            # Get user input
            print_user_prompt()
            user_input = input("").strip()

            # Check for special commands
            if user_input.lower() in ["quit", "exit", "bye"]:
                break
            if user_input.lower() == "messages":
                print_messages(client.messages)
                continue

            # Stream chat with the server
            print_assistant_header()
            await client.stream_chat(user_input)
            print()  # Add newline after response

    except KeyboardInterrupt:
        print("\nChat session terminated by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    print("\nChat session ended.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
