import requests
import sys
import json
from typing import List, Dict, Optional
from test_scripts.api_credentials import get_api_credentials, print_credentials_info


class SimpleConversation:
    def __init__(self):
        self.messages = []
        self.api_key = None
        self.api_base_url = None
        self.model = None

    def add_message(self, role, content):
        """Add a message to the conversation history.

        If content is a string, it will be wrapped in a simple message structure.
        If content is already a dict, it's assumed to be a properly formatted message.
        """
        if isinstance(content, dict):
            # Assume it's already a proper message structure
            message = content
            # Ensure the role is set correctly if not already present
            if "role" not in message:
                message["role"] = role
            self.messages.append(message)
        else:
            # Simple string content, create a basic message
            self.messages.append({"role": role, "content": content})

    def get_messages(self):
        return self.messages.copy()

    def print_messages(self):
        """Pretty print the current conversation messages with detailed structure."""
        print("\n=== CONVERSATION MESSAGES ===")
        for i, message in enumerate(self.messages):
            print(f"MESSAGE #{i+1}")
            print(json.dumps(message, indent=2))
            print("-" * 50)
        print("=== END OF MESSAGES ===\n")


def stream_raw_chunks(messages, model, api_key, api_base_url):
    """Stream raw chunks from the server with no processing.

    Returns:
        dict: A dictionary containing the complete assistant response data,
              either as text content or a tool call structure
    """
    print("\n=== RAW SERVER RESPONSE START ===")

    payload = {
        "messages": messages,
        "model": model,
        "api_key": api_key,
        "api_base_url": api_base_url,
        "session_id": None,
    }

    try:
        # Make the request with streaming enabled
        response = requests.post(
            "http://localhost:6274/v1/chat/completions/stream",
            json=payload,
            stream=True,
        )

        # Track the assistant's response components
        full_content = ""
        tool_calls = []
        tool_results = []
        is_tool_call = False
        tool_call_data = {}

        # Simply output every raw chunk
        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                chunk_count += 1
                chunk_str = chunk.decode("utf-8")
                print(f"CHUNK #{chunk_count}: {chunk_str}")

                # Try to extract data from the chunk
                if chunk_str.startswith("data: "):
                    try:
                        data = json.loads(chunk_str[6:])

                        # Extract regular content
                        if "content" in data:
                            full_content += data["content"]

                        # Track tool calls
                        if "tool_calls" in data:
                            is_tool_call = True
                            for tool_call in data["tool_calls"]:
                                # Find or create a tool call entry
                                if tool_call.get("id") and tool_call["id"] not in [
                                    t.get("id") for t in tool_calls if t.get("id")
                                ]:
                                    # New tool call
                                    tool_calls.append(
                                        {
                                            "id": tool_call.get("id"),
                                            "function": {
                                                "name": tool_call.get("function", {}).get("name"),
                                                "arguments": tool_call.get("function", {}).get("arguments", ""),
                                            },
                                        }
                                    )
                                else:
                                    # Add to existing tool call
                                    for t in tool_calls:
                                        if t.get("id") == tool_call.get("id") or (
                                            not tool_call.get("id") and t == tool_calls[-1]
                                        ):
                                            # Append arguments
                                            if tool_call.get("function", {}).get("arguments"):
                                                t["function"]["arguments"] += tool_call["function"]["arguments"]
                                            # Update name if provided
                                            if tool_call.get("function", {}).get("name"):
                                                t["function"]["name"] = tool_call["function"]["name"]

                        # Capture tool results
                        if "tool_result" in data:
                            tool_results.append(data["tool_result"])

                    except json.JSONDecodeError:
                        pass

        print(f"\nTotal chunks: {chunk_count}")
        print("=== RAW SERVER RESPONSE END ===")

        # Prepare the complete response data
        response_data = {"role": "assistant"}

        # If we detected tool calls, structure the response with tool_calls
        if is_tool_call and tool_calls:
            try:
                # Parse and format the tool call arguments as proper JSON
                for tool_call in tool_calls:
                    # Only try to parse if we have string arguments
                    if isinstance(tool_call["function"]["arguments"], str) and tool_call["function"]["arguments"]:
                        try:
                            # Check if arguments is already valid JSON
                            tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            # If not valid JSON, keep as string
                            pass

                # Store the tool results separately for later use, but don't include them in the message
                extracted_tool_results = tool_results

                # Add the tool call structure to the response WITHOUT the tool results
                # (tool results will be added as a separate user message)
                response_data["content"] = full_content if full_content else ""
                response_data["tool_calls"] = tool_calls

                # Store tool results in metadata for the stream_raw_chunks function
                # but don't include them in the actual message
                response_data["_extracted_tool_results"] = extracted_tool_results

                return response_data
            except Exception as e:
                print(f"Error processing tool call: {e}")
                # Fall back to text content
                return {"role": "assistant", "content": full_content}
        else:
            # Regular text response
            return {"role": "assistant", "content": full_content}

    except Exception as e:
        print(f"Error: {e}")
        return {"role": "assistant", "content": f"[Error: {str(e)}]"}


def chat_loop():
    # Initialize conversation
    conversation = SimpleConversation()

    # Get API configuration
    try:
        api_key, api_base_url, model = get_api_credentials()
        conversation.api_key = api_key
        conversation.api_base_url = api_base_url
        conversation.model = model

        print("\n=== ULTRA SIMPLE RAW CLIENT ===")
        print(f"Using model: {model}")
        print(f"API Base: {api_base_url}")
        print(f"API Key: {api_key[:4]}...{api_key[-4:]}")
        print("\nType 'exit' to quit.")
        print("Type 'messages' to see the current conversation.")
        print("-" * 50)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    message_count = 0

    while True:
        try:
            message_count += 1
            print(f"\n=== MESSAGE {message_count} ===")

            # Get user input
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            if user_input.lower() == "messages":
                conversation.print_messages()
                continue

            # Add user message to history
            conversation.add_message("user", user_input)

            # Stream raw response chunks and get the complete response data structure
            assistant_response = stream_raw_chunks(
                conversation.get_messages(), conversation.model, conversation.api_key, conversation.api_base_url
            )

            # Add the actual response to conversation history with proper structure
            if assistant_response:
                print("\n=== EXTRACTED RESPONSE ===")
                print(json.dumps(assistant_response, indent=2))
                print("=== END OF RESPONSE ===")

                # Add the full message structure to the conversation (without tool_results)
                # Create a clean copy without internal metadata
                assistant_message = {k: v for k, v in assistant_response.items() if not k.startswith("_")}
                conversation.messages.append(assistant_message)

                # If there are tool results stored in the metadata and they're meant to be
                # sent back to the LLM as user messages (e.g., tool execution results), add those too
                if "_extracted_tool_results" in assistant_response and assistant_response.get("tool_calls"):
                    for tool_result in assistant_response["_extracted_tool_results"]:
                        # Create a user message with the tool result
                        tool_name = tool_result.get("name", "unknown_tool")
                        result_text = tool_result.get("result", "No result")

                        tool_result_message = {
                            "role": "user",
                            "content": f"Results from executing {tool_name}:\n\n{result_text}",
                        }

                        print("\n=== ADDING TOOL RESULT AS USER MESSAGE ===")
                        print(json.dumps(tool_result_message, indent=2))
                        print("=== END OF TOOL RESULT MESSAGE ===")

                        # Add the tool result as a user message
                        conversation.messages.append(tool_result_message)
            else:
                # Fallback in case we couldn't extract the response
                print("\nWarning: Could not extract response content")
                conversation.add_message("assistant", {"role": "assistant", "content": "[Response extraction failed]"})

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    chat_loop()
