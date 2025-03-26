import copy
import json


class StreamProcessor:
    """
    Processes streaming responses and collects content/tool calls.

    This class handles the processing of streamed chunks from an LLM,
    extracting both regular text content and tool calls, and providing
    a clean interface for the rest of the application to use.
    """

    def __init__(self, mcp=None):
        self.mcp = mcp
        self.content = ""
        self.tool_calls = []
        self.is_streaming = False
        self.finish_reason = None

    def process_chunk(self, chunk):
        self.is_streaming = True

        streamable_content = {"content": None, "tool_name": None, "tool_args": None}

        if not hasattr(chunk, "choices") and not chunk.choices:
            return

        for choice in chunk.choices:
            if not hasattr(choice, "delta") and not choice.delta:
                continue

            if hasattr(choice.delta, "content") and choice.delta.content:
                self.content += choice.delta.content
                streamable_content["content"] = choice.delta.content

            if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    # Check if there's an id in the latest tool_call chunk which means a new tool call is starting
                    if hasattr(tool_call, "id") and tool_call.id:
                        try:
                            tool_call_json = tool_call.to_json()
                            parsed_tool_call = json.loads(tool_call_json)
                            self.tool_calls.append(parsed_tool_call)
                            # Manage streamable content for the new tool call
                            if hasattr(tool_call, "function") and tool_call.function:
                                if hasattr(tool_call.function, "name") and tool_call.function.name:
                                    streamable_content["tool_name"] = tool_call.function.name
                                if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                                    streamable_content["tool_args"] = tool_call.function.arguments
                        except json.JSONDecodeError as e:
                            print(f"Error parsing tool call JSON: {e}")
                    else:
                        # Update the last tool call with the new delta
                        tool_call_to_update = self.tool_calls[-1]
                        if tool_call.function.name:
                            tool_call_to_update["function"]["name"] += tool_call.function.name
                            streamable_content["tool_name"] = tool_call.function.name
                        if tool_call.function.arguments:
                            tool_call_to_update["function"]["arguments"] += tool_call.function.arguments
                            streamable_content["tool_args"] = tool_call.function.arguments

            # Check if we're done streaming and flip the flag
            if hasattr(choice, "finish_reason") and choice.finish_reason:
                self.is_streaming = False
                self.finish_reason = choice.finish_reason

        return streamable_content

    def get_assistant_message(self):
        # Assistant message format with tool calls
        # {
        #     'content': 'content',
        #     'role': 'assistant',
        #     'tool_calls': [
        #         {
        #             'index': 1,
        #             'function': {
        #                 'arguments': '{"location": "Boston", "unit": "fahrenheit"}',
        #                 'name': 'get_current_weather'
        #             },
        #            'id': 'toolu_01HxUDqLmTT23Tu1AQBSXXhg',
        #            'type': 'function'
        #         }
        #     ],
        #     'function_call': None
        # }
        if not self.tool_calls:
            return {
                "content": self.content,
                "role": "assistant",
            }
        # Tool calls are present, so we need to return a message with tool calls
        assistant_message = {
            "content": self.content,
            "role": "assistant",
            "tool_calls": copy.deepcopy(self.tool_calls),
            "function_call": None,
        }
        return assistant_message

    async def execute_tool_calls_and_get_user_message(self):
        if not self.tool_calls:
            return []

        messages = []
        # Execute the tool calls
        for tool_call in self.tool_calls:
            try:
                arguments = json.loads(tool_call["function"]["arguments"])

                # Use stringify arguments for tool call
                result = await self.mcp.call_tool(tool_call["function"]["name"], arguments=arguments)

                # Create the message from the result
                content = result.content[0].text if result.content and len(result.content) > 0 else "No result content"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": content,  # TODO: Handle multiple results and different content types
                        "is_error": result.isError,
                    }
                )
            except Exception as e:
                print(f"Error executing tool call: {e}")
                # Create an error message
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": f"Error executing tool: {str(e)}",
                        "is_error": True,
                    }
                )

        return messages
