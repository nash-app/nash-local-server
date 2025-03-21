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

        streamable_content = {
            "content": "",
            "tool_name": "",
            "tool_args": "",
        }

        if not hasattr(chunk, "choices") and not chunk.choices:
            return

        for choice in chunk.choices:
            if not hasattr(choice, "delta") and not choice.delta:
                continue

            if hasattr(choice.delta, "content") and choice.delta.content:
                self.content += choice.delta.content
                streamable_content["content"] = self.content

            if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    # Check if there's an id in the latest tool_call chunk which means a new tool call is starting
                    if hasattr(tool_call, "id") and tool_call.id:
                        self.tool_calls.append(json.loads(tool_call.to_json()))
                        tool_call = self.tool_calls[-1]
                        if "function" in tool_call:
                            if "name" in tool_call["function"]:
                                streamable_content["tool_name"] = tool_call["function"]["name"]
                            if "arguments" in tool_call["function"]:
                                streamable_content["tool_args"] = tool_call["function"]["arguments"]
                    else:
                        # Update the last tool call with the new delta
                        tool_call_to_update = self.tool_calls[-1]
                        if tool_call.function.name:
                            tool_call_to_update["function"]["name"] += tool_call.function.name
                            streamable_content["tool_name"] = tool_call_to_update["function"]["name"]
                        if tool_call.function.arguments:
                            tool_call_to_update["function"]["arguments"] += tool_call.function.arguments
                            streamable_content["tool_args"] = tool_call_to_update["function"]["arguments"]

            # Check if we're done streaming and flip the flag
            if hasattr(choice, "finish_reason") and choice.finish_reason:
                self.is_streaming = False
                self.finish_reason = choice.finish_reason
                # Convert args to a dictionary
                for tool_call in self.tool_calls:
                    tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])

        return streamable_content

    def get_assistant_message(self):
        pass

    def execute_tool_calls_and_get_user_message(self):
        pass
