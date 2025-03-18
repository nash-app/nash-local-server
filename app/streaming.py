import json


class StreamProcessor:
    """
    Processes streaming responses and collects content/tool calls.
    
    This class handles the processing of streamed chunks from an LLM,
    extracting both regular text content and tool calls, and providing
    a clean interface for the rest of the application to use.
    """
    
    def __init__(self):
        """Initialize the stream processor."""
        # Content from the assistant
        self.assistant_message = ""
        
        # Tool call tracking
        self.tool_use_info = None
        self.tool_call_chunks = []
        self.collecting_tool_call = False
        self.tool_call_id = None
        self.tool_name = None
        self.tool_args = ""
        
        # Stream state
        self.is_complete = False
        self.finish_reason = None
    
    def process_chunk(self, chunk):
        """
        Process a single chunk from the stream.
        
        Args:
            chunk: The raw chunk from the LLM streaming response
            
        Returns:
            str: Any text content that should be displayed immediately,
                 or None if no displayable content in this chunk
        """
        display_text = None
        
        # Check if the chunk has choices
        if hasattr(chunk, 'choices') and chunk.choices:
            # Extract the delta from the choices
            delta = chunk.choices[0].delta
            
            # Check if this is the final chunk with a finish reason
            if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason:
                self.finish_reason = chunk.choices[0].finish_reason
                self.is_complete = True
            
            # Check for tool calls in the delta
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                # We're receiving a tool call
                self.collecting_tool_call = True
                
                # Store the raw tool call data
                self.tool_call_chunks.append(delta.tool_calls)
                
                # Process the tool call
                for tool_call in delta.tool_calls:
                    # Get the tool call ID if this is the first chunk
                    if not self.tool_call_id and hasattr(tool_call, 'id'):
                        self.tool_call_id = tool_call.id
                    
                    # Get the function information
                    if hasattr(tool_call, 'function'):
                        # Extract tool name
                        if hasattr(tool_call.function, 'name') and tool_call.function.name:
                            if not self.tool_name:
                                self.tool_name = tool_call.function.name
                        
                        # Extract and accumulate arguments
                        if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                            self.tool_args += tool_call.function.arguments
            
            # Check for regular content
            elif hasattr(delta, 'content') and delta.content:
                content = delta.content
                display_text = content
                self.assistant_message += content
            
            # Check if we've completed a tool call
            if self.collecting_tool_call and self.finish_reason:
                # Process the completed tool call
                try:
                    parsed_args = json.loads(self.tool_args)
                    self.tool_use_info = {
                        "id": self.tool_call_id,
                        "name": self.tool_name,
                        "input": parsed_args
                    }
                except json.JSONDecodeError:
                    # If we can't parse the arguments, just use them as a string
                    self.tool_use_info = {
                        "id": self.tool_call_id,
                        "name": self.tool_name,
                        "input": {"raw_args": self.tool_args}
                    }
        
        return display_text
    
    def is_tool_call_detected(self):
        """Check if a complete tool call was detected in the stream."""
        return self.tool_use_info is not None
    
    def get_message_for_history(self):
        """
        Get the formatted message to add to conversation history.
        
        Returns:
            dict: Message object in the format expected by the LLM API
        """
        # If we have a tool call, format it as plain text to avoid validation issues
        if self.is_tool_call_detected():
            tool_call_text = f"\nI'm using the '{self.tool_use_info['name']}' tool with these parameters:\n"
            tool_call_text += json.dumps(self.tool_use_info["input"], indent=2)
            
            if self.assistant_message:
                return {
                    "role": "assistant",
                    "content": self.assistant_message + tool_call_text
                }
            else:
                return {
                    "role": "assistant",
                    "content": tool_call_text
                }
        # Otherwise just return the regular assistant message
        elif self.assistant_message:
            return {
                "role": "assistant",
                "content": self.assistant_message
            }
        else:
            return None
    
    async def execute_tool(self, mcp_handler):
        """
        Execute the detected tool call if any.
        
        Args:
            mcp_handler: The MCPHandler instance to use for tool execution
            
        Returns:
            dict: Information about the tool execution result
                {
                    'success': bool,
                    'result_text': str,
                    'result_message': dict  # Message formatted for conversation history
                }
        """
        if not self.is_tool_call_detected():
            return {
                'success': False,
                'result_text': "No tool call detected",
                'result_message': None
            }
        
        try:
            # Execute the tool with the parsed arguments
            tool_result = await mcp_handler.call_tool(
                self.tool_use_info['name'], 
                arguments=self.tool_use_info['input']
            )
            
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
            
            # Format a message for the conversation history
            result_message = {
                "role": "user",
                "content": f"Results from executing {self.tool_use_info['name']}:\n\n{result_text}"
            }
            
            return {
                'success': True,
                'result_text': result_text,
                'result_message': result_message
            }
        
        except Exception as e:
            error_message = str(e)
            
            # Format an error message for the conversation history
            result_message = {
                "role": "user",
                "content": f"Error when executing {self.tool_use_info['name']}: {error_message}"
            }
            
            return {
                'success': False,
                'result_text': f"Error: {error_message}",
                'result_message': result_message
            }
