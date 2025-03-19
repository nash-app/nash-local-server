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
        self.session_id = None
        self.messages = []
        
        # Add system message
        self.messages.append({"role": "system", "content": "You are a helpful AI assistant."})
    
    async def stream_chat(self, user_message):
        """Send a message to the server and stream the response."""
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})
        
        # Prepare request payload
        payload = {
            "messages": [{"role": msg["role"], "content": msg["content"]} for msg in self.messages],
            "model": self.model,
            "api_key": self.api_key,
            "api_base_url": self.api_base_url
        }
        
        # Add session ID if we have one
        if self.session_id:
            payload["session_id"] = self.session_id
        
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
                assistant_message = ""
                assistant_content_mode = False
                
                # Create a response reader
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Check for SSE prefix and extract data
                    if line.startswith('data: '):
                        data = line[6:]
                        
                        # Check for end of stream marker
                        if data == '[DONE]':
                            break
                        
                        try:
                            event = json.loads(data)
                            
                            # Handle session ID
                            if 'session_id' in event:
                                self.session_id = event['session_id']
                                print(f"\n[Session ID: {self.session_id}]", end="")
                            
                            # Handle regular content
                            if 'content' in event:
                                content = event['content']
                                if not assistant_content_mode:
                                    assistant_content_mode = True
                                print(content, end="", flush=True)
                                assistant_message += content
                            
                            # Handle status updates
                            if 'status' in event:
                                print(f"\n[Status: {event['status']}]")
                            
                            # Handle errors
                            if 'error' in event:
                                print(f"\n[Error: {event['error']}]")
                            
                            # Handle tool calls - just print notification
                            if 'tool_calls' in event:
                                if not assistant_content_mode:
                                    print("\n[Tool Call Detected]", end="")
                            
                            # Handle tool execution - just print notification
                            if 'executing_tool' in event:
                                executing_tool = event['executing_tool']
                                print(f"\n[Executing Tool: {executing_tool}]")
                            
                            # Handle tool results - just print notification
                            if 'tool_result' in event:
                                result = event['tool_result']
                                print(f"\n[Tool Result: {result['name']}]")
                            
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON: {data}")
                
                # Add assistant message to history
                if assistant_message:
                    self.messages.append({"role": "assistant", "content": assistant_message})

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