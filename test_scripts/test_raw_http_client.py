import requests
import sys
from typing import List, Dict, Optional
from test_scripts.api_credentials import get_api_credentials, print_credentials_info


class SimpleConversation:
    def __init__(self):
        self.messages = []
        self.api_key = None
        self.api_base_url = None
        self.model = None

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_messages(self):
        return self.messages.copy()


def stream_raw_chunks(messages, model, api_key, api_base_url):
    """Stream raw chunks from the server with no processing."""
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

        # Simply output every raw chunk
        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                chunk_count += 1
                print(f"CHUNK #{chunk_count}: {chunk.decode('utf-8')}")

        print(f"\nTotal chunks: {chunk_count}")
        print("=== RAW SERVER RESPONSE END ===")
        
    except Exception as e:
        print(f"Error: {e}")


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
                
            # Add user message to history
            conversation.add_message("user", user_input)
            
            # Stream raw response chunks
            stream_raw_chunks(
                conversation.get_messages(),
                conversation.model,
                conversation.api_key,
                conversation.api_base_url
            )
            
            # Add a placeholder assistant response to maintain conversation
            conversation.add_message("assistant", "[Response saved to history]")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    chat_loop()