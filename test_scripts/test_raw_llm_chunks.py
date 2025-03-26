import asyncio
import json
import sys
from test_scripts.api_credentials import get_api_credentials
from app.llm_handler import stream_llm_response
from app.prompts import get_system_prompt
from app.mcp_handler import MCPHandler


async def capture_chunks(message, output_file):
    """Capture raw LLM chunks to a file with minimal processing."""
    # Get credentials
    api_key, api_base_url, model = get_api_credentials()

    mcp = MCPHandler.get_instance()
    await mcp.initialize()
    tools = await mcp.list_tools_litellm()

    # Create message structure with system prompt
    system_prompt = get_system_prompt()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]

    # Collect chunks
    chunks = []
    try:
        async for chunk in stream_llm_response(
            messages=messages, model=model, api_key=api_key, api_base_url=api_base_url, tools=tools
        ):
            chunks.append(json.loads(chunk.to_json()))

        # Write to file
        with open(output_file, "w") as f:
            json.dump(chunks, f, indent=2)

        await mcp.close()

    except Exception as e:
        print(f"Error: {e}")
        await mcp.close()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_raw_llm_chunks.py <OUTPUT_FILE> <MESSAGE>")
        sys.exit(1)

    output_file = sys.argv[1]
    message = " ".join(sys.argv[2:])

    # Run capture
    await capture_chunks(message, output_file)
    print(f"Chunks saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
