import asyncio
import os
from dotenv import load_dotenv
from app.mcp_handler import MCPHandler
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables before MCPHandler is instantiated
load_dotenv()


def format_tool_result(result):
    """Format MCP tool result for display"""
    if hasattr(result, "content"):
        # Handle structured responses
        for content in result.content:
            if content.type == "text":
                return content.text
    # Fallback to string representation
    return str(result)


def format_tool_description(tool):
    """Format a tool's description for display"""
    # Get first line of description for summary
    desc = tool.description.strip().split("\n")[0]
    # Truncate if too long
    if len(desc) > 100:
        desc = desc[:97] + "..."
    return f"{tool.name}: {desc}"


async def test_mcp():
    # Get singleton instance
    mcp = MCPHandler.get_instance()

    try:
        print("Initializing MCP handler...")
        await mcp.initialize()

        print("\nAvailable MCP Tools:")
        print("-" * 80)
        tools = await mcp.list_tools()
        for tool in tools.tools:
            print(format_tool_description(tool))
        print("-" * 80)

        # Interactive tool testing loop
        while True:
            print("\nEnter a tool name to test (or 'quit' to exit):", end=" ")
            tool_name = input().strip()

            if tool_name.lower() in ["quit", "exit", "bye"]:
                break

            try:
                print(f"\nTesting tool: {tool_name}")
                print("-" * 80)
                result = await mcp.call_tool(tool_name)
                formatted_result = format_tool_result(result)
                print(formatted_result)
                print("-" * 80)
            except Exception as e:
                print(f"Error calling tool: {e}")

    except Exception as e:
        print(f"Error during MCP testing: {e}")
    finally:
        print("\nClosing MCP handler...")
        await mcp.close()


async def test_basic_mcp():
    print("Starting basic MCP test...")

    # Setup server parameters
    nash_path = os.getenv("NASH_PATH")
    if not nash_path:
        raise ValueError("NASH_PATH environment variable not set")

    server_params = StdioServerParameters(
        command=os.path.join(nash_path, ".venv/bin/mcp"),
        args=["run", os.path.join(nash_path, "src/nash_mcp/server.py")],
        env=None,
    )

    print("Connecting to MCP server...")
    client = stdio_client(server_params)

    async with client as (read, write):
        print("Creating session...")
        session = ClientSession(read, write)
        print("Initializing session...")
        await session.initialize()

        print("\nListing tools...")
        tools = await session.list_tools()
        print(f"Available tools: {tools}")


def main():
    try:
        asyncio.run(test_mcp())
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
