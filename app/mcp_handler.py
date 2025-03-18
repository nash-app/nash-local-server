import asyncio
import os
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPHandler:
    _instance: Optional["MCPHandler"] = None
    _initialized = False
    _session: Optional[ClientSession] = None
    _client_ctx = None
    _session_ctx = None
    _read = None
    _write = None
    _initialization_lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "MCPHandler":
        if cls._instance is None:
            cls._instance = MCPHandler()
        return cls._instance

    async def initialize(self):
        # Use a lock to prevent multiple simultaneous initializations
        async with self._initialization_lock:
            if self._initialized:
                return

            # Setup server parameters
            nash_path = os.getenv("NASH_PATH")
            if not nash_path:
                raise ValueError("NASH_PATH environment variable not set")

            mcp_cmd = os.path.join(nash_path, ".venv/bin/mcp")
            server_script = os.path.join(nash_path, "src/nash_mcp/server.py")
            server_params = StdioServerParameters(command=mcp_cmd, args=["run", server_script], env=None)

            try:
                # Create and enter client context
                self._client_ctx = stdio_client(server_params)
                self._read, self._write = await self._client_ctx.__aenter__()

                # Create and enter session context
                self._session = ClientSession(self._read, self._write)
                await self._session.__aenter__()
                await self._session.initialize()

                self._initialized = True
            except Exception as e:
                # Clean up if initialization fails
                await self.close()
                msg = f"Failed to initialize MCP: {str(e)}"
                raise RuntimeError(msg) from e

    async def close(self):
        """Gracefully close the MCP session and client"""
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None

        if self._client_ctx:
            try:
                await self._client_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._client_ctx = None

        self._read = None
        self._write = None
        self._initialized = False

    async def ensure_initialized(self):
        """Ensure the handler is initialized before use"""
        if not self._initialized:
            await self.initialize()

    async def list_tools(self):
        """List available MCP tools"""
        await self.ensure_initialized()
        return await self._session.list_tools()

    async def list_tools_litellm(self):
        """List available MCP tools in litellm tools format"""
        await self.ensure_initialized()
        tools = []
        for tool in (await self.list_tools()).tools:
            # Create a litellm-compatible tool definition
            tool_def = {
                "type": "function",
                "function": {"name": tool.name, "description": tool.description.strip(), "parameters": {}},
            }

            tool_def["function"]["parameters"] = tool.inputSchema
            tools.append(tool_def)

        return tools

    async def call_tool(self, tool_name: str, **kwargs):
        """Call an MCP tool with the given arguments"""
        await self.ensure_initialized()
        return await self._session.call_tool(tool_name, **kwargs)

    @property
    def is_initialized(self) -> bool:
        """Check if the handler is initialized"""
        return self._initialized
