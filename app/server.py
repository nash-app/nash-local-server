import asyncio
import datetime
import json
from typing import List

from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .llm_handler import (
    clean_up_tool_results_inline,
    stream_llm_response,
    get_conversation_token_info,
    get_response_metadata_from_headers,
)
from .mcp_handler import MCPHandler
from .prompts import get_system_prompt
from .stream_processor import StreamProcessor


app = FastAPI(title="Nash LLM Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],  # Or specify: ["GET", "POST"]
    allow_headers=["*"],  # Or specify required headers
)


class BaseRequest(BaseModel):
    """Base request model with common fields."""

    messages: List[dict] = Field(..., description="List of messages in the conversation")
    api_key: str = Field(..., description="API key to use for the request")
    api_base_url: str = Field(..., description="API base URL to use for the request")


class StreamRequest(BaseRequest):
    """Request model for streaming completions."""

    model: str = Field(..., description="Model to use for completion")


class TokenInfoRequest(BaseModel):
    """Request model for token information."""

    messages: List[dict] = Field(..., description="List of messages in the conversation")
    model: str = Field(..., description="Model to use for token counting")


@app.on_event("startup")
async def startup_event():
    """Configure services on server startup."""
    # Initialize MCP singleton
    mcp = MCPHandler.get_instance()
    await mcp.initialize()

    # Generate system prompt and store in app state
    app.state.system_prompt = get_system_prompt()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    mcp = MCPHandler.get_instance()
    await mcp.close()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


async def process_llm_stream(
    messages: list,
    model: str,
    api_key: str,
    api_base_url: str,
):
    """
    Format LLM responses into proper SSE format, handling tool calls on the server.

    This function:
    1. Streams the LLM response
    2. Detects tool calls
    3. Executes tools
    4. Sends tool results back to the LLM
    5. Continues the conversation
    6. Streams everything to the client
    """

    # Get MCP instance once outside the loop
    mcp = MCPHandler.get_instance()

    # Get available tools asynchronously
    tools = await mcp.list_tools_litellm()

    initial_message_count = len(messages)

    # Continue the conversation until no more tool calls
    while True:
        # Create a new processor but reuse the same MCP instance
        processor = StreamProcessor(mcp)

        try:
            # Truncate tool results that are long and have been responded to already
            clean_up_tool_results_inline(messages)

            # Stream the LLM response
            response = await stream_llm_response(
                messages=messages, model=model, api_key=api_key, api_base_url=api_base_url, tools=tools
            )
            async for chunk in response:
                # Process each chunk with the stream processor
                streamable_content = processor.process_chunk(chunk)

                # Augment the streamable content with http client specific attributes
                streamable_content["type"] = "stream"
                streamable_content["tool_result"] = None
                streamable_content["new_raw_llm_messages"] = []
                streamable_content["finish_reason"] = None
                streamable_content["sleep_seconds"] = 0

                yield f"data: {json.dumps(streamable_content)}\n\n"

                # If we have a finish reason, send it as a separate message
                if processor.finish_reason:
                    finish_message = {
                        "type": "finish_reason",
                        "content": None,
                        "tool_name": None,
                        "tool_args": None,
                        "tool_result": None,
                        "new_raw_llm_messages": None,
                        "finish_reason": processor.finish_reason,
                        "sleep_seconds": None,
                    }
                    yield f"data: {json.dumps(finish_message)}\n\n"

            # Get relevant metadata like limits and cost from the response headers
            response_metadata = get_response_metadata_from_headers(response._response_headers)

            assistant_message = processor.get_assistant_message()
            messages.append(assistant_message)
            if processor.finish_reason == "tool_calls":
                messages_for_tool_call_results = await processor.execute_tool_calls_and_get_user_message()
                messages.append(messages_for_tool_call_results[0])  # TODO: Handle multiple tool call results

                message_for_client = {
                    "type": "tool_result",
                    "content": None,
                    "tool_name": None,
                    "tool_args": None,
                    "tool_result": messages_for_tool_call_results[0]["content"],
                    "new_raw_llm_messages": [],
                    "finish_reason": None,
                    "sleep_seconds": None,
                }
                yield f"data: {json.dumps(message_for_client)}\n\n"
                if response_metadata["sleep_seconds"] > 0:
                    sleep_message_for_client = {
                        "type": "sleep_seconds",
                        "content": None,
                        "tool_name": None,
                        "tool_args": None,
                        "tool_result": None,
                        "new_raw_llm_messages": None,
                        "finish_reason": None,
                        "sleep_seconds": response_metadata["sleep_seconds"],
                    }
                    yield f"data: {json.dumps(sleep_message_for_client)}\n\n"
                    await asyncio.sleep(response_metadata["sleep_seconds"])
                if response_metadata["sleep_seconds"] > 0:
                    sleep_message_for_client = {
                        "type": "sleep_seconds",
                        "content": None,
                        "tool_name": None,
                        "tool_args": None,
                        "tool_result": None,
                        "new_raw_llm_messages": None,
                        "finish_reason": None,
                        "sleep_seconds": response_metadata["sleep_seconds"],
                    }
                    yield f"data: {json.dumps(sleep_message_for_client)}\n\n"
                    await asyncio.sleep(response_metadata["sleep_seconds"])
            else:
                clean_up_tool_results_inline(messages)
                message_for_client = {
                    "type": "new_raw_llm_messages",
                    "content": None,
                    "tool_name": None,
                    "tool_args": None,
                    "tool_result": None,
                    "new_raw_llm_messages": messages[initial_message_count:],
                    "finish_reason": None,
                    "sleep_seconds": None,
                }
                yield f"data: {json.dumps(message_for_client)}\n\n"
                if response_metadata["sleep_seconds"] > 0:
                    await asyncio.sleep(response_metadata["sleep_seconds"])
                break

        except Exception as e:
            # Handle errors in streaming
            error_message = f"{type(e).__name__}: {str(e)}"
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            print(f"Error in stream_llm_response: {error_message}")
            break

    # End of stream marker
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions/stream")
async def stream_completion(request: StreamRequest):
    """Stream chat completions with user-provided credentials."""

    async def error_stream(error_msg: str):
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
        yield "data: [DONE]\n\n"

    try:
        messages = [{"role": "system", "content": app.state.system_prompt}]
        messages.extend(request.messages)

        # Format the response
        return StreamingResponse(
            process_llm_stream(
                messages=messages,
                model=request.model,
                api_key=request.api_key,
                api_base_url=request.api_base_url,
            ),
            media_type="text/event-stream",
            headers={"Access-Control-Allow-Origin": "*"},
        )
    except Exception as e:
        return StreamingResponse(error_stream(str(e)), media_type="text/event-stream", status_code=500)


@app.post("/v1/mcp/list_tools")
async def list_tools():
    """List all available MCP tools."""
    try:
        mcp = MCPHandler.get_instance()
        tools = await mcp.list_tools()
        return {"tools": tools}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/mcp/list_prompts")
async def list_prompts():
    """List all available prompts."""
    try:
        mcp = MCPHandler.get_instance()
        prompts = await mcp.list_prompts()
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/mcp/get_prompt")
async def get_prompt(request: Request):
    """Get a specific prompt."""
    try:
        data = await request.json()
        prompt_name = data.get("prompt_name")
        arguments = data.get("arguments", {})

        if not prompt_name:
            raise HTTPException(status_code=400, detail="prompt_name is required")

        mcp = MCPHandler.get_instance()
        prompt = await mcp.get_prompt(prompt_name, arguments=arguments)
        return {"prompt": prompt}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/mcp/list_resources")
async def list_resources():
    """List all available resources."""
    try:
        mcp = MCPHandler.get_instance()
        resources = await mcp.list_resources()
        return {"resources": resources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/mcp/read_resource")
async def read_resource(request: Request):
    """Read a specific resource."""
    try:
        data = await request.json()
        resource_path = data.get("resource_path")

        if not resource_path:
            raise HTTPException(status_code=400, detail="resource_path is required")

        mcp = MCPHandler.get_instance()
        content = await mcp.read_resource(resource_path)
        return {"content": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/mcp/call_tool")
async def call_tool(request: Request):
    """Call a specific MCP tool."""
    try:
        data = await request.json()
        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})

        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name is required")

        mcp = MCPHandler.get_instance()
        result = await mcp.call_tool(tool_name, arguments=arguments)
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/token_info")
def get_token_info(request: TokenInfoRequest):
    """Get token usage information for a conversation."""
    try:
        token_info = get_conversation_token_info(messages=request.messages, model=request.model)
        return token_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6274)


if __name__ == "__main__":
    main()
