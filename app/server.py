from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

from .llm_handler import stream_llm_response
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


class Message(BaseModel):
    """A message in the conversation."""

    role: str = Field(..., description="The role of the message sender (user/assistant/system)")
    content: str = Field(..., description="The content of the message")


class BaseRequest(BaseModel):
    """Base request model with common fields."""

    messages: List[Message] = Field(..., description="List of messages in the conversation")
    api_key: str = Field(..., description="API key to use for the request")
    api_base_url: str = Field(..., description="API base URL to use for the request")


class StreamRequest(BaseRequest):
    """Request model for streaming completions."""

    model: str = Field(..., description="Model to use for completion")


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
    # Initialize the stream processor
    processor = StreamProcessor()
    mcp = MCPHandler.get_instance()

    # Get available tools asynchronously
    tools = await mcp.list_tools_litellm()

    # Keep track of the conversation
    conversation_messages = messages.copy()
    new_messages_from_stream = []

    # Continue the conversation until no more tool calls
    while True:
        # Reset the processor for this iteration
        processor = StreamProcessor()

        try:
            # Stream the LLM response
            async for chunk in stream_llm_response(
                messages=conversation_messages, model=model, api_key=api_key, api_base_url=api_base_url, tools=tools
            ):
                # Process each chunk with the stream processor
                streamable_content = processor.process_chunk(chunk)

                # Augment the streamable content with http client specific attributes
                streamable_content["type"] = "stream"
                streamable_content["tool_result"] = None
                streamable_content["new_raw_llm_messages"] = []

                yield f"data: {json.dumps(streamable_content)}\n\n"

            assistant_message = processor.get_assistant_message()
            conversation_messages.append(assistant_message)
            new_messages_from_stream.append(assistant_message)
            if processor.tool_calls:
                messages_for_tool_call_results = await processor.execute_tool_calls_and_get_user_message()
                conversation_messages.append(messages_for_tool_call_results[0])  # TODO: Handle multiple tool call results
                new_messages_from_stream.append(messages_for_tool_call_results[0])
                message_for_client = {
                    "type": "tool_result",
                    "content": None,
                    "tool_name": None,
                    "tool_args": None,
                    "tool_result": messages_for_tool_call_results[0]['content'],
                    "new_raw_llm_messages": []
                }
                yield f"data: {json.dumps(message_for_client)}\n\n"
            else:
                message_for_client = {
                    "type": "tool_result",
                    "content": None,
                    "tool_name": None,
                    "tool_args": None,
                    "tool_result": None,
                    "new_raw_llm_messages": new_messages_from_stream
                }
                yield f"data: {json.dumps(message_for_client)}\n\n"
                break

        except Exception as e:
            # Handle errors in streaming
            error_message = str(e)
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            print(f"Error in stream_llm_response: {error_message}")
            break

    # End of stream marker
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions/stream")
async def stream_completion(request: StreamRequest):
    """Stream chat completions with user-provided credentials."""
    try:
        messages = [{"role": "system", "content": app.state.system_prompt}]
        messages.extend([msg.dict() for msg in request.messages])

        async def error_stream(error_msg: str):
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"

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


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6274)


if __name__ == "__main__":
    main()
