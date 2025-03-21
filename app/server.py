from typing import List, Optional
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
    session_id: Optional[str] = Field(None, description="Session ID to use for the request")


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
    session_id: Optional[str] = None,
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

    # Start by sending session ID if provided
    if session_id:
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"

    # Get available tools asynchronously
    tools = await mcp.list_tools_litellm()

    # Keep track of the conversation
    conversation_messages = messages.copy()

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
                display_text, tool_call_data = processor.process_chunk(chunk)

                # If there's text to display, send it as an SSE event
                if display_text:
                    yield f"data: {json.dumps({'content': display_text})}\n\n"

                # If there's tool call data, send it to the client
                if tool_call_data:
                    # Convert tool call data to a serializable form
                    serializable_tool_calls = []
                    for tool_call in tool_call_data:
                        tool_call_dict = {}
                        if hasattr(tool_call, "id"):
                            tool_call_dict["id"] = tool_call.id
                        if hasattr(tool_call, "function"):
                            tool_call_dict["function"] = {}
                            if hasattr(tool_call.function, "name"):
                                tool_call_dict["function"]["name"] = tool_call.function.name
                            if hasattr(tool_call.function, "arguments"):
                                tool_call_dict["function"]["arguments"] = tool_call.function.arguments
                        serializable_tool_calls.append(tool_call_dict)

                    yield f"data: {json.dumps({'tool_calls': serializable_tool_calls})}\n\n"

                # If we're processing a tool call, send a notification
                if processor.collecting_tool_call:
                    # Send a special event to indicate a tool call is in progress
                    tool_status = {"tool_call_in_progress": True}
                    if processor.tool_name:
                        tool_status["tool_name"] = processor.tool_name
                    if processor.tool_call_id:
                        tool_status["tool_id"] = processor.tool_call_id

                    yield f"data: {json.dumps(tool_status)}\n\n"

            # Get the assistant's message to add to the conversation
            assistant_message = processor.get_message_for_history()
            if assistant_message:
                conversation_messages.append(assistant_message)

            # If a tool call was detected, execute it and continue the conversation
            if processor.is_tool_call_detected():
                # Send notification that we're executing a tool
                yield f"data: {json.dumps({'executing_tool': processor.tool_use_info['name']})}\n\n"

                # Execute the tool and get the result
                tool_result = await processor.execute_tool(mcp)

                # Send the tool result to the client
                tool_result_data = {
                    "tool_result": {
                        "name": processor.tool_use_info["name"],
                        "success": tool_result["success"],
                        "result": tool_result["result_text"],
                    }
                }
                yield f"data: {json.dumps(tool_result_data)}\n\n"

                # Add the tool result to the conversation
                if tool_result["success"] and tool_result["result_message"]:
                    conversation_messages.append(tool_result["result_message"])

                    # Yield a separator to indicate we're continuing with the LLM
                    yield f"data: {json.dumps({'status': 'continuing_with_tool_result'})}\n\n"

                    # Continue the loop - we'll get another response from the LLM
                    continue

            # If we got here, there were no tool calls or we've completed all tool calls
            break

        except Exception as e:
            # Handle errors in streaming
            error_message = str(e)
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            print(f"Error in stream_llm_response: {error_message}")
            break

    # Send session ID again at the end if provided
    if session_id:
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"

    # End of stream marker
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions/stream")
async def stream_completion(request: StreamRequest):
    """Stream chat completions with user-provided credentials."""
    try:
        messages = [{"role": "system", "content": app.state.system_prompt}]
        messages.extend([msg.dict() for msg in request.messages])

        async def error_stream(error_msg: str):
            if request.session_id:
                yield f"data: {json.dumps({'session_id': request.session_id})}\n\n"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            if request.session_id:
                yield f"data: {json.dumps({'session_id': request.session_id})}\n\n"
            yield "data: [DONE]\n\n"

        # Format the response
        return StreamingResponse(
            process_llm_stream(
                messages=messages,
                model=request.model,
                api_key=request.api_key,
                api_base_url=request.api_base_url,
                session_id=request.session_id,
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
