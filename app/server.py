from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

from .llm_handler import stream_llm_response
from .mcp_handler import MCPHandler
from .prompts import get_system_prompt


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
    role: str = Field(
        ...,
        description="The role of the message sender (user/assistant/system)"
    )
    content: str = Field(..., description="The content of the message")


class BaseRequest(BaseModel):
    """Base request model with common fields."""
    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation"
    )
    api_key: str = Field(
        ...,
        description="API key to use for the request"
    )
    api_base_url: str = Field(
        ...,
        description="API base URL to use for the request"
    )


class StreamRequest(BaseRequest):
    """Request model for streaming completions."""
    model: str = Field(
        ...,
        description="Model to use for completion"
    )


@app.on_event("startup")
async def startup_event():
    """Configure services on server startup."""
    # Initialize MCP singleton
    mcp = MCPHandler.get_instance()
    await mcp.initialize()

    # Get available tools
    tools = await mcp.list_tools()

    # Generate system prompt with tool definitions and store in app state
    app.state.system_prompt = get_system_prompt(tools)


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
    """Format LLM responses into proper SSE format."""
    # Stream content chunks
    try:
        async for chunk in stream_llm_response(
            messages=messages,
            model=model,
            api_key=api_key,
            api_base_url=api_base_url
        ):
            # Process the raw LiteLLM chunk
            if hasattr(chunk, 'choices') and chunk.choices:
                # Extract the content from the choices
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    # Format as SSE event
                    yield f"data: {json.dumps({'content': content})}\n\n"
    except Exception as e:
        # Handle errors in streaming
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
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
                api_base_url=request.api_base_url
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        return StreamingResponse(
            error_stream(str(e)),
            media_type="text/event-stream",
            status_code=500
        )


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
            raise HTTPException(
                status_code=400, 
                detail="prompt_name is required"
            )
            
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
            raise HTTPException(
                status_code=400, 
                detail="resource_path is required"
            )
            
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
            raise HTTPException(
                status_code=400, 
                detail="tool_name is required"
            )
            
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
