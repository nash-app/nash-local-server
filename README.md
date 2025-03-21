# nash-local-server

A local server for LLM interactions and MCP usage with proper session handling and streaming responses.

## Quick Start

```bash
poetry install
poetry run llm_server  # Terminal 1
poetry run client_example  # Terminal 2
```

## Testing with Different Providers

You can test the LLM functionality directly using our interactive test script:

```bash
poetry run python test_scripts/test_llm_handler.py
```

The test script provides an interactive CLI that allows you to:

- Choose between OpenAI and Anthropic providers
- Select from available models:
  - OpenAI: gpt-4-turbo, gpt-4-0125-preview, gpt-4, gpt-3.5-turbo
  - Anthropic: claude-3-opus/sonnet/haiku, claude-2.1
- Test conversation summarization
- View session IDs and token usage

### Environment Setup for Testing

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=sk-...      # Required for OpenAI models
ANTHROPIC_API_KEY=sk-...   # Required for Anthropic models
```

## Architecture

- **Stateless Server**: No conversation history stored server-side
- **Client-Side State**: Clients maintain message history and session IDs
- **Streaming**: Server-sent events with proper session handling

## Session ID Flow

1. **First Message**

   - Client sends request without session ID
   - Server generates new ID and sends it in first chunk
   - Client stores ID for future requests

2. **Subsequent Messages**
   - Client includes stored session ID
   - Server validates and maintains session continuity
   - Same ID returned in response

## Response Format

```
data: {"session_id": "uuid-here"}  # First chunk
data: {"content": "response text"}  # Content chunks
data: {"session_id": "uuid-here"}  # Last chunk
data: [DONE]
```

## API Endpoints

### 1. Stream Chat Completions

`POST /v1/chat/completions/stream`

Stream chat completions from the LLM with server-sent events.

#### Request Body

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "model": "gpt-4-turbo-preview", // Required
  "api_key": "sk-...", // Required
  "api_base_url": "https://api.openai.com/v1", // Required
  "session_id": "optional-uuid" // Optional
}
```

#### Response Format

```
data: {"session_id": "uuid-here"}  # First chunk
data: {"content": "response text"}  # Content chunks
data: {"session_id": "uuid-here"}  # Last chunk
data: [DONE]
```

### 2. Summarize Conversation

`POST /v1/chat/summarize`

Summarize a conversation to reduce token count while preserving context.

#### Request Body

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you!"
    }
  ],
  "model": "gpt-4-turbo-preview", // Required
  "api_key": "sk-...", // Required
  "api_base_url": "https://api.openai.com/v1", // Required
  "session_id": "optional-uuid" // Optional
}
```

#### Response Format

```json
{
  "success": true,
  "summary": "Conversation summary text",
  "messages": [
    {
      "role": "assistant",
      "content": "Previous conversation summary:\n[summary text]"
    }
  ],
  "token_reduction": {
    "before": 100,
    "after": 50
  },
  "session_id": "uuid-here"
}
```

### 3. MCP Methods

`POST /v1/mcp/{method}`

Generic endpoint for calling any MCP client method. The method name is specified in the URL path, and any arguments are passed in the request body.

```json
{
  // Method arguments as key-value pairs
  "arg1": "value1",
  "arg2": "value2"
}
```

Examples:

```bash
# List available tools
POST /v1/mcp/list_tools
{}

# Get a tool's schema
POST /v1/mcp/get_tool_schema
{
  "tool_name": "my_tool"
}

# Execute a tool
POST /v1/mcp/execute_tool
{
  "tool_name": "my_tool",
  "args": {
    "param1": "value1"
  }
}
```

The response format is consistent:

```json
{
  "result": <method result>
}
```

Error responses:

```json
{
  "error": "Error message"
}
```

or for 400 errors:

```json
{
  "detail": "Error details"
}
```

## Provider Support

The server supports multiple LLM providers through their respective base URLs:

- OpenAI: `https://api.openai.com/v1`
- Anthropic: `https://api.anthropic.com`

## Client Implementation Tips

1. **Session Management**

   - Store session ID from first response chunk
   - Verify against final chunk's ID
   - Pass ID in all subsequent requests

2. **Error Recovery**

   - Keep last known session ID
   - Can retry with same ID if connection drops
   - Server preserves ID even during errors

3. **API Configuration**
   - Always provide model, api_key, and api_base_url
   - Use appropriate base URL for your provider

See `client_example.py` for a complete implementation.

## Tests

Run tests with `pytest`

```bash
$ pytest tests
```
