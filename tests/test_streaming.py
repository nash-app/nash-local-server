import json
import os
from app.stream_processor import StreamProcessor
from litellm import ModelResponseStream


# Helper to load fixture files
def load_fixture(filename):
    """Load a fixture file from the fixtures directory."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    with open(os.path.join(fixtures_dir, filename)) as f:
        raw_chunks = json.load(f)
        # Convert each raw chunk dict to ModelResponseStream object
        return [ModelResponseStream(**chunk) for chunk in raw_chunks]


def test_process_chunk_single_tool_call():
    """Test processing a chunk with text content."""
    # Create a processor
    processor = StreamProcessor()

    # Load the fixture with tool calls
    chunks = load_fixture("anthropic_llm_chunks_tool_call_01.json")

    for chunk in chunks:
        processor.process_chunk(chunk)

    processor = StreamProcessor()

    # Load the fixture with tool calls
    chunks = load_fixture("anthropic_llm_chunks_tool_call_02.json")

    for chunk in chunks:
        processor.process_chunk(chunk)


def test_process_chunk_only_assistant():
    processor = StreamProcessor()

    # Load the fixture with tool calls
    chunks = load_fixture("anthropic_llm_chunks_only_assistant_01.json")

    for chunk in chunks:
        processor.process_chunk(chunk)
