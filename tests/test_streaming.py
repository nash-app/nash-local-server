import json
import os
from app.streaming import StreamProcessor
from litellm import ModelResponseStream


# Helper to load fixture files
def load_fixture(filename):
    """Load a fixture file from the fixtures directory."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    with open(os.path.join(fixtures_dir, filename)) as f:
        raw_chunks = json.load(f)
        # Convert each raw chunk dict to ModelResponseStream object
        return [ModelResponseStream(**chunk) for chunk in raw_chunks]


def test_process_chunk():
    """Test processing a chunk with text content."""
    # Create a processor
    processor = StreamProcessor()

    # Load the fixture with tool calls
    chunks = load_fixture("llm_chunks_tool_call.json")

    for chunk in chunks:
        processor.process_chunk(chunk)

    import pdb

    pdb.set_trace()
