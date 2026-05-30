import sys
from pathlib import Path

# Add backend/ to sys.path so we can import the agent module.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mcp.server.fastmcp import FastMCP
from agents.transcription_agent import transcribe

mcp = FastMCP("transcription") # instantiate the MCP server 

# define the tool that the MCP server will expose.
@mcp.tool()
def transcribe_video(file_path: str) -> str:
    """
    Transcribe a video file at the given path. Returns the transcript as a string.
    """
    return transcribe(file_path)

if __name__ == "__main__":
    mcp.run()