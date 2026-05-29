import sys
import os

# backend/ to path so we can import transcription agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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