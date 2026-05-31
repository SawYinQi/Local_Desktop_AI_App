import sys
from pathlib import Path

# Add backend/ to sys.path so we can import the agent module.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mcp.server.fastmcp import FastMCP
from agents.vision_agent import analyze

mcp = FastMCP("vision")  # instantiate the MCP server


# Tool exposed to the orchestrator via MCP.
@mcp.tool()
def analyze_video(file_path: str, query: str) -> str:
    """
    Analyze the visuals of a video file at the given path in the context of a user query.
    returns the analysis as a string.
    """
    return analyze(query, file_path)


if __name__ == "__main__":
    mcp.run()
