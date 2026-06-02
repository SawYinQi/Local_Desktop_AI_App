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
    Analyzes ONLY the VISUAL content — objects, scenes, on-screen text, graphs. 
    CANNOT hear speech; use transcribe_video for spoken words. For full understanding, 
    use BOTH
    """
    return analyze(query, file_path)


if __name__ == "__main__":
    mcp.run()
