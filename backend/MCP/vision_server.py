import sys
from pathlib import Path

# Add backend/ to sys.path so we can import the agent module.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mcp.server.fastmcp import FastMCP
from MCP import HOST, PORTS
from agents.vision_agent import analyze

# instantiate the MCP server, bound to its local Streamable HTTP port
mcp = FastMCP("vision", host=HOST, port=PORTS["vision"])


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
    mcp.run(transport="streamable-http")
