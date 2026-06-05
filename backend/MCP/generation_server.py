import sys
from pathlib import Path

# Add backend/ to sys.path so we can import the agent module.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mcp.server.fastmcp import FastMCP
from MCP import HOST, PORTS
from agents.generation_agent import make_pdf, make_pptx

# instantiate the MCP server, bound to its local Streamable HTTP port
mcp = FastMCP("generation", host=HOST, port=PORTS["generation"])


@mcp.tool()
def generate_pdf(title: str, sections: list[dict]) -> str:
    """
    Creates a PDF report from structured content. Use this when the user
    asks for a 'report', 'PDF', 'document', or 'summary document'.

    Args:
        title: report title that appears at the top of the first page.
        sections: list of {"heading": str, "body": str}. Each becomes a section
                  with a bold heading and a body paragraph below it. Body should
                  be coherent prose (2-3 sentences), NOT raw tool output.

    Returns:
        Absolute path to the generated PDF file.
    """
    return make_pdf(title, sections)


@mcp.tool()
def generate_pptx(title: str, sections: list[dict]) -> str:
    """
    Creates a PowerPoint slide deck from structured content. Use this when
    the user asks for a 'slide deck', 'PPTX', 'presentation', or 'slides'.

    Args:
        title: title for the opening slide.
        sections: list of {"heading": str, "body": str}. Each becomes one slide
                  with the heading as the slide title. The body should contain
                  bullet points separated by newlines (\\n). Each \\n-separated
                  line becomes one bullet on the slide.

    Returns:
        Absolute path to the generated PPTX file.
    """
    return make_pptx(title, sections)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
