import os
import sys
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# backend/ dir, used for cwd so subprocess can finds MCP servers
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the MCP servers and their connection parameters
SERVERS = {
    "transcription": StdioServerParameters(
        command= sys.executable,
        args=["-m", "MCP.transcription_server"],
        cwd=BACKEND_DIR
    )
}

# Async helper function to call a tool on an MCP server and return the result as a string
async def _call_tool_async(server_name: str, tool_name: str, arguments: dict) -> str:

    params = SERVERS[server_name] # get the server parameters for the requested server

    # connect to the MCP server using stdio and create a client session
    # ,then call the requested tool with the given arguments and return the result text
    async with stdio_client(params) as (read, write): 
        async with ClientSession(read, write) as session: 
            await session.initialize() 
            result = await session.call_tool(tool_name, arguments)
            return "".join(c.text for c in result.content if hasattr(c, "text"))

# Adapter function to call the async tool caller from synchronous code
def call_tool(server_name: str, tool_name: str, arguments: dict) -> str:
    return asyncio.run(_call_tool_async(server_name, tool_name, arguments))