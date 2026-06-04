import atexit
import subprocess
import sys
import time
import asyncio
from pathlib import Path
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from MCP import PORTS, server_url

# backend/ dir, used as cwd so the spawned servers can import MCP.
BACKEND_DIR = Path(__file__).resolve().parent

# Mapping of server name to the Python module that runs it. 
SERVER_MODULES = {name: f"MCP.{name}_server" for name in PORTS}

_procs = {}

def start_servers(timeout: float = 120.0) -> None:

    # if servers are already running, do nothing.
    if _procs:
        return

    # otherwise spawn a subprocess for each server module.
    for name, module in SERVER_MODULES.items():
        _procs[name] = subprocess.Popen(
            [sys.executable, "-m", module],
            cwd=BACKEND_DIR,
        )

    # cleanup on exit to avoid orphaned processes.
    atexit.register(stop_servers)

    # Wait until every server answers an MCP handshake before returning.
    asyncio.run(_wait_until_ready(timeout))

# async function to wait until all servers are ready.
async def _wait_until_ready(timeout: float) -> None:
    times_up = time.monotonic() + timeout
    for name in SERVER_MODULES:
        # loop until timeout
        while True:
            try:
                # try to connect and do a handshake. 
                async with streamablehttp_client(server_url(name)) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                break  # server up, break loop, move to the next
            except Exception:
                if time.monotonic() > times_up:
                    raise RuntimeError(f"MCP server '{name}' timeout")
                await asyncio.sleep(0.3)

# Stop all running servers. 
def stop_servers() -> None:
    # terminate any running servers.
    for proc in _procs.values():
        if proc.poll() is None: 
            proc.terminate()
    for proc in _procs.values():
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    _procs.clear()


# Async helper function to call a tool on an MCP server and return the result as a string
async def _call_tool_async(server_name: str, tool_name: str, arguments: dict) -> str:
    # connect to the MCP server over Streamable HTTP and open a client session,
    # then call the requested tool with the given arguments and return the result text
    async with streamablehttp_client(server_url(server_name)) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return "".join(c.text for c in result.content if hasattr(c, "text"))


# Adapter function to call the async tool caller from synchronous code
def call_tool(server_name: str, tool_name: str, arguments: dict) -> str:
    return asyncio.run(_call_tool_async(server_name, tool_name, arguments))


# Async helper function to list tools from an MCP server and return them as a list
async def _list_tools_async(server_name: str) -> list:
    async with streamablehttp_client(server_url(server_name)) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()
            return response.tools

# Discover all tools across every registered MCP server. Used by the orchestrator
# to give the LLM an up-to-date tool list without hand-maintaining schemas here.
def list_all_tools() -> dict:
    start_servers()  # start the servers 

    registry = {}
    for server_name in SERVER_MODULES:
        tools = asyncio.run(_list_tools_async(server_name))
        for tool in tools:
            registry[tool.name] = {
                "server": server_name,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    }
                }
            }

    return registry