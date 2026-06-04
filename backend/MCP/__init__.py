# Shared config for the MCP tool servers.

HOST = "127.0.0.1"  

# dict of server name to port number
PORTS = {
    "transcription": 8101,
    "vision": 8102,
    "generation": 8103,
}

# List of all server names, used for orchestrator startup 
def server_url(name: str) -> str:
    return f"http://{HOST}:{PORTS[name]}/mcp"
