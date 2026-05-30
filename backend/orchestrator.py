import copy
import json
import llm
import mcp_client

NEEDS_FILE_PATH_INJECTION = {"transcribe_video"}

SYSTEM_PROMPT = (
    "You are a helpful assistant for a local video analysis app. You have tools to "
    "transcribe, analyze, and generate reports from the user's uploaded videos.\n"
    "Call a tool ONLY when the user clearly asks for that action (e.g. 'transcribe the "
    "video', 'what objects are shown', 'make a PDF'). For greetings, opinions, general "
    "questions, or anything about the conversation itself, answer DIRECTLY without a tool.\n"
    "When you transcribe, return the tool's text essentially verbatim  with fixes to obvious"
    "misspellings of real words/brand names/phrase, but keep the wording and meaning intact.\n"
    "When asked to summarize, give a concise natural-language summary. Be concise and direct."
)

# Limit number of iteration of LLM tool call; prevent infinite loop
MAX_ITERATIONS = 3


# Discover available tools from MCP servers
_TOOLS = mcp_client.list_all_tools() 
# Log the discovered tools 
print(f"Orchestrator: found {len(_TOOLS)} tool(s): {list(_TOOLS)}")

# Create an event dict to stream back to client 
def _event(response: str = "", needs_clarification: bool = False,
           clarification_prompt: str = "", artifact_path: str = "") -> dict:
    return {
        "response": response,
        "needs_clarification": needs_clarification,
        "clarification_prompt": clarification_prompt,
        "artifact_path": artifact_path,
    }

# Create schema list for LLM based on discovered tools
def _tools_for_llm() -> list:
    schemas = []
    for tool_name, info in _TOOLS.items():
        schema = info["schema"]
        if tool_name in NEEDS_FILE_PATH_INJECTION:
            schema = copy.deepcopy(schema)
            params = schema["function"].setdefault("parameters", {})
            params.get("properties", {}).pop("file_path", None) # remove file_path, will be provided by the orchestrator 

            # if there's a required list, make a new list without file_path in it
            if "required" in params:
                params["required"] = [p for p in params["required"] if p != "file_path"]
        schemas.append(schema)

    return schemas

# Handles user query by querying the LLM and calling tools as needed, 
# yielding events for the gRPC server to stream back to the client
def handle_query(session_id: str, query: str, video_path: str | None):
    print(f"Orchestrator: session={session_id} query={query!r} video_path={video_path}")

    # Initialize the message history with the system prompt and the user's query
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    # Get the tool schemas for the LLM based on the discovered tools from MCP servers
    tools = _tools_for_llm()

    # Iterate up to MAX_ITERATIONS times, calling the LLM and tools as needed 
    # until we get a final text response or hit the max iteration limit
    for _ in range(MAX_ITERATIONS):
        # get result from querying the LLM with msg and tools schemas
        result = llm.chat(messages, tools=tools)

        # if the LLM returns a text response, yield it and end the conversation
        if result["type"] == "text":
            yield _event(response=result["content"])
            return

        # otherwise the LLM still needs tools
        tool_name = result["name"] 
        args = dict(result.get("arguments") or {}) 
        
        # check if tools called by LLM is in scope
        if tool_name not in _TOOLS:
            yield _event(response=f"(LLM tried to call unknown tool '{tool_name}'.)")
            return

        # if the tools needs file path, add it to argument dict
        if tool_name in NEEDS_FILE_PATH_INJECTION:
            if not video_path:
                yield _event(response="No video uploaded yet. Please upload a video first.")
                return
            args["file_path"] = video_path
        

        server_name = _TOOLS[tool_name]["server"]
        tool_result = mcp_client.call_tool(server_name, tool_name, args)

        # append LLM tool calls and tool results to message as context for next LLM query
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args)},
            }],
        })
        messages.append({
            "role": "tool",
            "name": tool_name,
            "content": tool_result,
        })

    # notify client of iteration limit
    yield _event(response="(Reached iteration limit without final answer.)")
