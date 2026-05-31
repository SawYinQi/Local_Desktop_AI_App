import copy
import json
import llm
import mcp_client

NEEDS_FILE_PATH_INJECTION = {"transcribe_video", "analyze_video"}

SYSTEM_PROMPT = (
    "You are a helpful assistant for a local video analysis app. You have tools to "
    "transcribe the audio and analyze the visuals of the user's uploaded videos.\n"

    "TOOL USE:\n"
    "- For ANY question about the video's content (describe it, what is "
    "happening, what objects/scenes/text/graphs appear, etc.), call analyze_video (visuals).\n"
    "- If the user explicitly wants just the spoken words (e.g. 'transcribe "
    "the video', 'what was said'), call transcribe_video.\n"
    "- For greetings, opinions, general questions, or talk about the conversation itself, "
    "answer DIRECTLY with no tool.\n"

    "When calling analyze_video, pass a 'query' that captures what the user wants to know.\n"
    "When transcribing, return the transcript essentially verbatim, fixing only obvious "
    "misspellings of real words/brand names/phrases; keep the wording and meaning intact.\n"
    "When summarizing, give a concise natural-language summary. Be concise and direct."
)

# Limit number of iteration of LLM tool call; prevent infinite loop
MAX_ITERATIONS = 3

_session_history: dict = {}

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

    prior = _session_history.get(session_id, [])

    # Initialize the message history with the system prompt and the user's query
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *prior, # add any prior messages in this session for context
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
            _session_history[session_id] = prior + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": result["content"]},
            ]
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
