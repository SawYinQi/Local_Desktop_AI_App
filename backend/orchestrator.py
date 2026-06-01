import copy
import json
import llm
import mcp_client

NEEDS_FILE_PATH_INJECTION = {"transcribe_video", "analyze_video"}
GENERATION_TOOLS = {"generate_pdf", "generate_pptx"}

TOOL_HEADERS = {
    "analyze_video": "VISUAL ANALYSIS FINDINGS",
    "transcribe_video": "AUDIO TRANSCRIPT",
    "generate_pdf": "MAKE PDF",
    "generate_pptx": "MAKE PPTX"
}

SYSTEM_PROMPT = (
    "You are an honest assistant for video content analysis and a general knowledge AI.\n"
    "You have access to the following tools: \n"
    "1. transcribe_video - used to transcribe the audio in the video for descriptive information\n"
    "2. analyze_video -   used for visual analysis of video content\n"
    "3. generate_pdf - used to generate a PDF report from structured content\n"
    "4. generate_pptx - used to generate a PowerPoint presentation from structured content\n"
    "You MUST NOT call a tool that is NOT in the above list; You MUST NEVER invent tool names/calls\n" 
    "You MUST ensure your response are concise\n"
    "You MUST NOT fabricate any information you did not get from tool results\n"
    "You MUST NOT make up any issue that you cannot prove with supporting evidence\n\n"

    "TOOL DECISION RULES (apply in order — rule 1 wins on ambiguity):\n"
    "\n"
    "1. DEFAULT — NO TOOL for:\n"
    "   - Greetings ('hi', 'hello'), thanks ('thank you', 'thanks', 'ok')\n"
    "   - Math, general knowledge questions\n"
    "   - Statements/questions about yourself, the user, or this app\n"
    "   - questions about the conversation ('what did we talk about?')\n"
    "   When in doubt → no tool.\n"
    "\n"
    "2. analyze_video ONLY — for questions about VISUAL content only:\n"
    "   - 'what is shown', 'describe the scene', 'what objects appear'\n"
    "   - colors, graphs, charts, on-screen text, faces, settings, environments\n"
    "\n"
    "3. transcribe_video ONLY — for questions about AUDIO content only OR DIRECT transcription VERBATIM:\n"
    "   - 'what was said', 'transcribe the audio', 'what does the speaker say'\n"
    "   - dialogue, narration, voice-over content\n"
    "\n"
    "4. BOTH analyze_video AND transcribe_video — for:\n"
    "   - Brand/product/name questions (info may be visual, audio, or both)\n"
    "   - 'summarize', 'describe', 'tell me about', 'what is the video about', ''\n"
    "   - Any 'comprehensive', 'in-depth', or 'overview' request\n"
    "   - Queries asking about multiple aspects (e.g. 'the cast and the dialogue')\n"
    "\n"
    "5. GENERATE PDF or PPTX — ONLY if the user EXPLICITLY says one of these keywords or is synonymous to one of theses keywords:\n"
    "   - For PDF: 'PDF', 'report', 'document', 'summary document' → generate_pdf\n"
    "   - For PPTX: 'PPTX', 'PPT', 'PowerPoint', 'slide deck', 'slides', 'presentation' → generate_pptx\n"
    "   When generating ABOUT THE VIDEO: first apply rule 4 (call BOTH analysis tools), THEN call the generation tool.\n"
    "   When generating ABOUT THE PRIOR CONVERSATION (e.g. 'summarize our discussion as a PDF'):\n"
    "   call generate_pdf or generate_pptx DIRECTLY — DO NOT call analyze_video or transcribe_video.\n"
    "\n"
    "6. ACKNOWLEDGMENT RULE — if the user just says 'thanks', 'ok', 'cool', 'got it':\n"
    "   Reply with a brief polite text ('You're welcome!') — NO tools, NO new report.\n"
    "   Even if a previous report was generated, do not re-run anything.\n"
    "\n"

)

# Limit number of iteration of LLM tool call; prevent infinite loop
MAX_ITERATIONS = 5

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

    last_artifact_path = "" # keep track of last generated artifact (e.g. PDF) to include in response events
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
            yield _event(response=result["content"], artifact_path=last_artifact_path)
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

        if tool_name in GENERATION_TOOLS:
            last_artifact_path = tool_result.strip()

        header = TOOL_HEADERS.get(tool_name, tool_name.upper())

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
            "content": f"[{header}]\n{tool_result}"
        })

    # notify client of iteration limit
    yield _event(response="(Reached iteration limit without final answer.)")
