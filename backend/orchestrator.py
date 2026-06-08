import copy
import json
import os
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
    "You are an honest and obedient assistant for video and general knowledge.\n"
    "You have access to the following tools to extract information from the video: \n"
    "1. transcribe_video\n"
    "2. analyze_video\n"
    "You MUST ALWAYS respond directly to the user UNLESSS the user ask for generation of a PDF or PPTX report\n"
    "You MUST NOT call a tool that is NOT in the above list; You MUST NEVER invent tool names/calls\n" 
    "You MUST ensure your response are concise\n"
    "You MUST NOT fabricate any information you did not get from tool results\n"
    "The selected video's file path is supplied to the tools AUTOMATICALLY by the system.\n\n"

    "CLARIFICATION RULE (overrides the tool RULES below):\n"
    "   If the request is genuinely ambiguous and you cannot tell what the user wants —\n"
    "   e.g. 'make me one', 'do that', 'analyze it' with no clear target, or a report request\n"
    "   that doesn't say what it should be about — DO NOT guess and DO NOT call any tool.\n"
    "   Reply with EXACTLY this on a single line:\n"
    "   CLARIFY: <one short question that would resolve the ambiguity>\n"
    "   After the user answers, continue normally.\n\n"

    "TOOL DECISION RULES (apply in order — RULE 1 wins on ambiguity); These are not tool calls so you MUST NOT try to call them as such:\n"
    "\n"
    "1. By DEFAULT — NO TOOL CALLS for:\n"
    "   - Greetings\n"
    "   - Math, general knowledge questions\n"
    "   - Statements/questions about yourself, the user, or this app\n"
    "   - questions about the conversation ('what did we talk about?')\n"
    "   - Summarizing / recapping THIS conversation or 'our discussion/chat' — answer from the"
    "   conversation history with NO tool, EVEN IF the user says 'summarize'."
    "   EXCEPTION: if the SAME message also asks for a PDF / PPTX / report, this NO-TOOL rule is"
    "   CANCELLED — follow RULE 5 and you MUST call the generation tool."
    "   When in doubt → (see CLARIFICATION RULE).\n"
    "\n"
    "2. analyze_video ONLY — for questions about VISUAL content only:\n"
    "   - 'what is shown', 'describe the scene', 'what objects appear'\n"
    "   - colors, graphs, charts, on-screen text, faces, settings, environments\n"
    "\n"
    "3. transcribe_video ONLY — for AUDIO content:\n"
    "   - 'transcribe' / 'transcript' ALWAYS mean VERBATIM — call transcribe_video DIRECTLY, no clarification.\n"
    "   - Return the transcript text EXACTLY as given by the tool; do NOT summarize,\n"
    "     shorten, paraphrase, or reformat it. Transcribing is NOT summarizing.\n"
    "   - Also: 'what was said', 'what does the speaker say', dialogue, narration, voice-over.\n"
    "\n"
    "4. When to use BOTH analyze_video AND transcribe_video — for questions about THE VIDEO's content:\n"
    "   - Brand/product/name questions (info may be visual, audio, or both)\n"
    "   - 'summarize the video', 'describe the video', 'what is the video about'\n"
    "   - Any 'comprehensive', 'in-depth', or 'overview' request\n"
    "   - Queries asking about multiple aspects (e.g. 'the cast and the dialogue')\n"
    "\n"
    "5. When to use generate_pdf or generate_pptx — ONLY when the user's CURRENT message explicitly contains one\n"
    "   of the format words below. If it does NOT, you MUST NOT call a generation tool and MUST NOT\n"
    "   mention making a file. 'transcribe', 'summarize', 'describe', 'what is shown' are NEVER, on\n"
    "   their own, generation requests — answer them in text.\n"
    "   - For PDF: 'PDF', 'report', 'document' → generate_pdf\n"
    "   - For PPTX: 'PPTX', 'PPT', 'PowerPoint', 'slide deck', 'slides', 'presentation' → generate_pptx\n"
    "   - This applies NO MATTER what the content is about — the video, OUR conversation, or general\n"
    "     knowledge. If a format word above appears, calling the generation tool is MANDATORY: first\n"
    "     gather the content (for a conversation summary, write it yourself from history; for video\n"
    "     questions, use the tool results), then pass it into generate_pdf / generate_pptx.\n"
    "   - You have NOT created a file unless you actually called the tool THIS turn. NEVER say a\n"
    "     report/PDF/PPTX was made, written, generated, or saved unless you called generate_pdf or\n"
    "     generate_pptx in this turn.\n"
    "\n"
    "6. ACKNOWLEDGMENT RULE — if the user just says something for pleasantries:\n"
    "   Reply directly with NO tools, NO new report.\n"
    "   Even if a previous report was generated, do not re-run anything.\n"
    "\n"
    "REPORT CONTENT RULES (when filling generate_pdf / generate_pptx):\n"
    "   - If the user lists specific sections/topics to include, you MUST create one slide/section\n"
    "     for EVERY topic they listed (e.g. people, what it's about, video type, product name →\n"
    "     a slide for each). Do NOT omit any, do NOT produce a partial deck, and do NOT offer to\n"
    "     add sections 'later' — generate the COMPLETE deck/report in a single call.\n"
    "   - Do NOT make one section per tool. NEVER use 'Visual Analysis' or 'Audio Transcript'\n"
    "     as section headings, and NEVER paste raw tool output.\n"
    "   - NEVER write placeholder tokens like '{{...}}', '[INSERT ...]', or variable names\n"
    "     (e.g. '{{analyze_video_results}}'). Write the ACTUAL content from the tool results.\n"
    "     If you don't have the content yet, call the gather tool FIRST and wait for its result.\n"
    "   - COMBINE the visual findings and the transcript, then reorganize into meaningful,\n"
    "     TOPIC-based sections that fit the request — e.g. 'People', 'Product', 'Key Claims', 'Summary'.\n"
    "   - Write each section in your own words as a synthesis of BOTH sources (2-3 sentences).\n"
    "   - If the visuals and the transcript disagree about a fact, TRUST THE TRANSCRIPT for\n"
    "     product names, brand names, and spoken claims (the visual model may guess wrong).\n"
    "   - Example: for a car advert, good sections are 'Vehicle', 'Key Features', 'Scene', 'Summary'\n"
    "   — NOT 'Visual Analysis' and 'Audio Transcript'.\n"
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
def _event(response: str = "", artifact_path: str = "") -> dict:
    return {
        "response": response,
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
    generated_formats: set = set()  # generation tools already run THIS query (prevents duplicate files)
    prior = _session_history.get(session_id, [])

    # Initialize the message history with the system prompt and the user's query
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *prior, # add any prior messages in this session for context
        {"role": "user", "content": query},
    ]

    # Get the tool schemas for the LLM based on the discovered tools from MCP servers
    tools = _tools_for_llm()

    # Hide vision and transcription tools if no video is loaded
    if not video_path:
        tools = [t for t in tools if t["function"]["name"] not in NEEDS_FILE_PATH_INJECTION]
        messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + (
                "\n\nNOTE: No video is currently loaded, so the video tools are unavailable. "
                "If the user asks about a video's content, briefly tell them to select a video "
                "first. Otherwise answer normally with no tool."
            ),
        }
    else:
        messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + (
                f"\n\nNOTE: A video IS loaded and ready: '{os.path.basename(video_path)}'. "
                "Its file path is supplied to the video tools AUTOMATICALLY. NEVER ask the user "
                "for a file path, filename, or to upload/provide a video — just call the tool."
            ),
        }

    # Iterate up to MAX_ITERATIONS times, calling the LLM and tools as needed
    # until we get a final text response or hit the max iteration limit
    for _ in range(MAX_ITERATIONS):
        
        # get result from querying the LLM with msg and tools schemas
        result = llm.chat(messages, tools=tools)

        # if the LLM returns a text response, yield it and end the conversation
        if result["type"] == "text":
            content = result["content"]

            # if the LLM indicates it needs clarification from the user
            stripped = content.lstrip()
            is_clarify = stripped.upper().startswith("CLARIFY:")
            reply = stripped[len("CLARIFY:"):].strip() if is_clarify else content

            _session_history[session_id] = prior + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": reply},
            ]

            # A clarification reply carries no artifact; a normal answer may include a generated file.
            yield _event(response=reply, artifact_path="" if is_clarify else last_artifact_path)
            return

        # gets the list of JSON payload from tool_calls
        calls = result["calls"]

        resolved = []  # (server, tool_name, raw_args, header, error)

        # Validate calls
        for call in calls:

            tool_name = call["name"]
            raw_args = call.get("arguments")
            args = dict(raw_args) if isinstance(raw_args, dict) else {}

            # check tool exist
            if tool_name not in _TOOLS:
                resolved.append((None, tool_name, args, "ERROR", f"Tool call failed: tool '{tool_name}' not found. Please try again."))
                continue

            # check tool need video file path and if video file path provided
            if tool_name in NEEDS_FILE_PATH_INJECTION:
                if not video_path:
                    resolved.append((None, tool_name, args, "ERROR", "No video uploaded yet. Please upload a video first."))
                    continue
                args["file_path"] = video_path

            resolved.append((_TOOLS[tool_name]["server"], tool_name, args, TOOL_HEADERS.get(tool_name, tool_name.upper()), None))

        has_gather = any(n in NEEDS_FILE_PATH_INJECTION for (_, n, _, _, e) in resolved if e is None)
        if has_gather:
            resolved = [(s, n, a, h, e) for (s, n, a, h, e) in resolved if n not in GENERATION_TOOLS]

        # append tool calls context to message
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                for (_, n, a, _, _) in resolved
            ]
        })

        # call all the listed tools in resolved that has no errors
        req = [(s, n, a) for (s, n, a, _, e) in resolved if e is None]
        print(f"Orchestrator: tool calls - {req}")
        outputs = iter(mcp_client.call_tools_parallel(req))

        # collect results 
        for (_, n, _, h, e) in resolved:
            # validation error
            if e is not None:
                content = f"[{h}]\n{e}"
            else:
                out = next(outputs)
                # executed calls raised errors
                if isinstance(out, Exception):
                    content = f"[{h}]\nERROR: {out}"
                elif n in GENERATION_TOOLS:
                    last_artifact_path = out.strip()
                    generated_formats.add(n)  # mark this format as produced for the query
                    content = (f"The requested file was created and saved to:\n{out.strip()}\n"
                               "Tell the user it's ready and give them this path. "
                               "Do NOT paste or restate the content.")
                # success,set results in contents
                else:
                    content = f"[{h}]\n{out}"

            # append append tool call result context
            messages.append({"role": "tool", "name": n, "content": content})

    
    yield _event(response="Something went wrong please try again.")
