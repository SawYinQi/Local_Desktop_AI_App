import json
import platform
from pathlib import Path
import re
from json_repair import repair_json

from utils.runtime import pick_backend, has_ov_model, has_mlx_model, has_hf_model

# Model paths per backend: OpenVINO (Intel), MLX (Apple Silicon), Hugging Face (other).
OV_MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-7b-int4"
MLX_MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-3b-instruct-mlx-8bit"
HF_MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-3b-instruct"


BACKEND = pick_backend() 

print(f"LLM: host = {platform.system()}/{platform.machine()}, backend = {BACKEND}")

# OpenVINO
if BACKEND == "openvino":
    MODEL_PATH = OV_MODEL_PATH
    # check model availability and raise error if not found, before loading.
    if not has_ov_model(MODEL_PATH):
        raise RuntimeError(f"No OpenVINO model available at {MODEL_PATH}")
    
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer
    
    print(f"LLM: loading {MODEL_PATH} ...")
    _model = OVModelForCausalLM.from_pretrained(str(MODEL_PATH), device="GPU") # load the OpenVINO-optimized model
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH)) # load the tokeniser


# MLX Apple's native Metal framework
elif BACKEND == "mlx":
    MODEL_PATH = MLX_MODEL_PATH
    # check model availability and raise error if not found, before loading.
    if not has_mlx_model(MODEL_PATH):
        raise RuntimeError(f"No MLX model available at {MODEL_PATH}")

    from mlx_lm import load
    from mlx_lm import generate as _mlx_generate

    print(f"LLM: loading {MODEL_PATH} ...")
    _model, _tokenizer = load(str(MODEL_PATH))

# Hugging Face Transformers
else:  
    MODEL_PATH = HF_MODEL_PATH
    # check model availability and raise error if not found, before loading.
    if not has_hf_model(MODEL_PATH):
        raise RuntimeError(f"No Hugging Face model available at {MODEL_PATH}")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"LLM: loading {MODEL_PATH}")

    # Loads model
    _model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float16
    )
    if torch.backends.mps.is_available():
        _model = _model.to("mps")
    elif torch.cuda.is_available():
        _model = _model.to("cuda")

    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH)) # load the tokeniser

print("LLM: model loaded.")

def chat(messages: list, tools: list | None = None, max_new_tokens: int = 3072) -> dict:
    """
    messages: list of {"role": "system" | "user", "content": str}
    tools: list of {"name": str, "description": str, "parameters": dict} 
    describing the available tools for the LLM to call. The schema is in OpenAI function-c
    """

    # tokenizer format the message and tools to string for the LLM  
    prompt = _tokenizer.apply_chat_template(
        # schema 
        messages, 
        tools=tools, 
        add_generation_prompt=True, # add special prompt to indicate LLM where to start generating the answer
        tokenize=False
    )

    # MLX
    if BACKEND == "mlx":
        # MLX library bundles whole pipline into one call
        text = _mlx_generate(
            _model,
            _tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            verbose=False,
        ).strip()
        print("RAW:" + text)

    # OpenVINO / Hugging Face Transformers
    else:
        # tokenizer converts the prompt into input tensors for the model
        inputs = _tokenizer(prompt, return_tensors="pt")

        # if using Hugging Face, move the inputs to the same device as the model (e.g. GPU) for faster inference
        if BACKEND == "transformers":
            inputs = inputs.to(_model.device)

        # call the model to generate output token ids
        output_ids = _model.generate(
            **inputs, # unpack the tokenized prompt as model inputs
            max_new_tokens=max_new_tokens,
            do_sample=False, # use greedy decoding
            pad_token_id=_tokenizer.eos_token_id  # set what to use for padding
        )

        # decode only the newly generated tokens to get the model's response as text
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:] # only take the output tokens 
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    calls = [] # to store any parsed tool calls from the generated text
    saw_tag = "<tool_call>" in text  
    i = text.find("<tool_call>") # find the first occurrence of the tool call tag in the generated text

    while i != -1:
        # JSON payload between <tool_call> and </tool_call> tags
        # extract the text after the <tool_call> tag
        after = text[i + len("<tool_call>"):] 
        # finds list of index close and open tags
        stops = [p for p in (after.find("</tool_call>"), after.find("<tool_call>")) if p != -1]
        # cut the payload at whichever comes first, the closing tag or the next
        # opening tag so a missing </tool_call> won't include the next tool_call lines
        payload = (after[:min(stops)] if stops else after).strip()

        call = None

        try:
            call = json.loads(payload) # try to parse the payload for the tool call
        except json.JSONDecodeError:
            try:
                call = repair_json(payload, return_objects=True) # try to repair the JSON if it's malformed
            except Exception:
                call = None

        # check if the parsed call is a dict with a "name" key before adding to calls
        if isinstance(call, dict) and "name" in call:
            calls.append({"name": call["name"], "arguments": call.get("arguments", {})})

        # look for the next <tool_call> tag in the text to continue parsing for more tool calls 
        i = text.find("<tool_call>", i + len("<tool_call>"))

    # returns the list tool call dicts
    if calls:
        return {"type": "tool_calls", "calls": calls}
    
    # if tool call format is malformed
    if saw_tag:
        return {"type": "text", "content": "failed to parse tool call JSON, please try again."}

    # Fallback: some models emit the bare tool-call JSON with NO
    # <tool_call> tag
    bare_pattern = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}')
    bare_matches = bare_pattern.findall(text)
    if bare_matches:
        calls = []
        for m in bare_matches:
            try:
                obj = json.loads(m)
            except json.JSONDecodeError:
                try:
                    obj = repair_json(m, return_objects=True)
                except Exception:
                    continue
            if isinstance(obj, dict) and "name" in obj:
                calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
        if calls:
            return {"type": "tool_calls", "calls": calls}

    # no (parseable) tool call found — return the generated text as the response
    return {"type": "text", "content": text}







