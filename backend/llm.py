import json
import platform
import re
from pathlib import Path
from utils.runtime import pick_backend 

# Model path for OpenVINO(intel) and Hugging Face(Mac/non-intel). 
OV_MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-7b-int4"
HF_MODEL_PATH = Path(__file__).parent / "models" / "qwen2.5-3b-instruct"

# check if OpenVINO model exist at the given path
def _has_ov_model(p: Path) -> bool:
    return (p / "openvino_model.bin").exists()

# check if Hugging Face model exist at the given path 
def _has_hf_model(p: Path) -> bool:
    return (p / "config.json").exists() and not _has_ov_model(p)


BACKEND = pick_backend() 

print(f"LLM: host = {platform.system()}/{platform.machine()}, backend = {BACKEND}")


# OpenVINO
if BACKEND == "openvino":
    MODEL_PATH = OV_MODEL_PATH
    # check model availability and raise error if not found, before loading.
    if not _has_ov_model(MODEL_PATH):
        raise RuntimeError(f"No OpenVINO model available at {MODEL_PATH}")
    
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer
    
    print(f"LLM: loading {MODEL_PATH} ...")
    _model = OVModelForCausalLM.from_pretrained(str(MODEL_PATH)) # load the OpenVINO-optimized model
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH)) # load the tokeniser
    DEVICE = None

# Hugging Face Transformers
else:  
    MODEL_PATH = HF_MODEL_PATH
    # check model availability and raise error if not found, before loading.
    if not _has_hf_model(MODEL_PATH):
        raise RuntimeError(f"No Hugging Face model available at {MODEL_PATH}")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"LLM: loading {MODEL_PATH}")

    # Loads model
    _model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float16,
        device_map="auto", # automatically map model layers to available devices (GPU/CPU)
    )
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH)) # load the tokeniser

print("LLM: model loaded.")

def chat(messages: list, tools: list | None = None, max_new_tokens: int = 512) -> dict:
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

    # check if the model is trying to call a tool by looking for a special <tool_call>...</tool_call> pattern in the output text.
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)

    if match:
        try:
            # parse the JSON inside the tags to get the tool call details (name and arguments) 
            call = json.loads(match.group(1)) 
            # return a dict indicating this is a tool call, along with the tool name and arguments for the orchestrator to execute
            return {"type": "tool_call", "name": call["name"], "arguments": call.get("arguments", {})}
        except (json.JSONDecodeError, KeyError):
            pass

    # if no tool call pattern is found, return the generated text as the response
    return {"type": "text", "content": text} 






