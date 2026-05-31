import sys
from pathlib import Path

# Add backend/ to sys.path so we can import utils.runtime
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils.runtime import pick_backend

OV_MODEL_PATH = BACKEND_DIR / "models" / "qwen2.5-vl-7b-int4"   # OpenVINO (Intel)
HF_MODEL_PATH = BACKEND_DIR / "models" / "qwen2.5-vl-3b"        # HF (Mac / other)


NUM_FRAMES = 6 # number of frames to sample
MAX_FRAME_SIDE = 512 # cap longest side of sampled frames to limit vision tokens and MPS memory  
MAX_NEW_TOKENS = 512 # max new tokens to generate in response 

_model = None
_processor = None
_BACKEND = None

# Lazy load the model and processor when the tool is first called
def _ensure_loaded():
    global _model, _processor, _BACKEND

    if _model is not None:
        return

    backend = pick_backend()
  
    if backend == "openvino" and not OV_MODEL_PATH.exists():
        backend = "transformers"

    if backend == "openvino":
        from optimum.intel.openvino import OVModelForVisualCausalLM
        from transformers import AutoProcessor
        _model = OVModelForVisualCausalLM.from_pretrained(str(OV_MODEL_PATH))
        _processor = AutoProcessor.from_pretrained(str(OV_MODEL_PATH))

    else:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        _model = AutoModelForImageTextToText.from_pretrained(
            str(HF_MODEL_PATH),
            dtype=torch.float16
        )
        if torch.backends.mps.is_available():
            _model = _model.to("mps")
        elif torch.cuda.is_available():
            _model = _model.to("cuda")
        _processor = AutoProcessor.from_pretrained(str(HF_MODEL_PATH))

    _BACKEND = backend

# Sample frames from the video and preprocess them for the model. Returns a list of PIL Images.
def _sample_frames(video_path: str, n: int = NUM_FRAMES):
    import cv2
    from PIL import Image

    # open video file with OpenCV
    cap = cv2.VideoCapture(video_path)
    # get total number of frames in the video
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    if total <= 0:
        return

    # Get the frame indices to sample, evenly spaced n frames
    indices = [int(total * i / n) for i in range(n)]
    frames = []

    # read and preprocess each sampled frame
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i) # skip to the frame index
        success, frame = cap.read()  

        # if read successfully 
        if success:
            # OpenCV reads BGR; convert to RGB for PIL/transformers.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image 
            img = Image.fromarray(rgb)
            # shrinks frame resolution 
            img.thumbnail((MAX_FRAME_SIDE, MAX_FRAME_SIDE))
            frames.append(img)
    cap.release() # close video
    return frames


def analyze(query: str, file_path: str) -> str:

    _ensure_loaded() 

    frames = _sample_frames(file_path) 
    if not frames:
        return "Could not extract any frames from the video."

    # Initilaize message with the user query and the sampled frames
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": f} for f in frames],
            {"type": "text", "text": query}
        ]
    }]

    # format message and frame inputs for the model with special tags
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # convert text and frames into tensors for model
    inputs = _processor(
        text=[text],
        images=frames,
        videos=None,
        return_tensors="pt",
        padding=True
    )

    if _BACKEND == "transformers":
        inputs = inputs.to(_model.device)

    # generate output token ids from the model based on the input query and frames
    output_ids = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # get only the newly generated tokens and decode them to text for the final response
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    response = _processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    return response.strip()
