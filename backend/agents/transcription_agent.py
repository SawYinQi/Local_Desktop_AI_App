import sys
from pathlib import Path

# Add backend/ to sys.path so we can import utils.runtime
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils.runtime import pick_backend

OV_MODEL_PATH = BACKEND_DIR / "models" / "whisper-base-int8-ov" # OpenVINO-optimized model path
HF_MODEL_PATH = BACKEND_DIR / "models" / "whisper-base" # Hugging Face model path 


_pipe = None

# Lazy load pipeline 
def _ensure_loaded():
    global _pipe
    # if already loaded, skip.
    if _pipe is not None:
        return
    
    # else load the appropriate model and create the transcription pipeline based on the host platform
    backend = pick_backend() 
    if backend == "openvino" and not OV_MODEL_PATH.exists():
        backend = "transformers"  
    
    # load OpenVINO model 
    if backend == "openvino":

        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        from transformers import AutoProcessor

        model = OVModelForSpeechSeq2Seq.from_pretrained(str(OV_MODEL_PATH),device="GPU") # load the OpenVINO-optimized model
        processor = AutoProcessor.from_pretrained(str(OV_MODEL_PATH))

    # load Hugging Face model 
    else:

        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        model = WhisperForConditionalGeneration.from_pretrained(str(HF_MODEL_PATH))
        if torch.backends.mps.is_available():
            model = model.to("mps")
        elif torch.cuda.is_available():
            model = model.to("cuda")
        processor = AutoProcessor.from_pretrained(str(HF_MODEL_PATH))

    from transformers import pipeline

    # transcription pipline
    _pipe = pipeline(
        "automatic-speech-recognition", # task name for the pipeline audio -> text
        model=model,  # the loaded model (either OpenVINO or Hugging Face)
        tokenizer=processor.tokenizer,  # the processor's tokenizer for text encoding/decoding
        feature_extractor=processor.feature_extractor, # the processor's feature extractor for audio preprocessing
        chunk_length_s=30, stride_length_s=5  # process long audio in 30s chunks with 5s overlap to avoid cutting off words
    )

def transcribe(video_path: str) -> str:
    # initialise the transcription pipeline if not already loaded
    _ensure_loaded()
    # call the pipeline on the given video file path and return the transcribed text
    return _pipe(video_path)["text"]                  

