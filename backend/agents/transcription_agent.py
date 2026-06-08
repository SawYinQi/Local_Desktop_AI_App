import sys
from pathlib import Path

# Add backend/ to sys.path so we can import utils.runtime
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils.runtime import pick_backend, has_model

OV_MODEL_PATH = BACKEND_DIR / "models" / "whisper-base-int8-ov" # OpenVINO-optimized model path
HF_MODEL_PATH = BACKEND_DIR / "models" / "whisper-base" # Hugging Face model path 
MLX_MODEL_PATH = BACKEND_DIR / "models" / "whisper-base-mlx-4bit" # MLX model path (Apple Silicon)


_pipe = None       # transformers / OpenVINO ASR pipeline (MLX doesn't use this)
_BACKEND = None

# Lazy load the model/pipeline when the tool is first called
def _ensure_loaded():
    global _pipe, _BACKEND
    # if already initialised, skip (otherwise the pipeline would be rebuilt every call).
    if _BACKEND is not None:
        return

    backend = pick_backend()

    # MLX (Apple Silicon): mlx-whisper exposes a stateless transcribe() that loads and
    # caches the model itself, so there's no pipeline to build — just check the model exists.
    if backend == "mlx":
        if not has_model(MLX_MODEL_PATH):
            raise RuntimeError(f"No MLX model available at {MLX_MODEL_PATH}")
        _BACKEND = backend
        return

    # load OpenVINO model
    if backend == "openvino":
        if not has_model(OV_MODEL_PATH):
            raise RuntimeError(f"No OpenVINO model available at {OV_MODEL_PATH}")

        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        from transformers import AutoProcessor

        model = OVModelForSpeechSeq2Seq.from_pretrained(str(OV_MODEL_PATH), device="GPU") # load the OpenVINO-optimized model
        processor = AutoProcessor.from_pretrained(str(OV_MODEL_PATH))

    # load Hugging Face model
    else:
        if not has_model(HF_MODEL_PATH):
            raise RuntimeError(f"No Hugging Face model available at {HF_MODEL_PATH}")

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
    _BACKEND = backend


def transcribe(video_path: str) -> str:
    # initialise the model/pipeline if not already loaded
    _ensure_loaded()

    # MLX (Apple Silicon): mlx-whisper decodes the audio (via ffmpeg) and runs natively on Metal.
    if _BACKEND == "mlx":
        import mlx_whisper
        result = mlx_whisper.transcribe(
            video_path,
            path_or_hf_repo=str(MLX_MODEL_PATH),
            verbose=False,
        )
        return result["text"].strip()

    # OpenVINO / Transformers: call the pipeline on the video file path and return the text
    return _pipe(video_path)["text"]

