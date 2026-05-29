import whisper

_model = None # global variable to cache the loaded model

# Lazy load the Whisper model 
def _get_model():
    global _model

    # load the model once and cache it for future use
    if _model is None:
        print("Transcription: loading Whisper model...")
        _model = whisper.load_model("base") 
        print("Transcription: model loaded.")
    return _model

# Transcribe the video at the given path and return the transcript text
def transcribe(video_path: str) -> str:
    model = _get_model()
    print(f"Transcription: transcribing {video_path} ...")
    result = model.transcribe(video_path, fp16=False) 
    text = result["text"].strip()
    print("Transcription: done.")
    return text