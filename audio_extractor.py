import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from faster_whisper import WhisperModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_WHISPER_PATH = os.path.join(BASE_DIR, "models", "whisper-base")

_audio_model = None

def _get_whisper_model():
    global _audio_model
    if _audio_model is None:
        if not os.path.exists(LOCAL_WHISPER_PATH):
            raise FileNotFoundError(f"Local Whisper model directory missing at {LOCAL_WHISPER_PATH}")
        _audio_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _audio_model

def extract_audio_transcript(audio_path: str) -> str:
    model = _get_whisper_model()
    try:
        segments, _ = model.transcribe(audio_path, beam_size=5)
        transcript = " ".join([segment.text for segment in segments]).strip()
        return transcript
    except Exception as e:
        print(f"Error transcribing audio {audio_path}: {e}")
        return ""
