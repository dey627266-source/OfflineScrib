"""Speech-to-text. CPU baseline now; swap in the AI Hub Whisper (NPU) next."""
from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_size: str = "base") -> str:
    # TODO: replace with Whisper exported from Qualcomm AI Hub and run via
    # providers.make_session() so encoder/decoder execute on the NPU.
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    return " ".join(seg.text.strip() for seg in segments)
