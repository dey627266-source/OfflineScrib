"""Local LLM summary. CPU baseline via llama.cpp with a quantized GGUF model."""
from llama_cpp import Llama

PROMPT = """You are a meeting assistant. From the transcript below, write:
1. A short summary (3-5 sentences)
2. Key decisions (bullets)
3. Action items with owners if mentioned (bullets)

Transcript:
{transcript}
"""


def summarize(transcript: str, model_path: str) -> str:
    # TODO: move to a Llama 3.2 3B build exported from Qualcomm AI Hub for the NPU.
    llm = Llama(model_path=model_path, n_ctx=8192, verbose=False)
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": PROMPT.format(transcript=transcript)}],
        max_tokens=600,
    )
    return out["choices"][0]["message"]["content"]
