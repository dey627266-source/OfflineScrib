import argparse
import time

from .summarize import summarize
from .transcribe import transcribe


def main():
    ap = argparse.ArgumentParser(description="OfflineScribe: private on-device meeting notes")
    ap.add_argument("audio", help="path to a meeting recording (wav/mp3)")
    ap.add_argument("--llm", default="models/llm.gguf", help="path to the local LLM file")
    ap.add_argument("--out", default="notes.md")
    ap.add_argument("--device", choices=["auto", "npu", "cpu"], default="auto")
    args = ap.parse_args()

    t0 = time.time()
    text = transcribe(args.audio)
    t1 = time.time()
    notes = summarize(text, args.llm)
    t2 = time.time()

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Meeting notes\n\n" + notes + "\n\n## Transcript\n\n" + text + "\n")
    print(f"Transcription: {t1 - t0:.1f}s | Summary: {t2 - t1:.1f}s | Saved to {args.out}")


if __name__ == "__main__":
    main()
