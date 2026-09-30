# OfflineScribe

**A private, on-device meeting assistant for Snapdragon-powered HP PCs.**

OfflineScribe records a meeting, transcribes it, and produces a summary with decisions and action items. Everything runs locally. No audio or text leaves the laptop.

> **Status:** hackathon prototype. The pipeline structure and CPU baseline are in place. NPU acceleration is wired through ONNX Runtime's QNN execution provider and needs models exported from Qualcomm AI Hub (see below).

## Why on-device, and why Snapdragon

- **Privacy:** meeting audio is sensitive (legal, health, finance, HR). Nothing is uploaded.
- **Battery:** the Snapdragon Hexagon NPU runs AI workloads at low power, so long meetings do not drain the laptop.
- **Offline:** works without a connection and without a subscription.

## Pipeline

```
Mic / audio file -> Speech-to-text (Whisper) -> Local transcript -> Local LLM summary -> Notes
```

| Stage | Model | Target hardware |
|---|---|---|
| Speech-to-text | Whisper Tiny / Base (Qualcomm AI Hub) | Hexagon NPU via QNN |
| Summary and action items | Llama 3.2 3B Instruct, quantized (Qualcomm AI Hub) | Hexagon NPU via QNN |

Please confirm the exact model names and export options on [Qualcomm AI Hub](https://aihub.qualcomm.com) before relying on them.

## Project layout

```
src/offlinescribe/
  providers.py    # picks NPU (QNN) or CPU for ONNX Runtime
  transcribe.py   # audio -> text
  summarize.py    # text -> summary + action items
  cli.py          # command-line entry point
models/           # put exported/downloaded model files here
```

## Setup (Snapdragon-powered HP PC, Windows on Arm)

1. Install a native Arm64 build of Python 3.11+.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. For NPU use, install ONNX Runtime with QNN support for Windows on Arm and the Qualcomm AI Engine Direct (QNN) runtime, following Qualcomm's current documentation.
4. Export Whisper and the LLM from Qualcomm AI Hub and place the files in `models/`.

## Usage

```
python -m offlinescribe.cli meeting.wav --out notes.md
```

Use `--device cpu` to force the CPU baseline and `--device npu` to require the NPU.

## Benchmarking (for the demo)

Run the same file with `--device cpu` and `--device npu` and compare the printed timings. Record battery drain over a longer recording for the demo.

## Roadmap

- [ ] Whisper on the NPU
- [ ] LLM summary on the NPU
- [ ] Live microphone capture
- [ ] Search across past meetings (local embeddings)
- [ ] Simple desktop UI

## License

MIT
