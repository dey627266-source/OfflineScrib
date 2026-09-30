"""Choose where ONNX Runtime executes: Snapdragon NPU (QNN) or CPU."""
import onnxruntime as ort


def npu_available() -> bool:
    return "QNNExecutionProvider" in ort.get_available_providers()


def make_session(model_path: str, device: str = "auto") -> ort.InferenceSession:
    """device: 'auto' (NPU if present), 'npu' (require it), or 'cpu'."""
    use_npu = device == "npu" or (device == "auto" and npu_available())
    if device == "npu" and not npu_available():
        raise RuntimeError("QNNExecutionProvider not available on this machine.")
    if use_npu:
        providers = [
            ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)
