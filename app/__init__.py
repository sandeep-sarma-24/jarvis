import os

# onnxruntime (Silero VAD) uploads telemetry to Microsoft over HTTPS unless this is set before it loads.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
