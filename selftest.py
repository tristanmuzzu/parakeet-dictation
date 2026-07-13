"""Self-test: downloads the model (first run only) and verifies the ASR
pipeline end-to-end without needing a microphone.

Run:  .venv\\Scripts\\python.exe selftest.py
"""
import numpy as np
import onnx_asr

print("loading model (first run downloads ~460 MB, then cached)...", flush=True)
try:
    m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")
    print("loaded int8 model", flush=True)
except Exception as e:
    print("int8 unavailable (", e, ") - falling back to fp32", flush=True)
    m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
print("model loaded OK", flush=True)

# 1 second of silence -> should return an empty/short string without crashing
audio = np.zeros(16000, dtype="float32")
txt = m.recognize(audio, sample_rate=16000)
print("recognize() on silence returned:", repr(txt), flush=True)
print("SELFTEST_OK", flush=True)
