#!/usr/bin/env python3
"""DC bias removal.

Regression test for the bug found on 2026-08-09: this laptop's digital
microphone (Mic1) puts a large constant DC offset on every sample. Measured at
16 kHz mono in a quiet room: mean +0.220 against an RMS of 0.221, i.e. ~87% of
the captured "signal" was DC.

That is not a headroom problem, it is a segmentation problem. Segmenter._step
classifies a frame as speech by comparing its RMS to ABS_FLOOR (0.004), and a
0.220 offset is 55x that floor, so every frame reads as speech and the noise
floor calibrates to the bias instead of the room.
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location("dictmod", os.path.join(ROOT, "dictation.py"))
d = importlib.util.module_from_spec(spec)
sys.modules["dictmod"] = d
spec.loader.exec_module(d)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


def quiet_room_with_bias(seconds=3.0, bias=0.220, noise=0.013):
    """Mimic the measured capture: small room noise sitting on a large offset."""
    rng = np.random.default_rng(1234)
    n = int(d.SAMPLE_RATE * seconds)
    return (rng.normal(0.0, noise, n) + bias).astype("float32")


print("DCBlocker")
x = quiet_room_with_bias()
blk = d.DCBlocker(d.SAMPLE_RATE)
y = np.concatenate([blk.process(x[i:i + 2048]) for i in range(0, x.size, 2048)])

check("input really is biased", abs(x.mean() - 0.220) < 0.01, f"mean={x.mean():.4f}")
check("bias is removed", abs(y.mean()) < 0.005, f"mean={y.mean():+.6f}")
check("length is preserved", y.size == x.size, f"{y.size} vs {x.size}")
check("output is float32", y.dtype == np.float32, str(y.dtype))

# The room noise itself must survive: this is a DC blocker, not a gate.
check("room noise survives", 0.5 < (y.std() / x.std()) < 1.5,
      f"ratio={y.std() / x.std():.3f}")

# No step discontinuity where two processed blocks meet.
edges = [abs(float(y[i] - y[i - 1])) for i in range(2048, y.size, 2048)]
check("no step at block boundaries", max(edges) < 0.15, f"max jump={max(edges):.4f}")

print("remove_dc (batch path)")
b = d.remove_dc(x)
check("batch bias removed", abs(b.mean()) < 1e-5, f"mean={b.mean():+.2e}")
check("batch length preserved", b.size == x.size)
check("empty input is safe", d.remove_dc(np.zeros(0, dtype="float32")).size == 0)

print("Segmenter no longer calibrates onto the bias")
seg = d.Segmenter()
for i in range(0, x.size, 2048):
    seg.push(x[i:i + 2048])
check("noise floor is near the room, not the bias", seg._floor < 0.05,
      f"floor={seg._floor:.5f} (bias was 0.220)")

# A silent-but-biased take must not be CLASSIFIED as speech.
#
# Deliberately asserting _speech_samples rather than "finalize() returns None".
# finalize() also emits on `loud_ok`, a safety net that fires whenever a take is
# long enough and its RMS clears ABS_FLOOR (0.004). This microphone's own noise
# floor measures ~0.016 RMS once the DC is stripped, which is four times that
# constant, so loud_ok fires on silence here and the take is still submitted.
#
# That is on purpose and must not be "fixed" by raising ABS_FLOOR: dropping words
# the user actually said is a far worse failure than sending a quiet take to the
# model and getting an empty string back. What this fix guarantees is that the
# frame classifier is no longer blinded by the bias.
seg2 = d.Segmenter()
for i in range(0, x.size, 2048):
    seg2.push(x[i:i + 2048])
need = int(d.MIN_SECONDS * d.SAMPLE_RATE)
check("biased silence is not classified as speech", seg2._speech_samples < need,
      f"speech_samples={seg2._speech_samples} >= {need}")
seg2.finalize()

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all passed")
