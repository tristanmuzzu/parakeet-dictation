"""Unit tests for the RMS Segmenter — pure logic, no worker, no pytest.

Run:  .venv\\Scripts\\python.exe tests\\test_segmenter.py

Synthetic waveforms: noise bursts at amp 0.3 stand in for speech, amp 0.001 for
silence. Every waveform starts with a short silence so the noise-floor
calibration sees the room, not the voice (the same thing that happens in real
use: you press the hotkey, then start talking).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dictation import Segmenter, SAMPLE_RATE, MIN_SEG, MAX_SEG

SR = SAMPLE_RATE


def silence(sec, rng):
    return (0.001 * rng.standard_normal(int(sec * SR))).astype("float32")


def speech(sec, rng):
    return (0.3 * rng.standard_normal(int(sec * SR))).astype("float32")


def test_cut_on_speech_then_silence():
    rng = np.random.default_rng(0)
    wav = np.concatenate([silence(0.4, rng), speech(3.2, rng),
                          silence(0.8, rng), speech(1.0, rng)])
    seg = Segmenter()
    emitted = seg.push(wav)
    assert len(emitted) == 1, f"expected 1 cut, got {len(emitted)}"
    # Cut fires after 0.6s of silence; trailing silence is trimmed to ~0.2s, so
    # the segment is roughly 0.4 (lead) + 3.2 (speech) + 0.2 (kept) = 3.8s.
    got = emitted[0].size / SR
    assert abs(got - 3.8) < 0.2, f"segment length {got:.2f}s not ~3.8s"
    tail = seg.finalize()
    assert tail is not None, "tail (1.0s speech) should still emit"
    print("ok test_cut_on_speech_then_silence")


def test_hard_cut_at_max_seg():
    rng = np.random.default_rng(1)
    wav = np.concatenate([silence(0.4, rng), speech(25.0, rng)])
    seg = Segmenter()
    emitted = seg.push(wav)
    assert len(emitted) >= 1, "continuous speech must hard-cut at MAX_SEG"
    first = emitted[0].size / SR
    assert abs(first - MAX_SEG) < 1.0, f"first cut {first:.2f}s not ~{MAX_SEG}s"
    tail = seg.finalize()
    assert tail is not None, "remaining speech should emit at finalize"
    print("ok test_hard_cut_at_max_seg")


def test_tail_finalize():
    # A single 2s utterance: never cut mid-stream, emitted only at finalize.
    rng = np.random.default_rng(2)
    wav = np.concatenate([silence(0.4, rng), speech(2.0, rng), silence(0.3, rng)])
    seg = Segmenter()
    emitted = seg.push(wav)
    assert len(emitted) == 0, f"2s utterance must not cut, got {len(emitted)}"
    tail = seg.finalize()
    assert tail is not None, "finalize must emit the 2s tail"
    assert tail.size / SR > 1.5, "tail should carry the whole utterance"
    print("ok test_tail_finalize")


def test_no_cut_short_utterance():
    # Below MIN_SEG of speech and below the silence threshold: no cut at all.
    rng = np.random.default_rng(3)
    wav = np.concatenate([silence(0.4, rng), speech(1.5, rng)])
    seg = Segmenter()
    assert seg.push(wav) == [], "short utterance must not produce a cut"
    print("ok test_no_cut_short_utterance")


def test_pure_silence_no_segment():
    rng = np.random.default_rng(4)
    seg = Segmenter()
    assert seg.push(silence(8.0, rng)) == [], "silence must never cut"
    assert seg.finalize() is None, "silence tail must not emit"
    print("ok test_pure_silence_no_segment")


def test_determinism():
    def run(seed):
        rng = np.random.default_rng(seed)
        wav = np.concatenate([silence(0.4, rng), speech(3.2, rng),
                              silence(0.8, rng), speech(2.0, rng),
                              silence(0.7, rng)])
        seg = Segmenter()
        out = seg.push(wav.copy())
        return out, seg.finalize()

    o1, t1 = run(5)
    o2, t2 = run(5)
    assert len(o1) == len(o2), "segment count not deterministic"
    for a, b in zip(o1, o2):
        assert np.array_equal(a, b), "segment content not deterministic"
    assert (t1 is None) == (t2 is None)
    if t1 is not None:
        assert np.array_equal(t1, t2), "tail not deterministic"
    print("ok test_determinism")


if __name__ == "__main__":
    test_cut_on_speech_then_silence()
    test_hard_cut_at_max_seg()
    test_tail_finalize()
    test_no_cut_short_utterance()
    test_pure_silence_no_segment()
    test_determinism()
    print("\nALL SEGMENTER TESTS PASSED")
