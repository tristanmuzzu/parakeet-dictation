"""Real-worker smoke test: spawn the actual ASR worker (model already cached),
push 8s of zeros through the full segment path, and prove there is no deadlock
and that an empty-string result is handled cleanly.

Run:  .venv\\Scripts\\python.exe tests\\smoke_pipeline.py
"""
import os
import sys
import threading
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import dictation


def main():
    parent, child = mp.Pipe()
    worker = mp.Process(target=dictation.asr_worker, args=(child,),
                        name="asr-worker", daemon=True)
    worker.start()

    # Blocks until model load + warmup are done.
    kind, payload = parent.recv()
    assert kind == "ready", f"worker did not become ready: {kind} {payload}"
    print("worker ready (model loaded + warmed up)")

    lock = threading.Lock()
    sess = dictation.AsrSession(parent, lock)

    zeros = np.zeros(8 * dictation.SAMPLE_RATE, dtype="float32")
    # Drive the segmenter directly with 8s of zeros (no mic/feeder). Zeros carry
    # no speech, so the segmenter emits nothing on push or finalize.
    for seg in sess.segmenter.push(zeros):
        sess.submit(seg)
    tail = sess.segmenter.finalize()
    if tail is not None:
        sess.submit(tail)

    # Force the whole 8s take through the real consumer -> worker roundtrip so we
    # exercise the send/recv path and confirm the worker returns "" for silence.
    sess.submit(zeros)
    sess.close()
    sess.join_consumer(timeout=60)
    assert not sess._consumer.is_alive(), "consumer thread hung — deadlock"

    assert sess.results.get(0) == "", f"expected empty result, got {sess.results.get(0)!r}"
    assert sess.ordered_texts() == [], f"empty result leaked into join: {sess.ordered_texts()}"
    print("8s zeros -> empty-string result handled, no deadlock")

    parent.send(("quit", None))
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    mp.freeze_support()
    main()
