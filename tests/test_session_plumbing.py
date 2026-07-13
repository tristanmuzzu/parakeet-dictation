"""Plumbing tests for AsrSession — no real worker, no pytest.

Run:  .venv\\Scripts\\python.exe tests\\test_session_plumbing.py

A FakeConn stands in for the worker Pipe end and enforces strict send->recv
FIFO pairing, so we can prove the consumer never abandons a send without its
matching recv (the pipe stays in sync) even when a dictation is cancelled.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import dictation


class FakeConn:
    def __init__(self, responses, on_recv=None):
        self._responses = list(responses)   # texts returned, in send order
        self._pending = []                   # sent but not yet recv'd
        self._i = 0
        self._on_recv = on_recv
        self._lock = threading.Lock()

    def send(self, msg):
        with self._lock:
            self._pending.append(msg)

    def recv(self):
        with self._lock:
            assert self._pending, "recv() with no outstanding send — pipe out of sync"
            self._pending.pop(0)
            text = self._responses[self._i] if self._i < len(self._responses) else ""
            self._i += 1
            n = self._i
        if self._on_recv:      # fire outside the lock so a cancel hook can't deadlock
            self._on_recv(n)
        return ("text", text)

    def balanced(self):
        return not self._pending


def _seg(sec=1.0):
    return np.zeros(int(dictation.SAMPLE_RATE * sec), dtype="float32")


def test_ordered_join():
    conn = FakeConn(["one", "two", "three"])
    s = dictation.AsrSession(conn, threading.Lock())
    s.submit(_seg())
    s.submit(_seg())
    s.submit(_seg())
    s.close()
    s.join_consumer(timeout=5)
    assert conn.balanced(), "unbalanced send/recv"
    assert s.ordered_texts() == ["one", "two", "three"], s.ordered_texts()
    assert " ".join(s.ordered_texts()) == "one two three"
    print("ok test_ordered_join")


def test_short_dictation_single_segment():
    conn = FakeConn(["hello world"])
    s = dictation.AsrSession(conn, threading.Lock())
    s.submit(_seg(0.5))
    s.close()
    s.join_consumer(timeout=5)
    assert conn.balanced()
    assert s.ordered_texts() == ["hello world"]
    print("ok test_short_dictation_single_segment")


def test_cancel_drains_pipe():
    # Cancel the moment the first reply is received. Every send that happened
    # must still be recv'd (balanced), later queued segments must NOT be sent,
    # and nothing is kept.
    holder = {}

    def on_recv(n):
        if n == 1:
            holder["s"].cancel()

    conn = FakeConn(["a", "b", "c"], on_recv=on_recv)
    s = dictation.AsrSession(conn, threading.Lock())
    holder["s"] = s
    s.submit(_seg())
    s.submit(_seg())
    s.submit(_seg())
    s.close()
    s.join_consumer(timeout=5)
    assert conn.balanced(), "pipe left out of sync after cancel"
    assert s.ordered_texts() == [], f"cancelled session kept results: {s.ordered_texts()}"
    print("ok test_cancel_drains_pipe")


def test_empty_result_filtered():
    # Worker returning "" for a segment must not inject a blank into the join.
    conn = FakeConn(["good", "", "also good"])
    s = dictation.AsrSession(conn, threading.Lock())
    s.submit(_seg())
    s.submit(_seg())
    s.submit(_seg())
    s.close()
    s.join_consumer(timeout=5)
    assert conn.balanced()
    assert s.ordered_texts() == ["good", "also good"], s.ordered_texts()
    print("ok test_empty_result_filtered")


if __name__ == "__main__":
    test_ordered_join()
    test_short_dictation_single_segment()
    test_cancel_drains_pipe()
    test_empty_result_filtered()
    print("\nALL PLUMBING TESTS PASSED")
