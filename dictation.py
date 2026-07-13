"""
Parakeet local dictation — Ctrl+Win toggle with on-screen indicator.

Hotkeys
  Ctrl + Win      start / stop dictation (press once to start, again to insert text)
  Esc             cancel the current recording without inserting anything
  Ctrl + Alt + Q  quit the app

Architecture: TWO processes.
  - Main process: keyboard hook, tkinter overlay, microphone capture. Kept
    deliberately light — Windows silently removes low-level keyboard hooks
    from processes that hog the CPU (LowLevelHooksTimeout), which is exactly
    what loading a 600 MB ONNX model does. So the model never lives here.
  - Worker process: loads nvidia parakeet-tdt-0.6b-v3 (onnx-asr) and serves
    recognize() requests over a multiprocessing Pipe.

Text is inserted by copying to the clipboard and sending Ctrl+V into the
currently focused window, then restoring the previous clipboard.
"""

import multiprocessing as mp
import os
import queue
import re
import socket
import threading
import time
import logging

import numpy as np

logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictation.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
)
log = logging.getLogger("dictation")

SAMPLE_RATE = 16000
MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
MIN_SECONDS = 0.3  # ignore taps shorter than this
# Hotkeys (Ctrl+Win toggle, Esc cancel, Ctrl+Alt+Q quit) are implemented via a
# raw keyboard hook in main() — see on_key() and the comment there for why
# add_hotkey() string combos are NOT used.
CLEANUP = True     # strip filler words (um/uh/...) and tidy spacing

# Incremental transcription tuning. The whole point of streaming is to move the
# ASR cost off the critical path: we chop the take into speech segments while
# the user is still talking and transcribe each in the background, so only the
# final tail is left to recognize when they hit stop. These numbers control the
# RMS-energy segmenter (no VAD library on purpose — one more dependency for a
# job cheap arithmetic already does well enough).
FRAME_MS = 30            # RMS is measured over ~30 ms frames
MIN_SEG = 3.0            # need this many seconds of speech before a silence cut
MAX_SEG = 20.0           # hard cut here even with no pause, so latency is bounded
SILENCE_CUT = 0.6        # trailing silence that triggers a speech-based cut
KEEP_SIL = 0.2           # trailing silence kept after trimming a cut segment
CALIB_SEC = 0.3          # first slice used to estimate the room's noise floor
ABS_FLOOR = 0.004        # RMS below this is silence no matter what calibration says
FLOOR_MULT = 3.0         # speech must exceed this multiple of the learned floor
EMA_ALPHA = 0.05         # how fast the noise floor tracks (slow, non-speech only)

# Standalone filler tokens (EN + DE), removed only as whole words. Conservative
# on purpose: does NOT touch "like"/"you know" etc. so meaning is never damaged.
_FILLER = re.compile(
    r"\b(u+m+|u+h+|uh+m+|e+r+m+|err+|hm+|mm+h*|ä+h+m*|ehm+)\b[,]?",
    re.IGNORECASE,
)


def clean_text(t):
    """Cheap, instant tidy-up of raw ASR text (no LLM)."""
    if not t:
        return t
    t = _FILLER.sub(" ", t)
    # collapse immediate duplicate words: "the the cat" -> "the cat"
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)   # no space before punctuation
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"^[\s,.;:]+", "", t)         # no leading punctuation/space
    if t:
        t = t[0].upper() + t[1:]
    return t


# ----------------------------------------------------------------------------
# Segmenter: pure, deterministic speech chopping (no I/O, no threads).
#
# It eats raw audio blocks and decides where to cut the take into segments the
# worker can transcribe independently. Kept side-effect free so it is trivially
# unit-testable with synthetic arrays — the streaming plumbing that feeds it and
# ships the pieces to the worker lives entirely in AsrSession.
# ----------------------------------------------------------------------------
class Segmenter:
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
        self.frame_len = max(1, int(sr * FRAME_MS / 1000))
        self._pending = np.zeros(0, dtype="float32")   # samples < one full frame
        # Start at the absolute floor and let calibration + EMA refine it. The
        # floor is deliberately NOT reset between segments: the room does not
        # get quieter halfway through a dictation.
        self._floor = ABS_FLOOR
        self._calib_remaining = int(CALIB_SEC * sr)
        self._calib_sum = 0.0
        self._calib_count = 0
        self._reset_segment()

    def _reset_segment(self):
        self._seg = []            # frames making up the current (open) segment
        self._seg_len = 0         # total samples in the current segment
        self._speech_samples = 0  # samples classified as speech in it
        self._trailing_sil = 0    # consecutive silence samples at the very end

    def push(self, block):
        """Consume one audio block (1-D float32 @ sr). Returns a list of cut
        segments (numpy arrays), most often empty."""
        block = np.asarray(block, dtype="float32").reshape(-1)
        out = []
        if block.size == 0:
            return out
        buf = np.concatenate([self._pending, block]) if self._pending.size else block
        n = buf.size // self.frame_len
        used = n * self.frame_len
        self._pending = buf[used:].copy()   # carry the ragged tail to next push
        for i in range(n):
            frame = buf[i * self.frame_len:(i + 1) * self.frame_len]
            seg = self._step(frame, allow_emit=True)
            if seg is not None and seg.size:
                out.append(seg)
        return out

    def finalize(self):
        """Flush the remaining tail at stop time. May be short; we still emit it
        as long as it carries real speech, because it holds the user's last
        words and must never be dropped."""
        if self._pending.size:
            self._step(self._pending, allow_emit=False)
            self._pending = np.zeros(0, dtype="float32")
        if not self._seg:
            return None
        concat = np.concatenate(self._seg)
        speech_ok = self._speech_samples >= MIN_SECONDS * self.sr
        # Safety net: if calibration got polluted (the user spoke with no leading
        # silence) speech frames can be misread as silence, so also emit when the
        # tail simply carries sound above the absolute floor for long enough.
        rms = float(np.sqrt(np.mean(concat * concat))) if concat.size else 0.0
        loud_ok = concat.size >= MIN_SECONDS * self.sr and rms > ABS_FLOOR
        self._reset_segment()
        return concat if (speech_ok or loud_ok) else None

    def _step(self, frame, allow_emit):
        rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
        self._seg.append(frame)
        self._seg_len += frame.size
        if self._calib_remaining > 0:
            self._calib_sum += rms
            self._calib_count += 1
            self._calib_remaining -= frame.size
            if self._calib_remaining <= 0:
                self._floor = self._calib_sum / max(1, self._calib_count)
        thresh = max(self._floor * FLOOR_MULT, ABS_FLOOR)
        if rms > thresh:
            self._speech_samples += frame.size
            self._trailing_sil = 0
        else:
            self._trailing_sil += frame.size
            if self._calib_remaining <= 0:   # track the floor on true silence only
                self._floor = (1 - EMA_ALPHA) * self._floor + EMA_ALPHA * rms
        if not allow_emit:
            return None
        # Cut A: enough speech banked and a real pause has landed. Trim the pause.
        if (self._speech_samples >= MIN_SEG * self.sr
                and self._trailing_sil >= SILENCE_CUT * self.sr):
            return self._cut(trim=True)
        # Cut B: segment got too long with no pause — cut hard to bound latency.
        if self._seg_len >= MAX_SEG * self.sr:
            return self._cut(trim=False)
        return None

    def _cut(self, trim):
        concat = np.concatenate(self._seg)
        if trim:
            keep = int(KEEP_SIL * self.sr)
            if self._trailing_sil > keep:
                drop = self._trailing_sil - keep
                concat = concat[:max(0, concat.size - drop)]
        self._reset_segment()
        return concat


# ----------------------------------------------------------------------------
# Worker process: owns the model. Heavy CPU stays out of the hook process.
# ----------------------------------------------------------------------------
def asr_worker(conn):
    import numpy as np  # noqa: F401  (worker-side import)
    import onnx_asr

    # Run recognition at above-normal priority so a busy machine (other heavy
    # background processes) can't starve it into multi-minute transcriptions.
    if os.name == "nt":
        try:
            import ctypes
            ABOVE_NORMAL = 0x00008000
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), ABOVE_NORMAL)
            log.info("worker: priority set to ABOVE_NORMAL")
        except Exception:
            log.exception("worker: could not raise priority")

    try:
        t0 = time.time()
        try:
            model = onnx_asr.load_model(MODEL_NAME, quantization="int8")
            log.info("worker: loaded int8 model")
        except Exception:
            log.exception("worker: int8 load failed, falling back to fp32")
            model = onnx_asr.load_model(MODEL_NAME)
        log.info("worker: model loaded OK in %.1fs", time.time() - t0)
        # Warm the graph once on 5s of silence. The first real recognize() pays
        # a large one-off cost (ONNX session warmup, allocator, thread pool);
        # paying it here means the first user dictation is fast like the rest.
        try:
            t1 = time.time()
            model.recognize(np.zeros(SAMPLE_RATE * 5, dtype="float32"),
                            sample_rate=SAMPLE_RATE)
            log.info("worker: warmup done in %.1fs", time.time() - t1)
        except Exception:
            log.exception("worker: warmup failed (non-fatal)")
        conn.send(("ready", None))
    except Exception as e:
        log.exception("worker: model load FAILED")
        conn.send(("load_error", str(e)))
        return

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg[0] == "quit":
            break
        if msg[0] == "recognize":
            audio, sr = msg[1], msg[2]
            try:
                text = model.recognize(audio, sample_rate=sr)
                conn.send(("text", (text or "").strip()))
            except Exception as e:
                log.exception("worker: recognize FAILED")
                conn.send(("asr_error", str(e)))


# ----------------------------------------------------------------------------
# Main process: hook + UI + mic. Everything below must stay cheap.
# ----------------------------------------------------------------------------
ui_q: "queue.Queue[tuple]" = queue.Queue()
state = {"recording": False, "busy": False, "model_ready": False}
frames: list = []
frames_lock = threading.Lock()
stream = None
worker_conn = None
worker_lock = threading.Lock()
session = None          # the live AsrSession while recording / draining the tail
_lock_sock = None


# ----------------------------------------------------------------------------
# AsrSession: the per-dictation streaming plumbing around the Segmenter.
#
# One session per recording. A feeder thread drains freshly captured audio into
# the Segmenter and queues each emitted segment; a SINGLE consumer thread ships
# them to the worker one at a time (under worker_lock) and stores the results in
# submit order. Single consumer = strict FIFO, so segments are never reordered.
#
# The consumer sends and recv's as one atomic pair under the lock, so a send is
# never left without its matching recv — that is what keeps the pipe in sync
# even when a dictation is cancelled mid-flight (we finish the in-flight recv,
# then just discard the result).
# ----------------------------------------------------------------------------
class AsrSession:
    def __init__(self, conn, lock):
        self.conn = conn
        self.lock = lock
        self.segmenter = Segmenter()
        self.seg_q: "queue.Queue" = queue.Queue()
        self.results = {}          # index -> transcribed text
        self.cancelled = False
        self._submitted = 0
        self._feeding = True
        self._cursor = 0           # how far into `frames` the feeder has read
        self._feeder = None
        self._consumer = threading.Thread(target=self._consume,
                                          name="asr-consumer", daemon=True)
        self._consumer.start()

    def start_feeder(self):
        self._feeder = threading.Thread(target=self._feed,
                                        name="asr-feeder", daemon=True)
        self._feeder.start()

    def submit(self, audio):
        idx = self._submitted
        self._submitted += 1
        self.seg_q.put((idx, audio))
        return idx

    def close(self):
        self.seg_q.put(None)       # sentinel: consumer drains then exits

    def cancel(self):
        self.cancelled = True
        self._feeding = False

    def join_consumer(self, timeout=None):
        self._consumer.join(timeout)

    def ordered_texts(self):
        return [self.results[i] for i in range(self._submitted)
                if self.results.get(i)]

    def _feed(self):
        # Poll the shared frames buffer rather than doing DSP in the mic
        # callback, so the audio thread only ever appends and never blocks.
        while self._feeding:
            self._drain_frames()
            time.sleep(0.05)

    def _drain_frames(self):
        with frames_lock:
            new = frames[self._cursor:]
            self._cursor = len(frames)
        for block in new:
            for seg in self.segmenter.push(block):
                if seg.size:
                    self.submit(seg)

    def stop_feeding(self):
        """Called on stop: halt the feeder, pick up any last frames, then flush
        the Segmenter's tail so nothing the user said is left unqueued."""
        self._feeding = False
        if self._feeder is not None:
            self._feeder.join(timeout=2.0)
        self._drain_frames()
        tail = self.segmenter.finalize()
        if tail is not None and tail.size:
            self.submit(tail)

    def _consume(self):
        while True:
            item = self.seg_q.get()
            if item is None:
                break
            if self.cancelled:
                # Cancelled before this one was sent: nothing is in flight, so
                # the pipe stays in sync. Just drop it.
                continue
            idx, audio = item
            t0 = time.time()
            try:
                with self.lock:
                    self.conn.send(("recognize", audio, SAMPLE_RATE))
                    kind, payload = self.conn.recv()
            except Exception:
                log.exception("asr consumer roundtrip failed")
                continue
            if kind == "text":
                text = payload or ""
                log.info("segment %d: %.1fs audio -> %d chars in %.1fs",
                         idx, len(audio) / SAMPLE_RATE, len(text), time.time() - t0)
                if not self.cancelled:
                    self.results[idx] = text
            else:
                log.info("segment %d: ASR error: %s", idx, payload)


def acquire_single_instance():
    """Bind a fixed local port so only one copy of the app can run at a time."""
    global _lock_sock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 49731))
    except OSError:
        return False
    s.listen(1)
    _lock_sock = s
    return True


def start_recording():
    global stream, frames, session
    import sounddevice as sd

    if state["recording"] or state["busy"]:
        return
    with frames_lock:
        frames = []
    # Spin up the streaming session BEFORE the mic opens, so its feeder starts
    # transcribing speech in the background the moment audio arrives.
    session = AsrSession(worker_conn, worker_lock)
    session.start_feeder()

    def cb(indata, n, t, status):
        with frames_lock:
            frames.append(indata.copy())

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", callback=cb)
        stream.start()
    except Exception as e:
        log.exception("mic open failed")
        session.cancel()
        session.close()
        session = None
        ui_q.put(("status", f"mic error: {e}"))
        threading.Timer(1.8, lambda: ui_q.put(("hide",))).start()
        return
    state["recording"] = True
    log.info("recording started")
    ui_q.put(("show",))


def stop_and_transcribe():
    global stream, session

    if not state["recording"]:
        return
    # Stop-to-paste is measured from HERE (the stop hotkey) to just after the
    # paste lands — that is the latency the user actually feels.
    t_stop = time.time()
    state["recording"] = False
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    stream = None
    # Keep the whole take around: it is the safety fallback if streaming
    # produced no segments (very short or very quiet dictation).
    with frames_lock:
        data = np.concatenate(frames) if frames else np.zeros((0, 1), dtype="float32")
    audio = data.reshape(-1).astype("float32")
    log.info("recording stopped: %.1fs audio", audio.size / SAMPLE_RATE)
    if audio.size < SAMPLE_RATE * MIN_SECONDS:
        if session is not None:
            session.cancel()
            session.close()
            session = None
        ui_q.put(("hide",))
        return
    state["busy"] = True
    ui_q.put(("processing",))
    sess = session

    def work():
        global session
        try:
            # Flush the tail through the segmenter, close the queue, and wait for
            # the consumer to finish. Mid-recording segments are already done, so
            # this wait is only the (short) tail — that is the whole win.
            sess.stop_feeding()
            sess.close()
            sess.join_consumer()
            if sess.cancelled:
                return
            texts = sess.ordered_texts()
            if not texts:
                # Nothing got segmented: fall back to exactly today's behaviour
                # and transcribe the whole take once.
                with worker_lock:
                    worker_conn.send(("recognize", audio, SAMPLE_RATE))
                    kind, payload = worker_conn.recv()
                joined = (payload or "").strip() if kind == "text" else ""
                n_segments = 1
            else:
                joined = " ".join(texts)
                n_segments = len(texts)
            # clean_text runs ONCE on the joined text, so filler/dup collapsing
            # works across segment boundaries just like the single-shot path did.
            out = clean_text(joined) if CLEANUP else joined
            log.info("ASR result: %d chars -> %d after cleanup",
                     len(joined), len(out))
            if out:
                save_transcript(out)
                paste_text(out)
            log.info("stop-to-paste: %.1fs (%d segments, %d chars)",
                     time.time() - t_stop, n_segments, len(out))
        except Exception as e:
            log.exception("stop/transcribe failed")
            ui_q.put(("status", f"error: {e}"))
            time.sleep(1.8)
        finally:
            state["busy"] = False
            session = None
            ui_q.put(("hide",))

    threading.Thread(target=work, daemon=True).start()


def save_transcript(text):
    """Append the transcription to a local history file (transcripts.log,
    next to the app, never committed). Safety net for the day the paste
    lands in the wrong window or the clipboard gets overwritten."""
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "transcripts.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{text}\n\n")
    except Exception:
        log.exception("could not save transcript history")


def paste_text(text):
    import keyboard
    import pyperclip

    pyperclip.copy(text)
    time.sleep(0.06)
    keyboard.send("ctrl+v")
    # The transcription deliberately STAYS in the clipboard, so if the paste
    # went to the wrong window (or nowhere), Ctrl+V drops it anywhere you like.


def toggle():
    log.info("hotkey fired: recording=%s busy=%s model_ready=%s",
             state["recording"], state["busy"], state["model_ready"])
    if state["busy"]:
        return
    if not state["model_ready"]:
        ui_q.put(("status", "Model still loading…"))
        threading.Timer(1.5, lambda: ui_q.put(("hide",))).start()
        return
    if state["recording"]:
        stop_and_transcribe()
    else:
        start_recording()


def cancel():
    global stream, session
    if state["recording"]:
        state["recording"] = False
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        stream = None
        # Mark the session cancelled: its consumer keeps the pipe in sync but
        # discards results, nothing is pasted, no transcript is saved.
        if session is not None:
            session.cancel()
            session.close()
            session = None
        log.info("recording cancelled")
        ui_q.put(("hide",))


def quit_app():
    ui_q.put(("quit",))


def _lerp_hex(a, b, t):
    """Interpolate two #rrggbb colors; t in [0,1]."""
    ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    r = round(ar + (br - ar) * t)
    g = round(ag + (bg - ag) * t)
    bl = round(ab + (bb - ab) * t)
    return f"#{r:02x}{g:02x}{bl:02x}"


class Overlay:
    KEY = "#010203"          # transparent color key (drawn areas hide it)
    PILL = "#16181d"         # pill fill
    BORDER = "#2b2f3a"       # subtle border
    TEXT = "#e7e9ee"

    def __init__(self, root):
        import tkinter as tk

        self.tk = tk
        self.root = root
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-transparentcolor", self.KEY)
        except Exception:
            pass
        self.sw = root.winfo_screenwidth()
        self.sh = root.winfo_screenheight()
        self.canvas = tk.Canvas(root, bg=self.KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.mode = None
        self.dot_id = None
        self.text_id = None
        self.pulse = 0.0
        self.pulse_dir = 1
        root.withdraw()
        self._animate()
        root.after(60, self.poll)

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r,
               x2, y2, x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r,
               x1, y1 + r, x1, y1]
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    def _draw_pill(self, text, base_color):
        c = self.canvas
        c.delete("all")
        pad_x, dot_r = 18, 5
        font = ("Segoe UI", 11)
        tmp = c.create_text(0, 0, text=text, font=font, anchor="nw")
        bb = c.bbox(tmp)
        c.delete(tmp)
        tw = bb[2] - bb[0]
        w = pad_x + dot_r * 2 + 9 + tw + pad_x
        h = 34
        self.root.geometry(f"{int(w)}x{h}+{(self.sw - int(w)) // 2}+{self.sh - h - 96}")
        self._round_rect(1, 1, w - 1, h - 1, h // 2 - 1,
                         fill=self.PILL, outline=self.BORDER, width=1)
        cy = h // 2
        dx = pad_x + dot_r
        self.dot_id = c.create_oval(dx - dot_r, cy - dot_r, dx + dot_r, cy + dot_r,
                                    fill=base_color, outline="")
        self.text_id = c.create_text(dx + dot_r + 9, cy, text=text, anchor="w",
                                     fill=self.TEXT, font=font)
        self._base_color = base_color

    def _draw_mini(self, color):
        c = self.canvas
        c.delete("all")
        w = h = 20
        self.root.geometry(f"{w}x{h}+{self.sw - w - 22}+{self.sh - h - 58}")
        self.dot_id = c.create_oval(5, 5, 15, 15, fill=color, outline="")
        self.text_id = None
        self._base_color = color

    def _animate(self):
        # smooth breathing pulse on the dot while recording
        if self.mode == "recording" and self.dot_id is not None:
            self.pulse += 0.08 * self.pulse_dir
            if self.pulse >= 1:
                self.pulse, self.pulse_dir = 1, -1
            elif self.pulse <= 0:
                self.pulse, self.pulse_dir = 0, 1
            col = _lerp_hex("#5a2230", "#ff5b6a", self.pulse)
            try:
                self.canvas.itemconfigure(self.dot_id, fill=col)
            except Exception:
                pass
        self.root.after(45, self._animate)

    def show_pill(self, text, color, mode):
        self.mode = mode
        self._draw_pill(text, color)
        self.root.deiconify()
        self.root.lift()

    def show_mini(self, color):
        self.mode = "loading"
        self._draw_mini(color)
        self.root.deiconify()
        self.root.lift()

    def hide(self):
        self.mode = None
        self.root.withdraw()

    def poll(self):
        try:
            while True:
                msg = ui_q.get_nowait()
                kind = msg[0]
                if kind == "show":
                    self.show_pill("Transcribing", "#ff5b6a", "recording")
                elif kind == "processing":
                    self.show_pill("Transcribing", "#f5b23e", "processing")
                elif kind == "loading":
                    self.show_mini("#f5b23e")
                elif kind == "status":
                    self.show_pill(msg[1], "#f5b23e", "status")
                elif kind == "hide":
                    self.hide()
                elif kind == "quit":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(60, self.poll)


def main():
    global worker_conn
    import tkinter as tk
    import keyboard

    log.info("=== app starting (pid=%s) ===", os.getpid())
    if not acquire_single_instance():
        log.info("another instance already running - exiting")
        return

    # Spawn the model worker FIRST, before any UI, so its heavy load never
    # shares a process with the keyboard hook.
    parent_conn, child_conn = mp.Pipe()
    worker = mp.Process(target=asr_worker, args=(child_conn,),
                        name="asr-worker", daemon=True)
    worker.start()
    worker_conn = parent_conn
    log.info("worker spawned (pid=%s)", worker.pid)

    root = tk.Tk()
    root.title("Parakeet Dictation")
    ov = Overlay(root)
    ui_q.put(("loading",))

    def wait_ready():
        try:
            kind, payload = worker_conn.recv()
        except (EOFError, OSError):
            ui_q.put(("status", "worker died"))
            return
        if kind == "ready":
            state["model_ready"] = True
            log.info("worker reports ready")
            ui_q.put(("hide",))
        else:
            ui_q.put(("status", f"load error: {payload}"))

    threading.Thread(target=wait_ready, daemon=True).start()

    # Manual Ctrl+Win detection via a raw hook (keyboard's add_hotkey chokes
    # on modifier-only combos). Key NAMES are locale-dependent: on a German
    # keyboard Ctrl reports as "strg" and the Win key as "linke/rechte
    # windows", so we normalize by locale before matching. No suppression is
    # needed: only Win *alone* opens the Start menu, and here it's held with
    # Ctrl. Esc (cancel) and Ctrl+Alt+Q (quit) are handled here too, because
    # add_hotkey's English key names also miss on non-English layouts.
    keys_down = set()
    combo = {"active": False}

    def norm(name):
        n = (name or "").lower()
        if "windows" in n or n in ("win", "cmd", "meta", "super"):
            return "windows"
        if "strg" in n or "ctrl" in n or "control" in n or "steuerung" in n:
            return "ctrl"
        if n in ("alt", "linke alt", "left alt", "rechte alt", "right alt"):
            return "alt"
        if n in ("esc", "escape"):
            return "esc"
        return n

    def on_key(e):
        k = norm(e.name)
        if e.event_type == "down":
            keys_down.add(k)
            if k == "esc":
                if state["recording"]:
                    cancel()
                return
            if "ctrl" in keys_down and "alt" in keys_down and k == "q":
                quit_app()
                return
            if {"ctrl", "windows"} <= keys_down and not combo["active"]:
                combo["active"] = True
                toggle()
        else:
            keys_down.discard(k)
            if k in ("ctrl", "windows"):
                combo["active"] = False

    keyboard.hook(on_key)
    log.info("manual ctrl+win hook installed (locale-aware)")
    log.info("entering mainloop")
    root.mainloop()
    try:
        with worker_lock:
            worker_conn.send(("quit", None))
    except Exception:
        pass
    log.info("mainloop exited")


if __name__ == "__main__":
    mp.freeze_support()
    main()
