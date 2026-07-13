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
# Worker process: owns the model. Heavy CPU stays out of the hook process.
# ----------------------------------------------------------------------------
def asr_worker(conn):
    import numpy as np  # noqa: F401  (worker-side import)
    import onnx_asr

    try:
        t0 = time.time()
        try:
            model = onnx_asr.load_model(MODEL_NAME, quantization="int8")
            log.info("worker: loaded int8 model")
        except Exception:
            log.exception("worker: int8 load failed, falling back to fp32")
            model = onnx_asr.load_model(MODEL_NAME)
        log.info("worker: model loaded OK in %.1fs", time.time() - t0)
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
_lock_sock = None


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
    global stream, frames
    import sounddevice as sd

    if state["recording"] or state["busy"]:
        return
    with frames_lock:
        frames = []

    def cb(indata, n, t, status):
        with frames_lock:
            frames.append(indata.copy())

    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", callback=cb)
        stream.start()
    except Exception as e:
        log.exception("mic open failed")
        ui_q.put(("status", f"mic error: {e}"))
        threading.Timer(1.8, lambda: ui_q.put(("hide",))).start()
        return
    state["recording"] = True
    log.info("recording started")
    ui_q.put(("show",))


def stop_and_transcribe():
    global stream
    import numpy as np

    if not state["recording"]:
        return
    state["recording"] = False
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    stream = None
    with frames_lock:
        data = np.concatenate(frames) if frames else np.zeros((0, 1), dtype="float32")
    audio = data.reshape(-1).astype("float32")
    log.info("recording stopped: %.1fs audio", audio.size / SAMPLE_RATE)
    if audio.size < SAMPLE_RATE * MIN_SECONDS:
        ui_q.put(("hide",))
        return
    state["busy"] = True
    ui_q.put(("processing",))

    def work():
        try:
            with worker_lock:
                worker_conn.send(("recognize", audio, SAMPLE_RATE))
                kind, payload = worker_conn.recv()
            if kind == "text":
                out = clean_text(payload) if CLEANUP else payload
                log.info("ASR result: %d chars -> %d after cleanup",
                         len(payload), len(out))
                if out:
                    paste_text(out)
            else:
                ui_q.put(("status", f"ASR error: {payload}"))
                time.sleep(1.8)
        except Exception as e:
            log.exception("worker roundtrip failed")
            ui_q.put(("status", f"error: {e}"))
            time.sleep(1.8)
        finally:
            state["busy"] = False
            ui_q.put(("hide",))

    threading.Thread(target=work, daemon=True).start()


def paste_text(text):
    import keyboard
    import pyperclip

    try:
        old = pyperclip.paste()
    except Exception:
        old = ""
    pyperclip.copy(text)
    time.sleep(0.06)
    keyboard.send("ctrl+v")
    time.sleep(0.18)
    try:
        pyperclip.copy(old)
    except Exception:
        pass


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
    global stream
    if state["recording"]:
        state["recording"] = False
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        stream = None
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
