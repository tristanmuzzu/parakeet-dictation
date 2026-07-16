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
import sys
import threading
import time
import logging

import numpy as np

# Where we write our side files (dictation.log, transcripts.log). Next to the
# .exe when frozen by PyInstaller (sys.executable is the exe), next to this
# source file otherwise. Both routes therefore keep their logs beside the thing
# the user actually launched, and source users see no change at all.
BASE_DIR = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "dictation.log"),
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

# Command keywords detected at the end of transcribed speech. "send" (or common
# ASR mishearings like "sent") presses Enter after pasting; bare number words type
# that digit. Commands are stripped from the pasted text.
_SEND_RX = re.compile(r"\b(send|sent|sand|sendt|sends|enter)\b[,.]?$", re.IGNORECASE)
_NUM_RX = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine)\b[,.]?$", re.IGNORECASE)
_NUMBER_MAP = {
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

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
# Personal dictionary: teach Parakeet the names and words it keeps getting wrong.
#
# A plain-text dictionary.txt next to the app (BASE_DIR) holds the words you care
# about, one per line. The primary way to use it is to write ONLY the correct
# spelling:
#     FeWo direkt
#     Zeticle
# and Parakeet fixes anything it transcribes that SOUNDS close, even spellings
# you never listed ("fivo direct", "fiwo direkt", "fave direkt" all collapse to
# "FeWo direkt"). That sound-matching is why you never enumerate wrong spellings.
#
# For the rare case where phonetics can't bridge the gap, an explicit override
# is still supported with an arrow — wrong on the left, right on the right, with
# |-separated alternatives:
#     cloud code -> Claude Code
# Explicit overrides run FIRST, then the sound-matching. The file hot-reloads on
# its mtime, so edits land on your next dictation with no restart.
# ----------------------------------------------------------------------------
DICTIONARY_FILE = os.path.join(BASE_DIR, "dictionary.txt")

DICTIONARY_TEMPLATE = """\
# Teach Parakeet your words. Write the correct spelling, one per line:
#
#   FeWo direkt
#   Zeticle
#   NeoData
#
# Parakeet then fixes anything it hears that SOUNDS like one of these, even
# spellings you never wrote down. You never list the wrong versions.
#
# Only words of four letters or more are matched this way (short words are too
# easy to confuse for it to be safe).
#
# Advanced escape hatch: if a word comes out mangled beyond what sound-matching
# can catch, add an explicit correction with an arrow (wrong -> right). Several
# wrong spellings can share a line with |:
#
#   cloud code -> Claude Code
#
# Explicit corrections run first, then the sound-matching. Lines starting with #
# are ignored. Edits apply on your next dictation, no restart needed.
"""

# A "word" is a maximal run of letters (Unicode-aware: German umlauts count).
# Digits, punctuation and spaces separate words and are never matched into one.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Only entries whose every word is at least this long are sound-matched. Short
# words (a, on, code) collide with far too much to correct safely by sound.
MIN_FUZZY_LEN = 4
# A candidate window must clear BOTH bars to be replaced: its sound-keys must be
# this similar on average, and its raw letters must be at least loosely similar
# (the raw bar is a sanity guard against absurd sound-only collisions).
SOUND_SIM_MIN = 0.70
RAW_SIM_MIN = 0.45

_dict_rules = []       # explicit overrides: (compiled_regex, replacement), longest-left first
_dict_entries = []     # sound-matched entries (dicts), longest phrase first
_dict_mtime = None     # mtime the current data was built from (None = unloaded)
_dict_lock = threading.Lock()

_VOWELS = frozenset("aeiou")
_UMLAUTS = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "s"})
_SOUND_SINGLE = str.maketrans({"c": "k", "w": "v", "b": "v", "z": "s", "y": "i", "j": "i"})


def sound_key(word):
    """Reduce a word to a rough phonetic skeleton, so words that SOUND alike map
    to the same key. Cross German/English sound classes, then keep the first
    letter plus the consonant skeleton. Examples: fivo/fewo/fave -> 'fv',
    fible/fable -> 'fvl' (b and v/w are mishearing neighbors, seen live),
    direct/direkt -> 'drkt'. Returns '' for a word with no usable letters."""
    s = (word or "").lower().translate(_UMLAUTS)
    # Multi-letter sound classes first, longest/most-specific ahead of the rest
    # so "sch" wins over "ch" and "ch" is handled before a bare "c".
    for a, b in (("sch", "s"), ("ph", "f"), ("ch", "k"), ("ck", "k")):
        s = s.replace(a, b)
    s = s.translate(_SOUND_SINGLE)
    s = "".join(ch for ch in s if "a" <= ch <= "z")    # drop anything left over
    if not s:
        return ""
    skel = s[0] + "".join(ch for ch in s[1:] if ch not in _VOWELS)
    out = []                                            # collapse doubled letters
    for ch in skel:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _levenshtein(a, b):
    """Classic edit distance (single-row DP). Small inputs — words, not text."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _sim(a, b):
    """Similarity in [0,1]: 1 - editdistance/longer. Two empties are identical."""
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / m


def ensure_dictionary_file():
    """Create dictionary.txt with the commented template on first run, so the
    user (exe or source) always has a file to edit and nothing to set up."""
    try:
        if not os.path.exists(DICTIONARY_FILE):
            with open(DICTIONARY_FILE, "w", encoding="utf-8") as f:
                f.write(DICTIONARY_TEMPLATE)
            log.info("dictionary: created template at %s", DICTIONARY_FILE)
    except Exception:
        log.exception("dictionary: could not create template file")


def _parse_dictionary(text):
    """Parse dictionary text into explicit overrides and sound-matched entries.
    A line with '->' is an explicit override (left|alts -> right); any other
    non-comment line is a correct-word entry to sound-match against. Malformed
    lines are skipped with one log line — a single bad line never aborts the
    rest. Returns (rules, entries, n_words, n_rules)."""
    flat = []          # explicit: (left_alternative, replacement)
    entries = []       # sound-matched word entries
    n_words = 0
    n_rules = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            left, _, right = line.partition("->")
            right = right.strip()
            alts = [a.strip() for a in left.split("|") if a.strip()]
            if not right or not alts:
                log.info("dictionary: skipping malformed override line: %r", line)
                continue
            for a in alts:
                flat.append((a, right))
            n_rules += 1
        else:
            words = _WORD_RE.findall(line)
            if not words:
                log.info("dictionary: skipping unusable line: %r", line)
                continue
            n_words += 1
            # Sound-matching needs every word to clear the length bar; shorter
            # entries are kept in the count but reachable only via an override.
            if all(len(w) >= MIN_FUZZY_LEN for w in words):
                entries.append({
                    "text": line,
                    "nwords": len(words),
                    "keys": [sound_key(w) for w in words],
                    "raw": [w.lower() for w in words],
                })
    # Longest override left-side first, so "cloud code" wins over a bare "cloud".
    flat.sort(key=lambda pr: len(pr[0]), reverse=True)
    rules = []
    for left, right in flat:
        try:
            rx = re.compile(r"\b" + re.escape(left) + r"\b", re.IGNORECASE)
        except Exception:
            # A bad override is skipped, logged once here at load, never at apply.
            log.exception("dictionary: skipping unbuildable override %r", left)
            continue
        rules.append((rx, right))
    # Longest phrase first, so a 2-word entry claims its words before a 1-word one.
    entries.sort(key=lambda e: (e["nwords"], len(e["text"])), reverse=True)
    return rules, entries, n_words, n_rules


def _maybe_reload():
    """(Re)load when dictionary.txt first appears or its mtime changes. Not
    thread-safe on its own — callers hold _dict_lock."""
    global _dict_rules, _dict_entries, _dict_mtime
    try:
        mtime = os.path.getmtime(DICTIONARY_FILE)
    except OSError:
        return                              # no file: keep whatever we had
    if mtime == _dict_mtime:
        return
    try:
        with open(DICTIONARY_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        log.exception("dictionary: could not read file")
        return
    _dict_rules, _dict_entries, n_words, n_rules = _parse_dictionary(text)
    _dict_mtime = mtime
    log.info("dictionary: %d words, %d explicit rules loaded", n_words, n_rules)


def _apply_explicit(text, rules):
    """Whole-word, case-insensitive override pass. Right side stays verbatim."""
    out = text
    replaced = 0
    for rx, right in rules:
        # A function replacement keeps the right side literal (no \1 / \g
        # backreference surprises) and counts hits in one pass.
        out, n = rx.subn(lambda m, r=right: r, out)
        replaced += n
    return out, replaced


def _window_matches(win_words, entry):
    """True when this same-length run of transcribed words sounds like the entry
    (and is not an absurd raw-letter mismatch). See SOUND_SIM_MIN / RAW_SIM_MIN."""
    win_keys = [sound_key(w) for w in win_words]
    ent_keys = entry["keys"]
    if not win_keys[0] or not ent_keys[0]:
        return False
    if win_keys[0][0] != ent_keys[0][0]:            # first sound-mapped letter
        return False
    sound = [_sim(a, b) for a, b in zip(win_keys, ent_keys)]
    if sum(sound) / len(sound) < SOUND_SIM_MIN:
        return False
    raw = [_sim(w.lower(), r) for w, r in zip(win_words, entry["raw"])]
    if sum(raw) / len(raw) < RAW_SIM_MIN:
        return False
    return True


def _apply_fuzzy(text, entries):
    """Replace any run of words that sounds like a dictionary entry with the
    entry's exact spelling. Longest entries first; a run consumed by one
    replacement is not matched again."""
    if not entries:
        return text, 0
    toks = [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if not toks:
        return text, 0
    consumed = [False] * len(toks)
    repls = []                                        # (start_char, end_char, text)
    for e in entries:
        n = e["nwords"]
        i = 0
        while i + n <= len(toks):
            if any(consumed[i:i + n]):
                i += 1
                continue
            window = toks[i:i + n]
            if _window_matches([w[0] for w in window], e):
                repls.append((window[0][1], window[-1][2], e["text"]))
                for j in range(i, i + n):
                    consumed[j] = True
                i += n
            else:
                i += 1
    if not repls:
        return text, 0
    repls.sort()
    parts = []
    last = 0
    for start, end, rep in repls:
        parts.append(text[last:start])
        parts.append(rep)
        last = end
    parts.append(text[last:])
    return "".join(parts), len(repls)


def apply_dictionary(text):
    """Apply personal corrections to a finished transcription: explicit
    overrides first, then sound-matching. Never throws — a broken dictionary
    must never break a dictation."""
    if not text:
        return text
    try:
        with _dict_lock:
            _maybe_reload()
            rules = _dict_rules
            entries = _dict_entries
        out, n_explicit = _apply_explicit(text, rules)
        out, n_fuzzy = _apply_fuzzy(out, entries)
        total = n_explicit + n_fuzzy
        if total:
            log.info("dictionary: %d corrections", total)
        return out
    except Exception:
        log.exception("dictionary: apply failed (leaving text unchanged)")
        return text


# ----------------------------------------------------------------------------
# Segmenter: pure, deterministic speech chopping (no I/O, no threads).
#
# It eats raw audio blocks and decides where to cut the take into segments the
# worker can transcribe independently. Kept side-effect free so it is trivially
# unit-testable with synthetic arrays — the streaming plumbing that feeds it and
# ships the pieces to the worker lives entirely in AsrSession.
# ----------------------------------------------------------------------------
class Segmenter:
    def __init__(self, sr=SAMPLE_RATE,
                 min_seg=MIN_SEG, max_seg=MAX_SEG,
                 silence_cut=SILENCE_CUT, keep_sil=KEEP_SIL):
        self.sr = sr
        self.min_seg = min_seg
        self.max_seg = max_seg
        self.silence_cut = silence_cut
        self.keep_sil = keep_sil
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
        if (self._speech_samples >= self.min_seg * self.sr
                and self._trailing_sil >= self.silence_cut * self.sr):
            return self._cut(trim=True)
        # Cut B: segment got too long with no pause — cut hard to bound latency.
        if self._seg_len >= self.max_seg * self.sr:
            return self._cut(trim=False)
        return None

    def _cut(self, trim):
        concat = np.concatenate(self._seg)
        if trim:
            keep = int(self.keep_sil * self.sr)
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
state = {"recording": False, "busy": False, "model_ready": False, "continuous": False}
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
        cm = state.get("continuous", False)
        self.segmenter = Segmenter(
            min_seg=0.3 if cm else MIN_SEG,
            max_seg=3.0 if cm else MAX_SEG,
            silence_cut=0.35 if cm else SILENCE_CUT,
            keep_sil=0.1 if cm else KEEP_SIL,
        )
        self.seg_q: "queue.Queue" = queue.Queue()
        self.results = {}          # index -> transcribed text
        self.cancelled = False
        self._cmd_action = None    # set by feeder when a command is detected inline
        self._cmd_text = None
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
            self._check_inline_cmd()
            time.sleep(0.05)

    def _check_inline_cmd(self):
        if not state.get("continuous"):
            return
        texts = [self.results.get(i)
                 for i in range(self._submitted) if self.results.get(i)]
        if not texts:
            return
        joined = " ".join(t for t in texts if t).strip()
        test_text, cmd = detect_and_strip_command(joined)
        if cmd:
            log.info("inline command detected: %s (text: %r)", cmd, test_text)
            self._cmd_action = cmd
            self._cmd_text = test_text
            self._feeding = False
            ui_q.put(("inline_cmd",))
        elif len(joined) > 2:
            log.info("inline check (no cmd): %r", joined[-60:])

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
    ui_q.put(("listen",) if state["continuous"] else ("show",))


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
        if state["continuous"]:
            ui_q.put(("restart",))
        else:
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
            # Personal dictionary fixes the model's known misfires (names,
            # brands) on the joined text BEFORE clean_text. Both the segmented
            # and the whole-take fallback path funnel through `joined`, so this
            # one call covers them both. The raw line makes "did the model even
            # transcribe that word or did a correction eat it" answerable from
            # the log alone (local file, same privacy as transcripts.log).
            log.info("raw ASR text: %r", joined)
            joined = apply_dictionary(joined)
            joined, cmd = detect_and_strip_command(joined)
            if cmd:
                log.info("command detected: %s", cmd)
            # clean_text runs ONCE on the joined text, so filler/dup collapsing
            # works across segment boundaries just like the single-shot path did.
            out = clean_text(joined) if CLEANUP else joined
            log.info("ASR result: %d chars -> %d after cleanup",
                     len(joined), len(out))
            if out:
                save_transcript(out)
                paste_text(out)
            if cmd:
                time.sleep(0.05)
                import keyboard
                keyboard.send(cmd)
            log.info("stop-to-paste: %.1fs (%d segments, %d chars)",
                     time.time() - t_stop, n_segments, len(out))
        except Exception as e:
            log.exception("stop/transcribe failed")
            ui_q.put(("status", f"error: {e}"))
            time.sleep(1.8)
        finally:
            state["busy"] = False
            session = None
            if state["continuous"]:
                ui_q.put(("restart",))
            else:
                ui_q.put(("hide",))

    threading.Thread(target=work, daemon=True).start()


def save_transcript(text):
    """Append the transcription to a local history file (transcripts.log,
    next to the app, never committed). Safety net for the day the paste
    lands in the wrong window or the clipboard gets overwritten."""
    try:
        path = os.path.join(BASE_DIR, "transcripts.log")
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


def toggle_continuous():
    global session
    if state["continuous"]:
        state["continuous"] = False
        if state["recording"]:
            stop_and_transcribe()
        log.info("continuous mode off")
    else:
        state["continuous"] = True
        log.info("continuous mode on")
        if not state["recording"] and not state["busy"] and state["model_ready"]:
            start_recording()


def detect_and_strip_command(text):
    m = _NUM_RX.search(text)
    if m:
        word = m.group(1).lower()
        return text[:m.start()].rstrip(), _NUMBER_MAP[word]
    m = _SEND_RX.search(text)
    if m:
        return text[:m.start()].rstrip(), "enter"
    return text, None


def handle_inline_command():
    global stream, session
    if not state["recording"] or session is None:
        return
    sess = session
    state["recording"] = False
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    stream = None
    state["busy"] = True
    ui_q.put(("processing",))

    def work():
        global session
        try:
            sess.stop_feeding()
            sess.close()
            sess.join_consumer()
            if sess.cancelled:
                return
            joined = " ".join(sess.ordered_texts()).strip()
            if not joined:
                return
            log.info("inline raw ASR: %r", joined)
            joined = apply_dictionary(joined)
            joined, cmd = detect_and_strip_command(joined)
            if cmd:
                log.info("inline command: %s (text: %r)", cmd, joined)
            out = clean_text(joined) if CLEANUP else joined
            if out:
                save_transcript(out)
                paste_text(out)
            if cmd:
                time.sleep(0.05)
                import keyboard
                keyboard.send(cmd)
        except Exception:
            log.exception("inline command handler failed")
        finally:
            state["busy"] = False
            session = None
            if state["continuous"]:
                ui_q.put(("restart",))
            else:
                ui_q.put(("hide",))

    threading.Thread(target=work, daemon=True).start()


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
        if state["continuous"]:
            ui_q.put(("restart",))
        else:
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
                elif kind == "listen":
                    self.show_pill("Listening", "#ff5b6a", "recording")
                elif kind == "processing":
                    self.show_pill("Transcribing", "#f5b23e", "processing")
                elif kind == "loading":
                    self.show_mini("#f5b23e")
                elif kind == "status":
                    self.show_pill(msg[1], "#f5b23e", "status")
                elif kind == "hide":
                    self.hide()
                elif kind == "restart":
                    self.hide()
                    self.root.after(400, start_recording)
                elif kind == "inline_cmd":
                    handle_inline_command()
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

    # Make sure the editable dictionary.txt exists (first run drops the commented
    # template next to the exe / in the repo root) and load it now, so the rule
    # count is logged at startup and corrections are ready before the first
    # dictation. Later edits hot-reload on their own, no restart needed.
    ensure_dictionary_file()
    with _dict_lock:
        _maybe_reload()

    # First launch on a fresh machine has to pull ~460 MB of model weights from
    # Hugging Face before anything works. Source users watch that happen in the
    # console; exe users have no console, so tell them what the wait is via the
    # status pill. If the snapshot is already cached we skip the message and keep
    # the quiet amber loading dot.
    hf_snapshot = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--istupakov--parakeet-tdt-0.6b-v3-onnx")
    model_cached = os.path.isdir(hf_snapshot)
    log.info("model cache present: %s (%s)", model_cached, hf_snapshot)

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
    if model_cached:
        ui_q.put(("loading",))
    else:
        # The normal ("hide",) on worker-ready clears this once the download and
        # load finish.
        ui_q.put(("status", "Downloading speech model (one time, ~460 MB)"))

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
    _last_hotkey = 0.0

    def norm(name):
        n = (name or "").lower()
        if "windows" in n or n in ("win", "cmd", "meta", "super"):
            return "windows"
        if "strg" in n or "ctrl" in n or "control" in n or "steuerung" in n:
            return "ctrl"
        if n in ("alt", "linke alt", "left alt", "rechte alt", "right alt"):
            return "alt"
        if "shift" in n:
            return "shift"
        if n in ("esc", "escape"):
            return "esc"
        return n

    def on_key(e):
        nonlocal _last_hotkey
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
            now = time.time()
            if now - _last_hotkey < 0.25:
                return
            if {"ctrl", "windows"} <= keys_down:
                log.info("hotkey raw: keys_down=%s  shift_in=%s",
                         sorted(keys_down), "shift" in keys_down)
                if "shift" in keys_down:
                    _last_hotkey = now
                    toggle_continuous()
                else:
                    _last_hotkey = now
                    toggle()
        else:
            keys_down.discard(k)

    keyboard.hook(on_key)
    log.info("manual hotkey hooks installed (ctrl+win, ctrl+shift+win, locale-aware)")
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
