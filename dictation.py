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
    recognize() requests over a multiprocessing Pipe. The worker is
    disposable: after PARAKEET_IDLE_UNLOAD_SEC without dictation it is shut
    down so its ~800 MB goes back to the OS, and the next dictation respawns
    it transparently (see spawn_worker / idle_monitor).

Text stays on the clipboard for recovery. Paste is dispatched once; the capsule
hides immediately afterward, without inspecting the destination accessibility tree.
"""

# GNOME launches this client for each shortcut. Exit before importing NumPy,
# initializing logging, or reading the application store/source files.
import sys
if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--send":
    import linux_backend
    sys.exit(linux_backend.send_command(sys.argv[2]))

import math
import multiprocessing as mp
import os
import queue
import re
import socket
import sys
import threading
import time
import logging
import uuid
import json
import hashlib
from pathlib import Path
from app_store import AppStore, SPEECH_MODELS
from prompt_editor import LiveEdit

import numpy as np

import linux_backend
import screen_metrics

# On Linux the global hotkey comes from GNOME (see install-gnome-hotkeys.sh) and
# text is inserted via wl-copy + ydotool; the `keyboard` package is neither
# importable-without-root nor functional under Wayland. See linux_backend.py.
IS_LINUX = linux_backend.is_linux()

# Where we write our side files (dictation.log, transcripts.log). Next to the
# .exe when frozen by PyInstaller (sys.executable is the exe), next to this
# source file otherwise. Both routes therefore keep their logs beside the thing
# the user actually launched, and source users see no change at all.
BASE_DIR = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.abspath(__file__)))
APP_VERSION = "0.2.0-dev"
try:
    BUILD_ID = hashlib.sha256(b"".join((Path(BASE_DIR)/name).read_bytes()
        for name in ("dictation.py","linux_backend.py","native_ui.py","app_store.py","prompt_editor.py","speech_models.py","qwen_speech.py","transcribe_speech.py"))).hexdigest()[:12]
except OSError:
    BUILD_ID = APP_VERSION

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "dictation.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
)
log = logging.getLogger("dictation")

SAMPLE_RATE = 16000
MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
MIN_SECONDS = 0.3  # ignore taps shorter than this
# Hotkeys (Ctrl+Win toggle, Ctrl+Shift+Win continuous mode, Esc cancel,
# Ctrl+Alt+Q quit) are implemented via a raw keyboard hook in main() — see
# on_key() and the comment there for why add_hotkey() string combos are NOT used.
CLEANUP = False    # preserve wording by default
ASR_TIMEOUT = 90.0

# Keep the model warm between dictations. Idle unloading is opt-in because the
# reload/warmup delay lands directly on the next take's stop-to-paste latency.
try:
    IDLE_UNLOAD_SEC = float(os.environ.get("PARAKEET_IDLE_UNLOAD_SEC", "0"))
except ValueError:
    IDLE_UNLOAD_SEC = 0.0
# How long a queued segment will wait for a (re)loading worker before being
# dropped. Generous: a reload is ~4 s, so hitting this means the worker is
# actually broken, and the drop is logged as an error.
WORKER_READY_WAIT = 120.0

# Which microphone to record from. Empty means "whatever the system calls the
# default input", which is the right default but NOT always a working device:
# this machine exposes two internal capture devices and the one the audio server
# prefers on priority ("Stereo Microphone", priority 200) is deaf, while the one
# that actually hears the room ("Digital Microphone", priority 100) loses the
# tie. Every profile change re-runs that tie-break, so the default can silently
# become a dead mic and every dictation comes back empty. Set this (env var, or
# edit here) to pin recording to a device by name substring or index and take
# the audio server's guess out of the loop. PortAudio reports the audio server's
# node names, so on this machine the working mic is PARAKEET_INPUT_DEVICE=Mic1
# ("alsa_input.pci-0000_04_00.6.HiFi__Mic1__source"); Mic2 is the deaf one.
INPUT_DEVICE = os.environ.get("PARAKEET_INPUT_DEVICE", "").strip()

# Command keywords detected at the end of transcribed speech, ONLY while
# continuous mode is on: a one-shot dictation legitimately ends in "sent" or
# "one" all the time, and stripping that word (plus firing a phantom Enter into
# the focused window) would corrupt normal dictations. "send" (or common ASR
# mishearings like "sent") presses Enter after pasting; bare number words type
# that digit. Commands are stripped from the pasted text.
_SEND_RX = re.compile(r"\b(send|enter)\b[\s,.!?]*$", re.IGNORECASE)
_NUM_RX = re.compile(r"^(zero|one|two|three|four|five|six|seven|eight|nine)[\s,.!?]*$", re.IGNORECASE)
_NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
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
BOUNDARY_WINDOW = 0.0    # original chunk timing; brief-pause experiment disabled
BOUNDARY_SILENCE = 0.18  # enough quiet to avoid splitting a spoken word
CALIB_SEC = 0.3          # first slice used to estimate the room's noise floor
ABS_FLOOR = 0.004        # RMS below this is silence no matter what calibration says
FLOOR_MULT = 3.0         # speech must exceed this multiple of the learned floor
EMA_ALPHA = 0.05         # how fast the noise floor tracks (slow, non-speech only)

# Standalone filler tokens (EN + DE), removed only as whole words. Conservative
# on purpose: does NOT touch "like"/"you know" etc. so meaning is never damaged.
_FILLER = re.compile(
    r"\b(u+h+|uh+m+|e+r+m+|ä+h+m*|ehm+)\b[,]?",
    re.IGNORECASE,
)
# Prompts often quote exact output or mention paths and identifiers. Leave those
# spans untouched, even when they contain a word that sounds like hesitation.
_CLEANUP_LITERAL = re.compile(
    r"```[\s\S]*?(?:```|$)|`[^`\n]*(?:`|$)|\"(?:\\.|[^\"\\])*(?:\"|$)"
    r"|(?<!\w)'(?:\\.|[^'\\\n])*'(?!\w)"
    r"|“[^”]*(?:”|$)|‘[^’]*(?:’|$)"
    r"|(?:https?://|www\.)[^\s]+|(?:[$@]|--?)[\w-]+|[/\\][\w./\\-]+"
    r"|\b[\w]+(?:[./\\:_-][\w]+)+\b"
)


def clean_text(t):
    """Remove hesitations in prose, preserving literal prompt content (no LLM)."""
    if not t:
        return t

    def prose(text):
        # Preserve uppercase abbreviations and meaningful repetitions. German
        # "um", "like", "you know", and spoken corrections also stay intact.
        text = _FILLER.sub(lambda m: m[0] if m[0].rstrip(",").isupper() else " ", text)
        text = re.sub(r"[^\S\r\n]+([,.!?;:])", r"\1", text)
        return re.sub(r"[^\S\r\n]{2,}", " ", text)

    parts = []
    end = 0
    for literal in _CLEANUP_LITERAL.finditer(t):
        parts.extend((prose(t[end:literal.start()]), literal[0]))
        end = literal.end()
    parts.append(prose(t[end:]))
    # The recognizer supplies capitalization. Do not capitalize a leading
    # identifier (e.g. npm) or collapse intentional newlines here.
    return re.sub(r"^[ ,.;:]+", "", "".join(parts)).strip()


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
            # A fuzzy phrase may span whitespace, never numbers, punctuation,
            # sentence boundaries or code. Otherwise "stable 42 release" loses 42.
            if any(not re.fullmatch(r"[ \t]+", text[a[2]:b[1]])
                   for a, b in zip(window, window[1:])):
                i += 1
                continue
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
# DCBlocker: strips the DC bias this laptop's digital microphone puts on every
# sample.
#
# Measured 2026-08-09 on Mic1 at 16 kHz mono, quiet room: mean +0.220 against an
# RMS of 0.221 — about 87% of the captured "signal" was DC, with 27.7% of the
# spectral energy below 20 Hz.
#
# This matters far more than the lost headroom. The Segmenter gates speech on
# frame RMS against ABS_FLOOR (0.004), and a 0.220 offset is fifty-five times
# that floor, so EVERY frame reads as speech and the calibration in _step
# converges on the bias instead of the room. A dictation that starts fine and
# then stops cutting segments correctly partway through is this bug.
#
# y[n] = x[n] - x[n-1] + R*y[n-1], with R placing the -3 dB corner near 13 Hz at
# 16 kHz — far below the ~85 Hz bottom of a male fundamental, so it removes bias
# and rumble and touches nothing that carries speech. Filter state is carried
# across blocks so there is no step at a block boundary.
# ----------------------------------------------------------------------------
class DCBlocker:
    def __init__(self, sr=SAMPLE_RATE, corner_hz=13.0):
        self.r = 1.0 - (2.0 * math.pi * corner_hz / float(sr))
        self._x1 = 0.0
        self._y1 = 0.0

    def process(self, block):
        x = np.asarray(block, dtype="float64").reshape(-1)
        if x.size == 0:
            return np.zeros(0, dtype="float32")
        y = np.empty_like(x)
        r, x1, y1 = self.r, self._x1, self._y1
        for i in range(x.size):
            xi = x[i]
            y1 = xi - x1 + r * y1
            x1 = xi
            y[i] = y1
        self._x1, self._y1 = x1, y1
        return y.astype("float32")


def remove_dc(audio):
    """Whole-take DC removal for the stop-to-paste path.

    Deliberately NOT DCBlocker.process here. That is a Python-level IIR loop and
    costs ~211 ms on a 20 s take (measured), which would land directly on the
    stop-to-paste latency the user actually feels. The bias on this microphone is
    a constant, not a drift, so subtracting the mean removes it exactly at
    ~0.4 ms for the same take. The streaming path still uses the real one-pole,
    where the cost is per-block, off the latency path, and drift-safe.
    """
    a = np.asarray(audio, dtype="float32").reshape(-1)
    if a.size == 0:
        return a
    return (a - a.mean()).astype("float32")


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
                 silence_cut=SILENCE_CUT, keep_sil=KEEP_SIL,
                 boundary_window=BOUNDARY_WINDOW):
        self.sr = sr
        self.min_seg = min_seg
        self.max_seg = max_seg
        self.silence_cut = silence_cut
        self.keep_sil = keep_sil
        self.boundary_window = boundary_window
        self.frame_len = max(1, int(sr * FRAME_MS / 1000))
        self._pending = np.zeros(0, dtype="float32")   # samples < one full frame
        # Strip the microphone's DC bias before anything measures RMS. Without
        # this the noise-floor calibration below converges on the bias.
        self._dc = DCBlocker(sr)
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
        block = self._dc.process(block)
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
        speech_ok = self._speech_samples > 0
        # Safety net: if calibration got polluted (the user spoke with no leading
        # silence) speech frames can be misread as silence, so also emit when the
        # tail simply carries sound above the absolute floor for long enough.
        rms = float(np.sqrt(np.mean(concat * concat))) if concat.size else 0.0
        loud_ok = rms > ABS_FLOOR
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
                # Immediate speech must not become an arbitrarily loud floor.
                # Keep the existing noise tracking but cap startup calibration.
                self._floor = min(0.02, self._calib_sum / max(1, self._calib_count))
        thresh = max(self._floor * FLOOR_MULT, ABS_FLOOR)
        if rms > thresh:
            self._speech_samples += frame.size
            self._trailing_sil = 0
        else:
            self._trailing_sil += frame.size
            if self._calib_remaining <= 0:   # track the floor on true silence only
                self._floor = min(0.02, (1 - EMA_ALPHA) * self._floor + EMA_ALPHA * rms)
        if not allow_emit:
            return None
        # Cut A: enough speech banked and a real pause has landed. Trim the pause.
        if (self._speech_samples >= self.min_seg * self.sr
                and self._trailing_sil >= self.silence_cut * self.sr):
            return self._cut(trim=True)
        # Cut B: approaching the deadline, use a brief inter-word pause rather
        # than waiting for a sentence pause and then cutting through a word.
        if (self.boundary_window > 0
                and self._seg_len >= max(self.min_seg, self.max_seg-self.boundary_window) * self.sr
                and self._speech_samples > 0
                and self._trailing_sil >= BOUNDARY_SILENCE * self.sr):
            return self._cut(trim=True)
        # Cut C: nonstop speech still has a bounded maximum chunk size.
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
def asr_worker(conn, speech_model="parakeet"):
    import numpy as np  # noqa: F401  (worker-side import)

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
        from speech_models import load_speech_model
        model = load_speech_model(speech_model, BASE_DIR, is_linux=IS_LINUX)
        log.info("worker: selected %s", SPEECH_MODELS[speech_model])
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
    if hasattr(model, "close"):
        model.close()


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
worker_proc = None
# Set while a loaded worker is ready to recognize. Consumers (the session
# consumer and the whole-take fallback) wait on this before touching the pipe,
# which is what lets a recording START while a respawned worker is still
# loading: segments queue up behind the event and flow the moment it flips.
# Starts SET so code that drives AsrSession without ever spawning a worker
# (the plumbing tests, smoke_pipeline) is not gated; spawn_worker clears it.
worker_ready_evt = threading.Event()
worker_ready_evt.set()
# Serialises spawn/unload decisions. Ordering: always taken OUTSIDE
# worker_lock, never while holding it.
_worker_spawn_lock = threading.Lock()
_ever_ready = False     # a model has finished loading at least once this run
_last_activity = time.time()
session = None          # the live AsrSession while recording / draining the tail
app_store = None
prompt_editor = None
overlay = None
last_retry = None
active_take_id = None
_lock_sock = None


def touch_activity():
    global _last_activity
    _last_activity = time.time()


def model_available():
    """Can a dictation start right now? True when the model is loaded, and
    also when it WAS loaded and the idle timer took it down — in that case
    start_recording() respawns the worker and the dictation just pays the
    reload in its paste latency. Only the first-ever startup load (which may
    include the ~460 MB model download) still refuses the hotkey."""
    return state["model_ready"] or _ever_ready


def spawn_worker(respawn=False):
    """Start the ASR worker process plus the thread that consumes its one-off
    ready/load_error handshake. That thread is the ONLY reader of the pipe
    until worker_ready_evt is set, so the handshake can never be mistaken for
    a recognize() reply by a session consumer (they all wait on the event).

    respawn=True (reload after an idle unload) keeps hands off the UI: a
    recording pill may already be on screen, and hiding it on ready — right
    for the startup loading dot — would wipe a live recording indicator."""
    global worker_conn, worker_proc, worker_lock
    worker_lock = threading.Lock()
    state["model_ready"] = False
    worker_ready_evt.clear()
    parent_conn, child_conn = mp.Pipe()
    selected = app_store.settings["speech_model"] if app_store else "parakeet"
    proc = mp.Process(target=asr_worker, args=(child_conn, selected),
                      name="asr-worker", daemon=True)
    proc.start()
    child_conn.close()
    worker_conn = parent_conn
    worker_proc = proc
    log.info("worker %s (pid=%s)", "respawned" if respawn else "spawned",
             proc.pid)

    def wait_ready():
        global _ever_ready
        try:
            kind, payload = parent_conn.recv()
        except (EOFError, OSError):
            ui_q.put(("status", "worker died"))
            return
        if parent_conn is not worker_conn:
            return
        if kind == "ready":
            state["model_ready"] = True
            _ever_ready = True
            touch_activity()
            worker_ready_evt.set()
            log.info("worker reports ready%s",
                     " (reloaded after idle unload)" if respawn else "")
            if not respawn:
                ui_q.put(("ready",))
        else:
            ui_q.put(("status", f"load error: {payload}"))

    threading.Thread(target=wait_ready, name="worker-ready-wait",
                     daemon=True).start()


def ensure_worker():
    """Bring the worker back if the idle timer (or a crash) took it down.
    Cheap when it is already running. Returns as soon as the process is
    spawned — the model load happens in the child, and consumers hold on
    worker_ready_evt until it reports ready."""
    with _worker_spawn_lock:
        if worker_proc is not None and worker_proc.is_alive():
            return
        spawn_worker(respawn=True)


def unload_idle_worker(idle):
    """Shut the worker down after `idle` seconds without dictation. Holds the
    spawn lock for the whole teardown so ensure_worker() can only observe
    'alive' or 'gone', never a half-dead worker; re-checks activity under the
    lock so a dictation that started since the monitor's unlocked peek wins."""
    global worker_conn, worker_proc
    with _worker_spawn_lock:
        if IDLE_UNLOAD_SEC <= 0 or time.time() - _last_activity < IDLE_UNLOAD_SEC:
            return
        if (state["recording"] or state["busy"] or state["continuous"]
                or session is not None):
            return
        if worker_proc is None or not worker_proc.is_alive():
            return
        state["model_ready"] = False
        worker_ready_evt.clear()
        pid = worker_proc.pid
        with worker_lock:
            try:
                worker_conn.send(("quit", None))
            except Exception:
                log.exception("worker: quit send failed during idle unload")
            worker_proc.join(timeout=5.0)
            if worker_proc.is_alive():
                log.warning("worker: no exit on quit; terminating (pid=%s)", pid)
                worker_proc.terminate()
                worker_proc.join(timeout=3.0)
            try:
                worker_conn.close()
            except Exception:
                pass
        worker_conn = None
        worker_proc = None
        log.info("worker: unloaded after %.0fs idle (pid=%s); "
                 "model reloads on next dictation", idle, pid)


def idle_monitor():
    """Poll for the idle-unload condition. Continuous mode never unloads (the
    mic re-arms constantly and the user has declared an ongoing session)."""
    poll = max(5.0, min(30.0, IDLE_UNLOAD_SEC / 10.0))
    while True:
        time.sleep(poll)
        if not state["model_ready"]:
            continue
        if (state["recording"] or state["busy"] or state["continuous"]
                or session is not None):
            continue
        idle = time.time() - _last_activity
        if idle >= IDLE_UNLOAD_SEC:
            unload_idle_worker(idle)


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
def join_tail_context(previous, revised):
    """Replace only a uniquely matching ending, otherwise request full context.

    The new decode starts with four seconds already recognized. Require three
    exact words as an anchor near the old ending; never guess through a mismatch
    or an ambiguous repetition. Text after the anchor comes from the new decode.
    """
    old = list(re.finditer(r"\w+", previous))
    new = list(re.finditer(r"\w+", revised))
    if len(old) < 3 or len(new) < 3:
        return None
    old_words = [m.group().casefold() for m in old]
    new_words = [m.group().casefold() for m in new]
    for start in range(min(3, len(new)-2)):
        anchor = new_words[start:start+3]
        hits = [i for i in range(max(0,len(old)-16),len(old)-2)
                if old_words[i:i+3] == anchor]
        if len(hits) == 1:
            return previous[:old[hits[0]].start()] + revised[new[start].start():]
    return None


def join_chunk_context(previous, revised):
    """Split a contextual decode without replacing any previously heard words.

    Match a unique spoken suffix, including every word through the old ending.
    Only its final punctuation may change. Ambiguous repetitions, changed
    words, quoted endings and missing continuations use independent decoding.
    """
    old = list(re.finditer(r"\w+", previous))
    new = list(re.finditer(r"\w+", revised))
    if len(old) < 3 or len(new) < 4:
        return None
    if not re.fullmatch(r"[\s,.!?;:]*", previous[old[-1].end():]):
        return None
    ow = [m.group().casefold() for m in old]
    nw = [m.group().casefold() for m in new]
    for start in range(min(3, len(new)-3)):
        anchor = nw[start:start+3]
        hits = [i for i in range(len(ow)-2) if ow[i:i+3] == anchor]
        if len(hits) != 1 or hits[0] < max(0, len(ow)-16):
            continue
        # A repeat later in the new audio is new speech, not an ambiguous
        # look-behind match. Only competing matches at the start are ambiguous.
        if sum(nw[i:i+3] == anchor for i in range(min(3, len(nw)-2))) != 1:
            continue
        suffix = ow[hits[0]:]
        end = start + len(suffix)
        if end >= len(nw) or nw[start:end] != suffix:
            continue
        punctuation = revised[new[end-1].end():new[end].start()]
        if not re.fullmatch(r"[\s,.!?;:]*", punctuation):
            continue
        return (previous[:old[-1].end()] + punctuation.rstrip(),
                revised[new[end].start():])
    return None


class AsrSession:
    def __init__(self, conn, lock, context_seconds=None):
        self.conn = conn
        self.lock = lock
        self.id = uuid.uuid4().hex
        self.frames = []
        self.frames_lock = threading.Lock()
        self.capture_error = False
        self.capture_delay = 0.0
        self.level = 0.0
        self.device = "Microphone"
        self.continuous = state.get("continuous", False)
        if context_seconds is None:
            context_seconds = (4 if IS_LINUX and app_store
                               and app_store.settings["speech_model"] == "cohere-transcribe-en"
                               else 0)
        self.context_samples = (0 if self.continuous else
                                int(min(4, max(0, context_seconds)) * SAMPLE_RATE))
        self.voice_commands = bool(app_store and app_store.settings["voice_commands"])
        self.errors = {}
        self.audio_by_index = {}
        self._worker_failed = False
        self.started = time.monotonic()
        cm = state.get("continuous", False)
        # Continuous mode cuts far more eagerly so inline commands ("send")
        # complete within a beat of being spoken. max_seg must stay generous
        # though: a hard cut lands mid-word and the space-join then splits
        # that word in two, so it may only guard against truly nonstop speech.
        self.segmenter = Segmenter(
            min_seg=0.3 if cm else MIN_SEG,
            max_seg=10.0 if cm else MAX_SEG,
            silence_cut=0.35 if cm else SILENCE_CUT,
            keep_sil=0.1 if cm else KEEP_SIL,
            boundary_window=0 if cm else BOUNDARY_WINDOW,
        )
        self.seg_q: "queue.Queue" = queue.Queue()
        self.results = {}          # index -> transcribed text
        self.live_edit = None
        self.cancelled = False
        self._cmd_action = None    # set by feeder when a command is detected inline
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

    def submit(self, audio, replace_index=None, tail_context=None):
        idx = self._submitted if replace_index is None else replace_index
        if replace_index is None:
            self._submitted += 1
        self.audio_by_index[idx] = audio
        self.errors.pop(idx, None)
        self.seg_q.put((idx, audio, tail_context))
        return idx

    def close(self):
        self.seg_q.put(None)       # sentinel: consumer drains then exits

    def cancel(self):
        self.cancelled = True
        self._feeding = False
        if self.live_edit:
            self.live_edit.close()

    def join_consumer(self, timeout=None):
        self._consumer.join(timeout)

    def ordered_texts(self):
        return [self.results[i] for i in range(self._submitted)
                if self.results.get(i)]

    def queue_edit(self, text=None):
        if (self.cancelled or self.continuous or self.errors or self.capture_error
                or not app_store or not app_store.settings["light_edit"] or not prompt_editor):
            return
        if text is None:
            text = apply_dictionary(" ".join(self.ordered_texts()).strip())
            if app_store.settings["cleanup"]:
                text = clean_text(text)
        if not text:
            return
        if self.live_edit is None:
            self.live_edit = LiveEdit(prompt_editor)
        self.live_edit.update(text)

    def _feed(self):
        # Poll the shared frames buffer rather than doing DSP in the mic
        # callback, so the audio thread only ever appends and never blocks.
        while self._feeding:
            self._drain_frames()
            self._check_inline_cmd()
            time.sleep(0.05)

    def _check_inline_cmd(self):
        if not self.continuous or not self.voice_commands or self.cancelled:
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
            self._feeding = False
            ui_q.put(("inline_cmd", self.id))

    def _drain_frames(self):
        with self.frames_lock:
            new = self.frames[self._cursor:]
            self._cursor = len(self.frames)
        for block in new:
            for seg in self.segmenter.push(block):
                if seg.size:
                    self.submit(seg)

    def stop_feeding(self):
        """Called on stop: halt the feeder, pick up any last frames, then flush
        the Segmenter's tail so nothing the user said is left unqueued."""
        self._feeding = False
        if self._feeder is not None:
            self._feeder.join(timeout=10.0)
            if self._feeder.is_alive():
                raise TimeoutError("Audio feeder did not finish")
        self._drain_frames()
        tail = self.segmenter.finalize()
        if tail is not None and tail.size:
            # Short endings need preceding words, including the 1.6-second
            # ending that returned empty in the reported take. Replace the
            # preceding segment once; never append duplicate context text.
            if tail.size < 3 * SAMPLE_RATE and self._submitted:
                idx = self._submitted - 1
                previous = self.audio_by_index[idx]
                # Decode only the nearby ending first. The consumer falls back
                # to the full previous chunk if it cannot align exact words.
                context = previous[-4*SAMPLE_RATE:]
                log.info("final tail: %.2fs with %.2fs preceding context", tail.size/SAMPLE_RATE, context.size/SAMPLE_RATE)
                self.submit(np.concatenate([context, tail]), replace_index=idx,
                            tail_context=(previous, tail))
            else:
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
            idx, audio, tail_context = item
            self.errors.pop(idx, None)
            if self._worker_failed:
                self.errors[idx] = "Speech worker unavailable"
                continue
            # After an idle unload the respawned worker may still be loading.
            # Hold the segment here until it reports ready — recording and
            # segmentation upstream never notice the wait.
            deadline = time.monotonic() + WORKER_READY_WAIT
            while not worker_ready_evt.wait(timeout=0.1):
                if self.cancelled:
                    break
                if time.monotonic() >= deadline:
                    self.errors[idx] = "Speech model did not become ready"
                    self._worker_failed = True
                    break
            if self.cancelled or idx in self.errors:
                continue
            t0 = time.time()
            try:
                # A thinking pause can cut the audio without ending a sentence.
                # Rehear a bounded suffix with the next chunk; no second model
                # or whole-recording pass is needed. Keep the actual pause.
                previous_update = None
                previous_text = self.results.get(idx-1, "")
                context = (self.audio_by_index[idx-1][-self.context_samples:]
                           if self.context_samples and tail_context is None
                           and previous_text and idx-1 in self.audio_by_index else None)
                decode_audio = np.concatenate([context, audio]) if context is not None else audio
                kind, payload = recognize_request(self.conn, self.lock, decode_audio)
                if kind != "text":
                    raise RuntimeError(str(payload))
                text = (payload or "").strip()
                if self.cancelled:
                    continue
                if context is not None:
                    joined = join_chunk_context(previous_text, text)
                    if joined is None:
                        log.info("chunk context: no safe alignment; decoding current chunk independently")
                        kind, payload = recognize_request(self.conn, self.lock, audio)
                        if kind != "text":
                            raise RuntimeError(str(payload))
                        text = (payload or "").strip()
                    else:
                        previous_update, text = joined
                        log.info("chunk context: aligned preceding words")
                if tail_context is not None:
                    joined = join_tail_context(self.results.get(idx, ""), text)
                    if joined is None:
                        log.info("tail context has no unique anchor; decoding complete preceding chunk")
                        audio = np.concatenate(tail_context)
                        kind, payload = recognize_request(self.conn, self.lock, audio)
                        if kind != "text":
                            raise RuntimeError(str(payload))
                        text = (payload or "").strip()
                    else:
                        text = joined
                # A retry with quiet right context helps an abruptly ended word.
                frame_size = SAMPLE_RATE * FRAME_MS // 1000
                nframes = audio.size // frame_size
                level_frames = (np.sqrt(np.mean(audio[:nframes*frame_size].reshape(nframes,frame_size)**2,axis=1))
                                if nframes else np.zeros(0))
                # A nonzero microphone noise floor is not evidence of speech.
                # Require several frames above the calibrated speech threshold.
                voiced = np.count_nonzero(level_frames > max(ABS_FLOOR,self.segmenter._floor*FLOOR_MULT)) >= 3
                if not text and voiced:
                    kind, payload = recognize_request(self.conn, self.lock,
                        np.concatenate([audio, np.zeros(SAMPLE_RATE//4, dtype="float32")]))
                    if kind != "text":
                        raise RuntimeError(str(payload))
                    text = (payload or "").strip()
                    if not text:
                        self.errors[idx] = "A voiced segment returned no words"
                log.info("segment %d: %.1fs audio -> %d chars in %.1fs",
                         idx, len(audio)/SAMPLE_RATE, len(text), time.time()-t0)
                if not self.cancelled:
                    updates = {idx: text}
                    if previous_update is not None:
                        updates[idx-1] = previous_update
                    self.results.update(updates)
                    try:
                        self.queue_edit()
                    except Exception:
                        # Optional polish must never turn valid ASR into an
                        # incomplete take or retire the speech worker.
                        log.exception("could not schedule background edit")
            except Exception as exc:
                self.errors[idx] = str(exc)
                self._worker_failed = True
                log.exception("ASR segment %d failed", idx)


def retire_worker(conn):
    """Retire only the failed generation, never a replacement or another task."""
    global worker_conn, worker_proc
    with _worker_spawn_lock:
        if conn is not worker_conn:
            return
        state["model_ready"] = False
        worker_ready_evt.clear()
        if worker_proc is not None and worker_proc.is_alive():
            worker_proc.terminate()
            worker_proc.join(timeout=2)
        try:
            conn.close()
        except Exception:
            pass
        worker_conn = None
        worker_proc = None


def recognize_request(conn, lock, audio):
    """Bound send as well as recv. The helper owns its pipe exchange until done.

    On timeout the entire worker generation is retired: no future session can
    consume a late response as if it belonged to another recording.
    """
    result = queue.Queue(maxsize=1)
    def exchange():
        try:
            with lock:
                conn.send(("recognize", audio, SAMPLE_RATE))
                result.put((True, conn.recv()))
        except Exception as exc:
            result.put((False, exc))
    threading.Thread(target=exchange, name="asr-exchange", daemon=True).start()
    try:
        ok, payload = result.get(timeout=ASR_TIMEOUT)
    except queue.Empty:
        retire_worker(conn)
        raise TimeoutError("Speech model stopped responding") from None
    if not ok:
        retire_worker(conn)
        raise payload
    return payload


def acquire_single_instance():
    """Bind a fixed local port so only one copy of the app can run at a time."""
    global _lock_sock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Without SO_REUSEADDR a clean shutdown locks the app out of its own port for
    # the ~60s the last command-channel connection spends in TIME_WAIT, so
    # `--send quit` followed by a relaunch logs "another instance already
    # running" and exits — dictation stays down and it looks like a crash.
    # Measured here, and it is exactly what happened restarting to pick up the
    # overlay fix.
    #
    # It does NOT weaken the single-instance guarantee: on Linux SO_REUSEADDR
    # only bypasses TIME_WAIT, and a second bind while another socket is really
    # LISTENing still fails with EADDRINUSE (verified). Sharing a live port would
    # need SO_REUSEPORT, which is not set.
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", 49731))
    except OSError:
        return False
    s.listen(1)
    _lock_sock = s
    return True


def _resolve_input_device(sd):
    """Turn INPUT_DEVICE into something sounddevice accepts, or None for default.

    Accepts an index ("6") or a case-insensitive substring of the device name, so
    a human can write "Digital" instead of hunting for the index that PortAudio
    happens to hand out this boot. An unmatched name is logged and ignored rather
    than raised: falling back to the default mic is a better failure than not
    recording at all.
    """
    choice = app_store.settings["input_device"] if app_store and "input_device" in app_store.settings else INPUT_DEVICE
    if not choice:
        return None
    if choice.isdigit():
        return int(choice)
    needle = choice.lower()
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0 and needle in dev["name"].lower():
                return idx
    except Exception:
        log.exception("could not enumerate input devices")
        return None
    log.warning("PARAKEET_INPUT_DEVICE=%r matched no input device - using the "
                "system default", INPUT_DEVICE)
    return None


def _describe_input_device(sd, active_stream):
    """Human-readable name of the device a live stream actually opened."""
    try:
        idx = active_stream.device
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        return f"{sd.query_devices(idx)['name']} (index {idx})"
    except Exception:
        return "unknown"


def _reinit_portaudio(sd):
    """Tear PortAudio down and bring it back up.

    PortAudio is initialised once, when sounddevice is imported, and it caches
    its device list there. If that state goes bad, every later stream start
    fails with "PortAudio not initialized" [PaErrorCode -10000] and keeps
    failing for the life of the process - a dictation daemon that autostarts
    at login and runs for days then needs a manual restart to record again
    (seen 2026-08-21, after a long session of restarting pipewire underneath
    it). Re-initialising is the only way back, and it refreshes the cached
    device list as a bonus, so devices that appeared after startup show up.

    Deliberately tolerant of _terminate() failing: if PortAudio is already
    torn down, the thing we want is the _initialize() that follows.
    """
    try:
        sd._terminate()
    except Exception:
        log.exception("PortAudio terminate failed; re-initialising anyway")
    sd._initialize()


def start_recording():
    global stream, frames, session, active_take_id
    import sounddevice as sd

    if state["recording"] or state["busy"]:
        return
    # The idle timer may have unloaded the worker; bring it back BEFORE the
    # session wires itself to worker_conn. This returns as soon as the process
    # exists — the mic opens right away and the reload (~4 s) is paid in this
    # dictation's paste latency, not in lost audio.
    state["busy"] = True  # Claim the worker before the idle monitor can unload it.
    try:
        ensure_worker()
    except Exception:
        state["busy"] = False
        log.exception("could not start speech worker")
        ui_q.put(("status", "Could not start the speech model"))
        return
    touch_activity()
    with frames_lock:
        frames = []
    # Spin up the streaming session BEFORE the mic opens, so its feeder starts
    # transcribing speech in the background the moment audio arrives.
    session = AsrSession(worker_conn, worker_lock)
    active_take_id = session.id
    session.start_feeder()
    capture = session
    frames = capture.frames

    def cb(indata, n, t, status):
        with capture.frames_lock:
            capture.frames.append(indata.copy())
        # Standard deviation rejects this microphone's DC bias. No Python DSP loop
        # or UI work in PortAudio's real-time callback.
        capture.level = float(np.std(indata))
        capture.capture_delay = max(0.0, t.currentTime - t.inputBufferAdcTime - n/SAMPLE_RATE)
        if status:
            capture.capture_error = True
            log.warning("audio capture discontinuity: %s", status)

    def _open():
        # Resolve the device inside the retry, not outside it: after a
        # re-init the cached device list is rebuilt and the index that
        # PARAKEET_INPUT_DEVICE resolves to can legitimately have moved.
        device = _resolve_input_device(sd)
        s = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                           dtype="float32", callback=cb, device=device)
        try:
            s.start()
        except Exception:
            s.close()
            raise
        return s

    try:
        try:
            stream = _open()
        except sd.PortAudioError:
            # One retry, because the common failure here is recoverable state
            # inside PortAudio rather than a missing microphone. If the second
            # attempt fails too, fall through to the handler below and report.
            log.warning("mic open failed; re-initialising PortAudio and "
                        "retrying once", exc_info=True)
            _reinit_portaudio(sd)
            stream = _open()
        # Name the device in the log on every take. When a dictation comes back
        # empty the first question is always "which microphone was that?", and
        # answering it from the log beats guessing from the audio server's
        # current state minutes later.
        capture.device = _describe_input_device(sd, stream)
        log.info("input device: %s", capture.device)
    except Exception as e:
        log.exception("mic open failed")
        session.cancel()
        session.close()
        session = None
        state["busy"] = False
        state["continuous"] = False
        ui_q.put(("status", "No microphone available — open Settings"))
        return
    state["recording"] = True
    state["busy"] = False
    log.info("recording started")
    ui_q.put(("listen",) if state["continuous"] else ("show",))


def stop_and_transcribe():
    global stream
    if not state["recording"] or session is None:
        return
    sess = session
    state["recording"] = False
    state["busy"] = True
    log.info("stop requested: buffered capture delay %.3fs", sess.capture_delay)
    try:
        stream.stop()
        stream.close()
    except Exception:
        sess.capture_error = True
        log.exception("microphone close failed")
    stream = None
    ui_q.put(("processing", sess.id))
    threading.Thread(target=_finish_take, args=(sess,), name="finish-take", daemon=True).start()


def prepare_paste(sess, text):
    """Publish the native clipboard and unmap the capsule on its GUI thread."""
    if not IS_LINUX or overlay is None:
        return False
    hidden = threading.Event()
    copied = []
    ui_q.put(("prepare_paste", sess.id, hidden, text, copied))
    deadline = time.monotonic() + 2
    while not hidden.wait(.05):
        if sess.cancelled:
            return False
        if time.monotonic() >= deadline:
            raise TimeoutError("Recording capsule did not finish clipboard handoff")
    return bool(copied and copied[0])


def _finish_take(sess):
    global session, last_retry
    t_stop = time.monotonic()
    outcome = "error"
    message = "Recognition did not finish"
    audio = np.zeros(0, dtype="float32")
    out = raw = ""
    take_id = None
    try:
        sess.stop_feeding()
        sess.close()
        sess.join_consumer(timeout=ASR_TIMEOUT + WORKER_READY_WAIT + 5)
        log.info("final recognition wait: %.3fs", time.monotonic()-t_stop)
        if sess._consumer.is_alive():
            retire_worker(sess.conn)
            raise TimeoutError("Recognition did not finish in time")
        if sess.cancelled:
            return
        with sess.frames_lock:
            audio = remove_dc(np.concatenate(sess.frames).reshape(-1)) if sess.frames else audio
        duration = audio.size / SAMPLE_RATE
        log.info("recording stopped: %.1fs audio", duration)
        if duration < MIN_SECONDS:
            outcome, message = "idle", "Recording cancelled"
            return
        texts = sess.ordered_texts()
        # If no segment produced text, retry the full captured take with context.
        if not texts and not sess.errors:
            if not worker_ready_evt.wait(timeout=WORKER_READY_WAIT):
                raise TimeoutError("Speech model is not ready")
            kind, payload = recognize_request(sess.conn, sess.lock, audio)
            if kind != "text":
                raise RuntimeError(str(payload))
            texts = [payload] if payload else []
        raw = " ".join(texts).strip()
        log.info("raw ASR text: %r", raw)
        joined = apply_dictionary(raw)
        cmd = None
        # Session mode is immutable. Revalidate after the tail: never reinstate
        # an earlier inline command contradicted by later speech.
        if sess.continuous and sess.voice_commands and not sess.errors:
            joined, cmd = detect_and_strip_command(joined)
        cleanup = app_store.settings["cleanup"] if app_store else CLEANUP
        out = clean_text(joined) if cleanup else joined
        if sess.cancelled:
            return
        incomplete = bool(sess.errors) or sess.capture_error
        edit_requested = bool(out and not incomplete and not sess.continuous
                              and app_store and app_store.settings["light_edit"] and prompt_editor)
        status = "incomplete" if incomplete else "editing" if edit_requested else "ready"
        if app_store and (out or incomplete):
            take_id = app_store.add(out, raw, duration, status)
        if edit_requested:
            ui_q.put(("editing", sess.id))
            try:
                sess.queue_edit(out)
                edited = sess.live_edit.finish(out, cancelled=lambda: sess.cancelled)
                log.info("light edit: %s; stop wait %.3fs", edited.reason, edited.seconds)
                out = edited.text
            except Exception:
                log.exception("light edit failed; retaining transcript")
            if sess.cancelled:
                app_store.mark(take_id, "cancelled")
                return
            app_store.finish_edit(take_id, out)
            status = "ready"
        if out:
            save_transcript(out)
        if incomplete:
            last_retry = (take_id, audio.copy())
            if out and IS_LINUX:
                linux_backend.copy_to_clipboard(out)
            outcome, message = "error", "Recognition interrupted"
            return  # A partial take must not silently paste or send.
        if not out and not cmd:
            outcome, message = "error", "No sound detected" if float(np.std(audio)) <= ABS_FLOOR else "No words recognised"
            if app_store and take_id is None:
                take_id=app_store.add("",raw,duration,"no_speech")
            last_retry = (take_id, audio.copy())
            return
        if sess.cancelled:
            return
        if out:
            clipboard_ready = prepare_paste(sess, out)
            result = paste_text(out, cancelled=lambda:sess.cancelled, clipboard_ready=clipboard_ready)
            status = result.status
            if app_store:
                app_store.mark(take_id, status)
            outcome = "inserted" if result.status in ("inserted", "dispatched") else "recover"
            message = "Inserted" if result.confirmed else result.message
            if cmd and result.confirmed and not sess.cancelled:
                press_command_key(cmd)
        elif cmd:
            press_command_key(cmd)
            outcome, message = "inserted", "Command sent"
        log.info("stop-to-delivery: %.1fs (%d chars, %s)", time.monotonic()-t_stop,len(out),status if out else outcome)
    except Exception as exc:
        log.exception("stop/transcribe failed")
        message = "Recognition interrupted"
        if not sess.cancelled:
            if not audio.size:
                with sess.frames_lock:
                    audio = remove_dc(np.concatenate(sess.frames).reshape(-1)) if sess.frames else audio
            if app_store and take_id is None:
                take_id=app_store.add(out,raw,audio.size/SAMPLE_RATE,"incomplete")
            last_retry=(take_id,audio.copy())
    finally:
        if sess.live_edit:
            sess.live_edit.close()
        if session is sess:
            session = None
            state["busy"] = False
            touch_activity()
            if not sess.cancelled:
                ui_q.put(("outcome", sess.id, outcome, message))
                if state["continuous"] and outcome == "inserted":
                    ui_q.put(("restart",))
                elif outcome in ("error","recover"):
                    state["continuous"] = False


def retry_audio(take_id):
    global session, active_take_id
    if state["recording"] or state["busy"]:
        return
    if last_retry is None or last_retry[0] != take_id:
        ui_q.put(("status", "Audio retry is available for the last incomplete take only"))
        return
    state["busy"] = True
    ensure_worker()
    sess = AsrSession(worker_conn,worker_lock)
    sess.frames = [last_retry[1].copy()]
    session = sess
    active_take_id = sess.id
    ui_q.put(("processing", sess.id))
    threading.Thread(target=_finish_take,args=(sess,),daemon=True).start()


def report_empty_transcript(audio):
    """Explain an empty transcript instead of silently dropping it.

    There are two very different reasons the model returns nothing, and the user
    cannot tell them apart from the outside:

      - the microphone delivered no sound at all (dead device, muted input, wrong
        device picked as the system default), or
      - real audio arrived and the model just did not find words in it.

    The loudest 30 ms frame in the take separates the two, using the same floor
    the segmenter uses, so the verdict here always agrees with the segmenter's.
    """
    peak_frame = 0.0
    try:
        n = audio.size // (SAMPLE_RATE * FRAME_MS // 1000)
        if n:
            fl = SAMPLE_RATE * FRAME_MS // 1000
            frames_2d = audio[:n * fl].reshape(n, fl)
            peak_frame = float(np.sqrt(np.mean(frames_2d * frames_2d, axis=1)).max())
    except Exception:
        log.exception("could not measure the take's level")

    if peak_frame <= ABS_FLOOR:
        log.warning("EMPTY TRANSCRIPT - no audio above the silence floor "
                    "(loudest 30ms frame %.5f <= floor %.5f). The microphone "
                    "delivered silence: check that the system default input is "
                    "a device that can actually hear, or pin one with "
                    "PARAKEET_INPUT_DEVICE.", peak_frame, ABS_FLOOR)
        ui_q.put(("status", "no sound from microphone"))
    else:
        log.warning("EMPTY TRANSCRIPT - audio was present (loudest 30ms frame "
                    "%.5f > floor %.5f) but the model found no words in it.",
                    peak_frame, ABS_FLOOR)
        ui_q.put(("status", "no speech recognised"))
    # Leave the pill up long enough to be read; the caller hides it afterwards.
    time.sleep(1.8)


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


def paste_text(text, cancelled=lambda:False, clipboard_ready=False):
    if IS_LINUX:
        return linux_backend.paste_text(text, shortcut=app_store.settings["paste_shortcut"] if app_store else "ctrl+v",cancelled=cancelled,clipboard_ready=clipboard_ready)

    import keyboard
    import pyperclip

    pyperclip.copy(text)
    time.sleep(0.06)
    if cancelled():
        return linux_backend.DeliveryResult("cancelled", "Cancelled")
    keyboard.send("ctrl+v")
    return linux_backend.DeliveryResult("uncertain", "Ready to paste · insertion not confirmed")
    # The transcription deliberately STAYS in the clipboard, so if the paste
    # went to the wrong window (or nowhere), Ctrl+V drops it anywhere you like.


def press_command_key(cmd):
    """Fire a command keystroke ("enter" or a digit) after paste_text. The wait
    gives the target app time to process the Ctrl+V first — slow apps would
    otherwise see the Enter land before the pasted text."""
    if IS_LINUX:
        linux_backend.press_command_key(cmd)
        return

    import keyboard

    time.sleep(0.15)
    keyboard.send(cmd)


def toggle():
    log.info("hotkey fired: recording=%s busy=%s model_ready=%s",
             state["recording"], state["busy"], state["model_ready"])
    touch_activity()
    if state["busy"]:
        return
    if not model_available():
        ui_q.put(("status", "Model still loading…"))

        return
    if state["recording"]:
        stop_and_transcribe()
    else:
        start_recording()


def toggle_continuous():
    global session
    touch_activity()
    if state["continuous"]:
        state["continuous"] = False
        if state["recording"]:
            stop_and_transcribe()
        log.info("continuous mode off")
    else:
        state["continuous"] = True
        log.info("continuous mode on")
        if not state["recording"] and not state["busy"] and model_available():
            start_recording()


def restart_if_continuous():
    """Continuous-mode re-arm, fired 400ms after a dictation finishes. The
    delay is a window in which the user may have toggled continuous mode OFF —
    re-check at fire time, or the mic would silently reopen after they said
    stop. start_recording() itself guards recording/busy."""
    if state["continuous"] and model_available():
        start_recording()


def detect_and_strip_command(text):
    text = text.strip()
    m = _NUM_RX.fullmatch(text)
    if m:
        word = m.group(1).lower()
        return text[:m.start()].rstrip(), _NUMBER_MAP[word]
    m = _SEND_RX.search(text)
    if m:
        return text[:m.start()].rstrip(), "enter"
    return text, None


def handle_inline_command(session_id=None):
    if session is None or (session_id is not None and session.id != session_id):
        return
    stop_and_transcribe()


def cancel():
    global stream, session, active_take_id
    active_take_id = None
    state["continuous"] = False
    sess = session
    if sess is None:
        ui_q.put(("hide",))
        return
    sess.cancel()
    if state["recording"]:
        try:
            stream.stop()
            stream.close()
        except Exception:
            log.exception("microphone close during cancel failed")
        sess.close()
    stream = None
    session = None
    state["recording"] = False
    state["busy"] = False
    touch_activity()
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


def pump_ui(ov):
    try:
        while True:
            msg = ui_q.get_nowait()
            kind = msg[0]
            if kind in ("processing", "editing", "inline_cmd", "outcome", "prepare_paste") and len(msg)>1:
                if msg[1] != active_take_id:
                    continue
            if kind == "action":
                actions = {"toggle": toggle, "continuous": toggle_continuous,
                           "cancel": cancel, "quit": quit_app}
                if msg[1] in actions:
                    actions[msg[1]]()
            elif kind in ("show", "listen"):
                ov.show_pill("Listening" if kind=="show" else "Listening · hands-free", "#d8f58a", "recording")
            elif kind == "processing":
                ov.show_pill("Finishing your words", "#d8f58a", "processing")
            elif kind == "editing":
                ov.show_pill("Polishing your wording", "#d8f58a", "processing")
            elif kind == "prepare_paste":
                ov.hide()
                def publish(handoff=msg):
                    if handoff[1] != active_take_id:
                        handoff[2].set()
                        return
                    try:
                        handoff[4].append(bool(ov.copy_to_clipboard(handoff[3])) if hasattr(ov,"copy_to_clipboard") else False)
                    except Exception:
                        log.exception("Native clipboard handoff failed")
                        handoff[4].append(False)
                    finally:
                        handoff[2].set()
                ov.root.after(80, publish)
            elif kind == "loading":
                if not state["model_ready"]:
                    ov.show_mini("#d8f58a")
            elif kind == "ready":
                if not state["recording"] and not state["busy"]:
                    ov.hide()
                    if state["continuous"]:
                        start_recording()
            elif kind == "status":
                ov.show_pill(msg[1], "#f4cc8a", "status")
            elif kind == "outcome":
                if msg[2] == "error":
                    ov.show_pill(msg[3], "#f4cc8a", "status")
                else:
                    # Paste once, leave the text on the clipboard, disappear.
                    # No confirmation screen or second user action is required.
                    ov.hide()
            elif kind == "hide":
                if not state["recording"] and not state["busy"]:
                    ov.hide()
            elif kind == "restart":
                ov.root.after(400, restart_if_continuous)
            elif kind == "inline_cmd":
                handle_inline_command(msg[1])
            elif kind == "quit":
                cancel()
                ov.root.destroy()
                return False
    except queue.Empty:
        pass
    return True


def _gtk_poll(ov):
    """The GTK half of the poll loop. GLib timeouts replace tk's `after`."""
    if pump_ui(ov) is False:
        return
    ov.after(60, lambda: _gtk_poll(ov))


class Overlay:
    # `-transparentcolor` is a Windows-only tk attribute; on X11/XWayland the
    # call below fails and the window really does paint KEY. So KEY is chosen to
    # sit just under the pill fill rather than at near-black: the only pixels
    # that ever show it are the four corner arcs, and at this radius they read as
    # a soft shadow instead of a black notch. (Per-pixel transparency is not
    # available to tk on X11 at all — a shaped window would need the X Shape
    # extension, which tk does not expose.)
    # KEY is deliberately only a hair off PILL. tk cannot do per-pixel alpha on
    # X11 at all, so the four corner arcs outside the rounded rect are really
    # painted — at near-black they read as black notches on a rounded box, which
    # is exactly what they look like. Keeping KEY within a couple of steps of the
    # fill makes them invisible against the chip, and ALPHA below softens the
    # whole window so the remaining difference reads as a shadow rather than a
    # corner. This is the closest thing to a rounded, floating chip that tk can
    # actually draw here.
    KEY = "#101318"
    PILL = "#12151a"         # chip fill, darker so the type separates cleanly
    BORDER = "#2f3644"       # hairline only; it should not read as a button edge
    TEXT = "#d8dee9"         # soft white — pure #fff glares against a dark chip
    ALPHA = 0.92             # whole-window opacity; X11 + a compositor honour it

    # The overlay is drawn by tk, which knows nothing about the display scale, so
    # everything below is expressed in logical units and multiplied by this.
    #
    # NOT a constant any more, and that is the whole point. It used to be a
    # hardcoded 1.7, which was a calibration for exactly one display
    # configuration: at fractional scale 1.25 the compositor squeezed XWayland's
    # 3072px-wide framebuffer onto a 1920px panel, so 1.7 reached the glass as
    # 1.0625. Drop to scale 1.0 and there is no squeeze, so the same 1.7 draws
    # the chip 70% oversized. screen_metrics measures the squeeze and solves for
    # the multiplier instead. PARAKEET_UI_SCALE still overrides.
    SCALE = screen_metrics.LEGACY_SCALE

    # Fonts: "Segoe UI" does not exist on Linux, so tk silently fell back to a
    # default that is neither the system font nor well hinted. Pick the first
    # family that is actually installed.
    FONT_CANDIDATES = ("Ubuntu Sans", "Ubuntu", "Cantarell", "DejaVu Sans", "Noto Sans")

    def _pick_font(self):
        try:
            from tkinter import font as tkfont
            have = {f.lower() for f in tkfont.families(self.root)}
            for fam in self.FONT_CANDIDATES:
                if fam.lower() in have:
                    return fam
        except Exception:
            pass
        return "TkDefaultFont"

    def s(self, v):
        """Scale a logical dimension to device pixels."""
        return int(round(v * self.SCALE))

    def _refresh_screen(self):
        """Re-measure the screen and the scale, right before we place the chip.

        This is the fix for the chip disappearing at a scale change, and it has
        to be a real measurement rather than another `winfo_screenwidth()`: Tk
        reads that out of Xlib's Screen struct, which is filled in once at
        connect and only ever refreshed by `XRRUpdateConfiguration()`. Tk never
        calls it — `libtk8.6.so` does not even link libXrandr — so Tk's answer is
        frozen for the life of the process. It was still reporting the old
        3072x1728 hours after the desktop became 1920x1080, which is precisely
        how the chip ended up placed 437px below the bottom of the screen.

        Fails soft in every direction: if the measurement is unavailable we keep
        whatever we had, which is no worse than the old behaviour.
        """
        got = screen_metrics.metrics(fallback_size=(self.sw, self.sh))
        if not got:
            return
        sw, sh, scale = got
        if (sw, sh, scale) != (self.sw, self.sh, self.SCALE):
            log.info("overlay: screen now %dx%d, scale %.3f (was %dx%d, %.3f)",
                     sw, sh, scale, self.sw, self.sh, self.SCALE)
        self.sw, self.sh, self.SCALE = sw, sh, scale

    def __init__(self, root):
        import tkinter as tk

        self.tk = tk
        self.root = root
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            # Windows-only; fails silently on X11. Left in so the Windows build
            # keeps its real per-pixel key transparency.
            root.attributes("-transparentcolor", self.KEY)
        except Exception:
            pass
        try:
            root.attributes("-alpha", self.ALPHA)
        except Exception:
            pass
        self.sw = root.winfo_screenwidth()
        self.sh = root.winfo_screenheight()
        self._refresh_screen()
        self.font_family = self._pick_font()
        log.info("overlay: font=%r scale=%.2f screen=%dx%d",
                 self.font_family, self.SCALE, self.sw, self.sh)
        # Canvas background is the chip fill, NOT the transparency key.
        #
        # tk cannot make the key colour transparent on X11 (-transparentcolor is
        # Windows-only), so anything the chip does not cover is really painted.
        # Drawing a rounded pill inside the window therefore left four painted
        # corner blocks — invisible on a dark desktop, obvious black squares over
        # anything light, which is exactly what Tristan could see.
        #
        # So the chip now fills its window edge to edge and the corners are
        # rounded by the compositor instead: the rounded-window-corners shell
        # extension clips window corners with real alpha, which is the one place
        # on this machine where per-pixel transparency is actually available to
        # an X11 window.
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

    def _shape_rounded(self, w, h, r):
        """Clip the window itself to a rounded rectangle, via the X Shape extension.

        This is the only way to get rid of the corners on this toolkit. tk has no
        per-pixel transparency on X11 — `-transparentcolor` is Windows-only — so
        every pixel of the window is really painted and a rounded chip drawn
        inside it leaves four painted corner blocks. Invisible on a dark desktop,
        obvious black squares over anything light.

        X can clip the window's shape even though tk cannot blend it. tk hands us
        the XID, GDK wraps that foreign window, and the corners are then simply
        not part of the window: whatever is behind shows through exactly.

        The mask is binary, so the corners are hard-edged rather than
        antialiased. That is the trade for using tk at all, and a crisp corner
        beats a black block. Fails soft: if anything here is unavailable the chip
        just keeps square corners.
        """
        try:
            # GDK must open an X11 display, not a Wayland one, or
            # X11Window.foreign_new_for_display cannot wrap tk's XID at all. This
            # process has no GDK_BACKEND of its own, so GDK would default to
            # Wayland and the shape would silently never apply.
            os.environ.setdefault("GDK_BACKEND", "x11")
            import gi
            gi.require_version("Gdk", "3.0")
            from gi.repository import Gdk
            import cairo
        except Exception:
            return
        try:
            xid = self.root.winfo_id()
            display = Gdk.Display.get_default()
            if display is None:
                return
            gwin = Gdk.X11Window.foreign_new_for_display(display, xid)
            if gwin is None:
                return
            surface = cairo.ImageSurface(cairo.FORMAT_A8, int(w), int(h))
            cr = cairo.Context(surface)
            cr.set_source_rgba(0, 0, 0, 0)
            cr.set_operator(cairo.OPERATOR_SOURCE)
            cr.paint()
            cr.set_operator(cairo.OPERATOR_OVER)
            cr.set_source_rgba(1, 1, 1, 1)
            import math as _m
            r = min(r, w / 2.0, h / 2.0)
            cr.new_sub_path()
            cr.arc(w - r, r, r, -_m.pi / 2, 0)
            cr.arc(w - r, h - r, r, 0, _m.pi / 2)
            cr.arc(r, h - r, r, _m.pi / 2, _m.pi)
            cr.arc(r, r, r, _m.pi, 3 * _m.pi / 2)
            cr.close_path()
            cr.fill()
            region = Gdk.cairo_region_create_from_surface(surface)
            gwin.shape_combine_region(region, 0, 0)
        except Exception:
            pass

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r,
               x2, y2, x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r,
               x1, y1 + r, x1, y1]
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    def _draw_pill(self, text, base_color):
        c = self.canvas
        self._refresh_screen()
        c.delete("all")
        # This is a status indicator, not a button. The type is deliberately the
        # quiet element — cap height around a third of the chip, with padding
        # doing the work — and the dot carries the signal. Setting the text near
        # the chip height (the earlier 12 against 34) is what made it read as an
        # oversized button.
        pad_x, dot_r, gap = self.s(15), self.s(4), self.s(10)
        font = (self.font_family, self.s(9))
        tmp = c.create_text(0, 0, text=text, font=font, anchor="nw")
        bb = c.bbox(tmp)
        c.delete(tmp)
        tw = bb[2] - bb[0]
        w = pad_x + dot_r * 2 + gap + tw + pad_x
        h = self.s(28)
        bw = max(1, self.s(1))
        w = int(w)
        # Clamped, so a screen that shrinks under us costs the chip a few pixels
        # of position rather than pushing it off the desktop entirely.
        px, py = screen_metrics.clamp_on_screen(
            (self.sw - w) // 2, self.sh - h - self.s(96), w, h, self.sw, self.sh)
        self.root.geometry(f"{w}x{h}+{px}+{py}")
        # Edge to edge, deliberately square here: the corners are rounded by the
        # compositor (rounded-window-corners), which can do it with real alpha.
        # Drawing the rounding ourselves is what produced painted black corner
        # blocks, because everything this canvas does not cover is still painted.
        # Corner radius is a compromise, and the reason is worth writing down.
        #
        # tk cannot make anything transparent on X11, so whatever the chip does
        # not cover is painted in KEY — four wedges at the corners. Their AREA
        # grows with the radius, so a full pill (radius = h/2) produces the
        # biggest possible black wedges, which is what showed up against a white
        # background. A modest radius keeps the rounded look and shrinks the
        # artifact to a few pixels.
        #
        # Everything that would remove it outright was tried and failed here:
        # a GTK window with a real ARGB visual (draws correctly standalone, never
        # becomes visible inside this app), and an X11 Shape mask over tk's own
        # XID (never applied, on either GDK backend). See FINDINGS S-007.
        radius = self.s(8)
        self._round_rect(0, 0, w, h, radius, fill=self.PILL,
                         outline=self.BORDER, width=bw)
        cy = h // 2
        dx = pad_x + dot_r
        self.dot_id = c.create_oval(dx - dot_r, cy - dot_r, dx + dot_r, cy + dot_r,
                                    fill=base_color, outline="")
        self.text_id = c.create_text(dx + dot_r + gap, cy, text=text, anchor="w",
                                     fill=self.TEXT, font=font)
        self._base_color = base_color

    def _draw_mini(self, color):
        c = self.canvas
        self._refresh_screen()
        c.delete("all")
        w = h = self.s(20)
        r = self.s(5)
        px, py = screen_metrics.clamp_on_screen(
            self.sw - w - self.s(22), self.sh - h - self.s(58),
            w, h, self.sw, self.sh)
        self.root.geometry(f"{w}x{h}+{px}+{py}")
        self.dot_id = c.create_oval(r, r, w - r, h - r, fill=color, outline="")
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
        if pump_ui(self) is False:
            return
        self.root.after(60, self.poll)


def main():
    global app_store, overlay, prompt_editor
    if not IS_LINUX:
        import tkinter as tk
        import keyboard

    log.info("=== app starting (pid=%s) ===", os.getpid())
    if not acquire_single_instance():
        log.info("another instance already running - exiting")
        return

    if IS_LINUX:
        # GNOME owns the key grab and pokes us over the lock socket, so the
        # single-instance port doubles as the command channel.
        linux_backend.serve_commands(_lock_sock, {
            "toggle": lambda: ui_q.put(("action", "toggle")),
            "toggle-continuous": lambda: ui_q.put(("action", "continuous")),
            "cancel": lambda: ui_q.put(("action", "cancel")),
            "quit": lambda: ui_q.put(("action", "quit")),
            "status": lambda: json.dumps(dict(state, version=APP_VERSION, build=BUILD_ID,
                                               model=SPEECH_MODELS[app_store.settings["speech_model"]] if app_store else MODEL_NAME,
                                               cleanup=app_store.settings["cleanup"] if app_store else CLEANUP,
                                               light_edit=bool(app_store and app_store.settings["light_edit"]),
                                               editor=prompt_editor.state if prompt_editor else "disabled",
                                               pid=os.getpid(), worker_pid=worker_proc.pid if worker_proc else None)),
        })

    # Make sure the editable dictionary.txt exists (first run drops the commented
    # template next to the exe / in the repo root) and load it now, so the rule
    # count is logged at startup and corrections are ready before the first
    # dictation. Later edits hot-reload on their own, no restart needed.
    app_store = AppStore(BASE_DIR)
    if app_store.settings["light_edit"]:
        import atexit
        from prompt_editor import PromptEditor
        prompt_editor = PromptEditor(BASE_DIR)
        atexit.register(prompt_editor.close)
        prompt_editor.start()
    if not app_store.settings_file.exists() and INPUT_DEVICE:
        app_store.update("input_device", INPUT_DEVICE)
    app_store.import_legacy(os.path.join(BASE_DIR,"transcripts.log"))
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
    spawn_worker()

    # Arm the idle unload: a dictation daemon that autostarts at login used to
    # hold the loaded model (~800 MB) forever, most of it eventually paid for
    # in swap/zram churn on a busy machine. See IDLE_UNLOAD_SEC for why the
    # unload is a process kill and what a reload costs.
    if IDLE_UNLOAD_SEC > 0:
        threading.Thread(target=idle_monitor, name="idle-unload",
                         daemon=True).start()
        log.info("idle unload armed: worker exits after %.0fs without "
                 "dictation (PARAKEET_IDLE_UNLOAD_SEC)", IDLE_UNLOAD_SEC)
    else:
        log.info("idle unload disabled (PARAKEET_IDLE_UNLOAD_SEC=0)")

    if IS_LINUX and os.environ.get("PARAKEET_LEGACY_UI") != "1":
        from native_ui import NativeUI
        def sample():
            sess=session
            return {"level": sess.level if sess else 0,
                    "device": sess.device if sess else "Microphone off",
                    "continuous": state["continuous"]}
        ov = NativeUI(app_store, {
            "toggle": toggle, "cancel": cancel, "quit": quit_app,
        }, sample)
        root=ov
        ov.after(60,lambda:_gtk_poll(ov))
        log.info("overlay: native recording capsule")
    else:
        import tkinter as tk
        root=tk.Tk()
        root.title("Parakeet Dictation")
        ov=Overlay(root)
    overlay=ov

    if not model_cached and app_store.settings["speech_model"] == "parakeet":
        # The normal ("hide",) on worker-ready clears this once the download and
        # load finish.
        ui_q.put(("status", "Downloading speech model (one time, ~460 MB)"))

    # Manual Ctrl+Win / Ctrl+Shift+Win detection via a raw hook (keyboard's
    # add_hotkey chokes
    # on modifier-only combos). Key NAMES are locale-dependent: on a German
    # keyboard Ctrl reports as "strg" and the Win key as "linke/rechte
    # windows", so we normalize by locale before matching. No suppression is
    # needed: only Win *alone* opens the Start menu, and here it's held with
    # Ctrl. Esc (cancel) and Ctrl+Alt+Q (quit) are handled here too, because
    # add_hotkey's English key names also miss on non-English layouts.
    keys_down = set()
    # One fire per physical press: Windows auto-repeats held modifier keys, so
    # without this latch a combo held a beat too long would toggle over and
    # over. The fire itself is DELAYED 150ms because the keys of a chord never
    # land in one fixed order — Ctrl→Win→Shift would otherwise match plain
    # Ctrl+Win the instant Win goes down and start a one-shot recording the
    # user never asked for. The delay gives a late Shift time to arrive before
    # we decide which toggle was meant; 150ms is imperceptible on a hotkey.
    combo = {"active": False, "shift_seen": False}

    def fire_combo():
        if combo["shift_seen"] or "shift" in keys_down:
            ui_q.put(("action", "continuous"))
        else:
            ui_q.put(("action", "toggle"))

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
        k = norm(e.name)
        if e.event_type == "down":
            keys_down.add(k)
            if k == "esc":
                ui_q.put(("action", "cancel"))
                return
            if "ctrl" in keys_down and "alt" in keys_down and k == "q":
                quit_app()
                return
            if {"ctrl", "windows"} <= keys_down:
                if not combo["active"]:
                    combo["active"] = True
                    combo["shift_seen"] = "shift" in keys_down
                    threading.Timer(0.15, fire_combo).start()
                elif k == "shift":
                    combo["shift_seen"] = True
        else:
            keys_down.discard(k)
            if k in ("ctrl", "windows"):
                combo["active"] = False

    if not IS_LINUX:
        keyboard.hook(on_key)
        log.info("manual hotkey hooks installed (ctrl+win, ctrl+shift+win, locale-aware)")
    else:
        log.info("linux: hotkeys served by GNOME via the command channel")
    log.info("entering mainloop")
    try:
        root.mainloop()
    finally:
        if prompt_editor is not None:
            prompt_editor.close()
        if worker_conn is not None:
            retire_worker(worker_conn)
        if _lock_sock is not None:
            _lock_sock.close()
    log.info("mainloop exited")


if __name__ == "__main__":
    mp.freeze_support()
    main()
