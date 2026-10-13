"""Local preferences and recoverable transcripts. No UI or ASR dependencies."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager


SPEECH_MODELS = {"parakeet": "nemo-parakeet-tdt-0.6b-v3",
                 "parakeet-unified": "nvidia/parakeet-unified-en-0.6b",
                 "qwen-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
                 "cohere-transcribe-en": "CohereLabs/cohere-transcribe-03-2026 (English, Vulkan Q8)"}
DEFAULTS = {"input_device": "", "cleanup": False, "voice_commands": False,
            "paste_shortcut": "ctrl+v", "reduced_motion": False, "light_edit": False,
            "speech_model": "parakeet"}


class AppStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.settings_file = self.directory / "settings.json"
        self.history_file = self.directory / "history.sqlite3"
        self.lock = threading.RLock()
        self.settings = dict(DEFAULTS)
        try:
            data = json.loads(self.settings_file.read_text())
            if not isinstance(data, dict):
                raise ValueError("Preferences must be an object")
            for key, default in DEFAULTS.items():
                if isinstance(data.get(key), type(default)):
                    if key == "speech_model" and data[key] not in SPEECH_MODELS:
                        continue
                    self.settings[key] = data[key]
        except (OSError, ValueError, TypeError):
            pass
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS takes (id TEXT PRIMARY KEY, created REAL, text TEXT, raw TEXT, duration REAL, status TEXT)")
        try:
            os.chmod(self.history_file, 0o600)
        except OSError:
            pass

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.history_file, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def update(self, key, value):
        if key not in DEFAULTS or type(value) is not type(DEFAULTS[key]):
            raise ValueError("Unknown preference or invalid value")
        if key == "speech_model" and value not in SPEECH_MODELS:
            raise ValueError("Unsupported speech model")
        if key == "paste_shortcut" and value not in ("ctrl+v", "ctrl+shift+v"):
            raise ValueError("Unsupported paste shortcut")
        with self.lock:
            settings = dict(self.settings, **{key: value})
            fd, name = tempfile.mkstemp(dir=self.directory, prefix=".settings-")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(settings, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(name, self.settings_file)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            self.settings = settings

    def add(self, text, raw="", duration=0, status="ready", take_id=None):
        take_id = take_id or uuid.uuid4().hex
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO takes VALUES (?,?,?,?,?,?)",
                       (take_id, time.time(), text, raw, duration, status))
        return take_id

    def mark(self, take_id, status):
        with self.lock, self.connect() as db:
            db.execute("UPDATE takes SET status=? WHERE id=?", (status, take_id))

    def finish_edit(self, take_id, text):
        """Replace display text, retaining the original ASR text for recovery."""
        with self.lock, self.connect() as db:
            db.execute("UPDATE takes SET text=?, status='ready' WHERE id=?", (text, take_id))

    def recent(self, limit=50):
        with self.lock, self.connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(
                "SELECT * FROM takes ORDER BY created DESC LIMIT ?", (limit,))]

    def import_legacy(self, filename):
        """One-time migration of existing text history, without editing it."""
        import re
        with self.lock, self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS migrations (name TEXT PRIMARY KEY)")
            if db.execute("SELECT 1 FROM migrations WHERE name='legacy'").fetchone():
                return
            try:
                content = Path(filename).read_text(encoding="utf-8")
            except FileNotFoundError:
                content = ""
            for stamp, text in re.findall(r"^\[([^\]\n]+)\]\n(.*?)(?=^\[|\Z)", content, re.M | re.S):
                try:
                    created = time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
                if text.strip():
                    db.execute("INSERT INTO takes VALUES (?,?,?,?,?,?)",
                               (uuid.uuid4().hex, created, text.strip(), "", 0, "saved"))
            db.execute("INSERT INTO migrations VALUES ('legacy')")
