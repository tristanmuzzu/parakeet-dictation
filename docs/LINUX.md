# Linux setup

Source-only support for a GNOME desktop with XWayland. The UI is a small GTK
recording capsule. Recognition runs in a separate worker; the capsule hides
before text is pasted into the focused app.

## Dependencies

On Ubuntu/Debian, install the desktop/Python prerequisites:

```bash
sudo apt install python3-venv python3-gi python3-gi-cairo python3-cairo \
  gir1.2-gtk-3.0 libportaudio2 wl-clipboard ydotool xwayland
```

The app needs a working **ydotool daemon** and permission to use its socket for
Ctrl+V injection. Service names and socket locations vary by distribution;
inspect `systemctl status ydotoold` and your package's instructions first.
Do not grant broad permissions to `/dev/uinput` or run the app as root.
The backend defaults to `/run/ydotoold.socket`; set `YDOTOOL_SOCKET` to the
actual user-accessible socket if different. The daemon can inject input into
other apps, so limit its socket access to the intended user.

Create an environment that can see system GTK bindings:

```bash
git clone https://github.com/tristanmuzzu/parakeet-dictation.git
cd parakeet-dictation
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import gi, cairo, sounddevice; gi.require_version('Gtk', '3.0'); from gi.repository import Gtk"
.venv/bin/python selftest.py
```

First-run recognition downloads the default model (about 460 MB). Wait for
`SELFTEST_OK`. If your Python version has no compatible ONNX wheel, use one
supported by `onnx-asr` and matching system GTK bindings. Do not replace system
Python. The project does not require CUDA or a GPU for default recognition.

## Launch and configure shortcuts

Check for a running instance before launching:

```bash
.venv/bin/python dictation.py --send status
```

If none is running, launch from the repo in a terminal:

```bash
GDK_BACKEND=x11 .venv/bin/python dictation.py
```

Keep the terminal open during first verification. Model readiness can be checked
from a second terminal with `--send status`. Do not launch another recorder.

For a **new** GNOME installation with no existing dictation shortcuts:

```bash
PARAKEET_PYTHON="$PWD/.venv/bin/python" bash install-gnome-hotkeys.sh
```

The installer preserves unrelated custom bindings. Defaults:
Ctrl+Super+Space toggles recording, Ctrl+Shift+Super+Space toggles continuous mode,
Ctrl+Super+Escape cancels, Ctrl+Alt+Q quits. Environment overrides let you choose
another binding, for example `TOGGLE_KEY='<Control><Alt>space'`.
Check for conflicts in GNOME's Keyboard settings. Existing installs may use an
extension and different keys; do not add competing bindings over them.

Focus a plain text editor, record a short sentence, stop, and verify the text
appears once. If automatic paste misses, Ctrl+V recovers it. Do not automatically
retry an uncertain paste. `--send toggle`, `--send cancel`, `--send quit` and
`--send status` use the existing loopback command channel.

## Settings and privacy

`settings.json` in the app directory supports `input_device`, `cleanup`,
`light_edit`, `speech_model`, `voice_commands`, `paste_shortcut` and
`reduced_motion`. Load settings at startup. Default recognition is `parakeet`;
cleanup, voice commands and light editing are off. `paste_shortcut` accepts
`ctrl+v` or `ctrl+shift+v` for terminals. `dictionary.txt` hot-reloads.

The model stays warm by default on Linux. `PARAKEET_IDLE_UNLOAD_SEC` can unload
it after an idle timeout, at the cost of the next take's reload delay.
`PARAKEET_INPUT_DEVICE` can select a microphone; inspect your device names first.

Text stays on the clipboard. `transcripts.log` and `history.sqlite3` keep local
history. No microphone audio is saved by the app. These files and settings are
gitignored; keep them out of bug reports and commits.

## Optional models

These are source-only experiments. Keep default Parakeet for a normal setup;
alternative models may be slower or less accurate for your speech. No trial is
an automatic upgrade. Weights have their own licenses.

| Setting | Route | Limits |
| --- | --- | --- |
| `parakeet-unified` | `tools/setup_parakeet_unified.py` | English-only, local ONNX CPU. |
| `qwen-asr-1.7b` | `tools/setup_qwen_asr.py` | Multilingual CPU trial; can be slow. |
| `cohere-transcribe-en` | `tools/setup_transcribe_speech.py` | Linux x64, hardware Vulkan, English hint, Q8. No CPU fallback. |

Run the relevant script's `--help` first. Install its setup requirements where
needed (`tools/requirements-speech-setup.txt`). Downloads are pinned and verified;
installers do not change active settings. Allow several GB for native model trials.
Quit only while idle, select `speech_model` in settings, then relaunch with the
same environment. Verify `--send status` reports the selected model and readiness.
Rollback selects `parakeet`; keep the original model cache.

Optional `light_edit` uses a separately installed local editor from
`tools/setup_prompt_editor.py`. It is off by default, may add latency, and falls
back to original text on failed or rejected edits. Original transcripts remain
in history. Do not install it as part of normal transcription setup.

## Development and removal

Run `.venv/bin/python -m pytest tests/ -q` with pytest installed in the development
environment. Unit tests use synthetic audio and isolated data. Microphone,
shortcuts, window focus and actual paste need a desktop check.

Quit with `dictation.py --send quit`. Remove only shortcuts created by the generic
installer with `bash install-gnome-hotkeys.sh --remove`; leave an existing extension
alone. Login startup is optional and should use this same interpreter and working
directory. Remove only the startup entry you created. Keep history and model caches
unless the user explicitly asks to delete them.
