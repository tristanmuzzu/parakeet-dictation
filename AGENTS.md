# Parakeet Dictation: agent guide

Use [INSTALL.md](INSTALL.md) for installation. Windows setup is in
[docs/WINDOWS-AGENT-SETUP.md](docs/WINDOWS-AGENT-SETUP.md); Linux setup is in
[docs/LINUX.md](docs/LINUX.md). Follow a setup runbook only for a requested setup
or repair, not as a prerequisite to ordinary development.

## Product constraints

- Recognition is local. Default to multilingual Parakeet; optional model trials
  require separately installed assets. Never fall back to a cloud API.
- Keep speech inference in a separate worker. Loading a model in the UI/hotkey
  process can kill Windows' keyboard hook or freeze the recording capsule.
- Linux has one non-focusable recording capsule: record, transcribe while
  speaking, paste on stop, hide. No panel, tray or insertion-confirmation step.
- Preserve clipboard recovery and local history. Do not repeat an uncertain paste.
- Executable and source share the model cache and loopback single-instance lock
  at `127.0.0.1:49731`. Inspect before launching or replacing a running app.
- Preserve existing shortcuts. Linux bindings belong to the desktop; the generic
  GNOME installer must not overwrite an existing extension's integration.
- macOS is not implemented. Windows PowerShell installers cannot run on Linux.

## Development

Read the changed module and matching tests first. Run `python -m pytest tests/`
with the project's environment. On Linux, GTK tests need system introspection
bindings. Tests use synthetic inputs and isolated stores; real microphone/paste
verification is a separate check on the target platform.

`dictation.py` owns recording and worker coordination; `linux_backend.py` owns
command dispatch and paste; `native_ui.py` owns the Linux capsule. `app_store.py`
owns settings/history. `speech_models.py` selects the worker's model. Optional
installers are under `tools/` and must verify their pinned downloads.

Keep chunk boundaries, cancellation and worker ownership intact. Run a matching
regression for late worker output, short final chunks or clipboard changes.

## Local data

Never commit or upload `dictionary.txt`, `settings.json`, `history.sqlite3`,
`transcripts.log`, `dictation.log`, virtual environments or model caches.
Keep bug-report fixtures synthetic. Do not quote real dictated text publicly.

When a user asks to correct their dictionary, append the correct spelling on its
own line. Use `wrong -> right` only when a specific override is needed. Preserve
existing entries, mention the addition, and do not treat an uncertain name as a
confirmed transcription error. The dictionary hot-reloads without a restart.
