# Agent runbook: set up Parakeet Dictation for the user

You are setting up free, unlimited voice dictation on the user's **Windows** machine, built mainly for talking to AI coding agents (Cursor, Claude Code, ChatGPT and the like) without paying for a tool like Wispr Flow. Everything runs locally; the only download is the speech model from Hugging Face on first run. Target result: the user presses **Ctrl+Win**, speaks, presses **Ctrl+Win** again, and the recognized text is typed into the focused window.

Non-Windows: this app is Windows-only (global-hotkey + overlay implementation). On macOS/Linux, tell the user and stop.

Exe vs source: there is also a packaged one-click build (`ParakeetDictation.exe`, shipped as `ParakeetDictation-win64.zip` on the GitHub Releases page) for non-technical humans who have no Python. As an agent, prefer the from-source route below: it is what this runbook documents, it is easier to verify and customize, and the fix table maps to source. Only point a user at the exe if they explicitly do not want Python. Both routes share one model cache and the single-instance lock (127.0.0.1:49731), so only one copy ever runs regardless of which was launched.

## 1. Preconditions

1. Windows 10/11.
2. Python **3.10-3.12** available (`py -3.12 --version` or `python --version`). 3.13+ may lack prebuilt wheels for some dependencies, so if only 3.13+ exists, install 3.12 first (winget: `winget install Python.Python.3.12`).
3. A working microphone (Windows Settings → Privacy → Microphone allowed for desktop apps).
4. ~1.5 GB free disk (venv + model cache).

## 2. Install

From the repo root, in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

Creates `.venv` and installs `requirements.txt`. If PowerShell scripting is blocked entirely, do it manually:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Verify the ASR core (no mic needed)

```powershell
.\.venv\Scripts\python.exe selftest.py
```

- First run downloads the model (~460 MB int8). Takes minutes; be patient.
- Success = output ends with `SELFTEST_OK`.
- If int8 weights fail to download it falls back to fp32 (~600 MB, slower load), which is still fine.

## 4. Launch and verify end-to-end

```powershell
Start-Process -FilePath ".\.venv\Scripts\pythonw.exe" -ArgumentList "dictation.py" -WorkingDirectory (Get-Location)
```

(or tell the user to double-click `Start Dictation (debug).bat` to watch it in a console).

The app writes `dictation.log` in the repo root. That is your primary verification surface:

1. Wait for `worker reports ready` in the log (~15-30 s int8).
2. Have the USER press **Ctrl+Win**, say a sentence, press **Ctrl+Win** again, with a text field (e.g. Notepad) focused.
3. Log must show: `hotkey fired` → `recording started` → `recording stopped: N.Ns audio` → `ASR result: N chars`.
4. Confirm the text appeared in the focused field and is accurate.

Only one instance runs at a time (TCP lock on 127.0.0.1:49731); extra launches exit silently, so check the log's `another instance already running` line if confused.

## 5. Auto-start (recommended, needs one admin approval)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install-autostart.ps1
```

Registers Task Scheduler task `ParakeetDictation` (at logon, elevated, silent) and starts it now. Verify: `Get-ScheduledTask -TaskName ParakeetDictation` → State `Ready`/`Running`. Removal: `uninstall-autostart.ps1`.

## 6. Known failure modes (all hit in real setups, check these before debugging blind)

| Symptom | Cause | Fix |
|---|---|---|
| Hotkey never fires, log shows no `hotkey fired` | **Localized key names.** Non-English Windows reports e.g. German `strg` (Ctrl) and `linke windows` (Win). English + German are already normalized in `norm()` in `dictation.py`. | Run `tools/keytest.py`, have the user press Ctrl+Win, read the printed names, add them to `norm()`. |
| Hotkey dead only after boot / model load | Keyboard hook killed by Windows when the hooked process pegs CPU (LowLevelHooksTimeout). Should not happen, since the model loads in a separate worker process by design. | Verify only ONE dictation instance runs and that you didn't merge worker into main. |
| `add_hotkey` style combos don't fire | The `keyboard` lib mishandles modifier-only combos (Ctrl+Win has no regular key), and `suppress=True` makes it eat keys globally without firing. | Don't refactor to `add_hotkey`; keep the raw `keyboard.hook` + manual state tracking. No suppression needed, since only Win *alone* opens the Start menu. |
| `mic error` in overlay/log | No default input device or mic privacy blocked | Windows Settings → Sound → Input; Privacy → Microphone. |
| int8 download fails midway | Interrupted Hugging Face snapshot; loader then errors with "incomplete snapshot" | Re-run `selftest.py` with network; or delete `%USERPROFILE%\.cache\huggingface\hub\models--istupakov--parakeet-tdt-0.6b-v3-onnx` and retry. |
| Model load very slow (~90 s) | fp32 fallback active | Confirm int8 weights downloaded (`encoder-model.int8.onnx` in the HF cache); rerun selftest with network. |
| Text pastes but clipboard lost | Should not happen (old clipboard is restored), but clipboard managers can interfere | Note it to the user; harmless. |

## 7. Customization the user may ask for

- **Different hotkey**: edit the combo logic in `on_key()` in `dictation.py` (track the desired key names in `keys_down`). Keep the raw-hook pattern.
- **Disable filler cleanup** (keep "um"s): set `CLEANUP = False` in `dictation.py`.
- **Fix mangled names/brands**: edit `dictionary.txt` next to the app (created on first run, in the repo root from source). Primary syntax is one correct word/phrase per line (e.g. `FeWo direkt`); the app sound-matches and rewrites anything transcribed that sounds close, so wrong spellings are never enumerated. Only entries whose every word is 4+ letters are sound-matched. Advanced override: a `wrong -> right` line (with `|`-separated alternatives) still works and runs before the sound-matching. Case-insensitive, hot-reloads on save. It holds personal names, so it is gitignored; do not commit it.
- **English-only / other model**: change `MODEL_NAME` (see onnx-asr supported models).
- **Recover a lost dictation**: the last transcription is still in the clipboard (Ctrl+V), and every transcription is appended to `transcripts.log` in the repo root. Both by design; do not "clean up" the history write or re-add clipboard restore.

## 8. Maintain the user's dictionary for them (standing instruction)

If the user dictates their prompts to you through this app, you will sometimes SEE the transcription faults in the text you receive. When that happens, fix the dictionary for them:

- **When to act:** the user corrects a word themselves ("I said FeWo direkt, not Fible direct"), or a word in a dictated prompt is obviously a mishearing of a name/brand/term you know from context (the project is called NeoData and the prompt says "neo data" or "neodater"). Genuine confidence only; when unsure, ask in one short line instead of guessing.
- **What to do:** append the CORRECT spelling as its own line to `dictionary.txt` (next to the exe, or the repo root of this app's install; find it via the running app's location or ask once). Correct-word lines are enough in most cases; the sound-matcher handles unseen variants. Only use an explicit `wrong -> right` line when the mishearing changes the word count ("neo data -> NeoData") or the first consonant sound, which the sound-matcher cannot bridge.
- **Then tell the user, briefly:** one line like "Added 'FeWo direkt' to your dictation dictionary, it'll come out right from now on." Never do it silently.
- **Rules:** never remove or rewrite existing entries without being asked; the file is personal data, never commit or upload it; entries take effect on the user's next dictation automatically (hot reload, no restart).

## 9. What NOT to do

- Do not commit or upload `dictation.log`, `.venv/`, or the Hugging Face cache.
- Do not run multiple instances or register the scheduled task twice (use `-Force` semantics of the installer instead).
- Do not "fix" the two-process split by loading the model in the main process, which reintroduces the silent hook-death bug.
