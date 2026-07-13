# Parakeet Dictation

**Free, offline voice dictation for Windows.** Press **Ctrl+Win**, talk, press **Ctrl+Win** again — your words are typed into whatever app has focus (chat, browser, Word, IDE, anywhere).

No subscription, no cloud, no account. Speech recognition runs entirely on your laptop CPU using NVIDIA's open [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model — near state-of-the-art quality in 25 European languages (English, Italian, German, French, Spanish, ...) with automatic language detection and punctuation.

Built as a free replacement for paid dictation apps (Wispr Flow and friends).

## Setup — the easy way (let your AI do it)

Give this repository to any capable AI assistant (Claude Code, Codex, Cursor, ...) and say:

> Clone https://github.com/tristanmuzzu/parakeet-dictation and set it up for me, following its AGENTS.md.

[`AGENTS.md`](AGENTS.md) is a step-by-step runbook written for AI agents — setup, verification, and the known failure modes with fixes.

## Setup — by hand (3 steps, ~5 minutes)

Requirements: Windows 10/11, Python 3.10–3.12 ([download](https://www.python.org/downloads/), tick "Add to PATH"), a microphone.

1. **Download** this repo (green "Code" button → Download ZIP → extract, or `git clone`).
2. **Install**: right-click `setup.ps1` → *Run with PowerShell* (creates a private Python environment, installs dependencies).
3. **First run**: double-click **`Start Dictation (debug).bat`**. The first launch downloads the speech model (~460 MB, one time). When the small amber dot in the bottom-right corner disappears, it's ready.

Daily use: double-click **`Start Dictation.bat`** (runs silently in the background), or install auto-start (below) and never think about it again.

## Usage

| Keys | Action |
|---|---|
| **Ctrl + Win** | Start recording — a small "Transcribing" pill appears bottom-center |
| **Ctrl + Win** (again) | Stop — the recognized text is typed into the focused window |
| **Esc** | Cancel the current recording (nothing is inserted) |
| **Ctrl + Alt + Q** | Quit the app |

Speak in any supported language — it auto-detects, even sentence by sentence. Light cleanup is applied automatically (removes "um"/"uh"-type fillers, fixes duplicate words and spacing) — pure text processing, no AI rewriting of your words.

## Auto-start with Windows (optional)

Right-click `install-autostart.ps1` → *Run with PowerShell* (approve the admin prompt once). Dictation then starts at every login, ready ~15–30 seconds after boot. Remove anytime with `uninstall-autostart.ps1`.

## Troubleshooting

- **Ctrl+Win does nothing** → your keyboard may report localized key names (German layouts say `strg` instead of `ctrl`; already handled). Run `tools/keytest.py` to see your layout's names and extend `norm()` in `dictation.py`. Open an issue with your keytest output and locale — happy to add it.
- **"Model still loading…"** → wait for the amber corner dot to disappear (~15–30 s with the int8 model; ~90 s if it fell back to fp32).
- **Mic error** → check Windows Settings → Privacy → Microphone, and that a default input device exists.
- **Nothing pastes** → the target app must accept Ctrl+V paste (virtually all do).
- **Want a different hotkey?** → edit the `on_key` combo logic in `dictation.py` (see comments), or ask your AI to do it.

## How it works (for the curious)

Two processes: a featherweight one owns the global hotkey hook, the tiny overlay UI, and microphone capture; a separate worker process loads the ~600 MB ONNX model and serves recognition over a pipe. The split matters: Windows silently kills keyboard hooks belonging to CPU-pegged processes, which is exactly what a model load does. Recognition uses [onnx-asr](https://github.com/istupakov/onnx-asr) (int8 quantized weights, CPU only). Text is inserted via clipboard paste, and your previous clipboard is restored afterward.

## Privacy

Everything runs locally. Audio never leaves your machine. The only network access is the one-time model download from Hugging Face on first run.

## Licenses & credits

- **This code**: [MIT](LICENSE).
- **Speech model**: [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA, licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The model is **not** distributed with this repo — it downloads from Hugging Face (ONNX conversion: [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)) on first run under its own license.
- **Key dependencies**: [onnx-asr](https://github.com/istupakov/onnx-asr) (Apache-2.0), [sounddevice](https://github.com/spatialaudio/python-sounddevice) (MIT), [keyboard](https://github.com/boppreh/keyboard) (MIT), [pyperclip](https://github.com/asweigart/pyperclip) (BSD-3), NumPy (BSD-3).

Thanks to Jonah for the "just run Parakeet locally" nudge that started this.
