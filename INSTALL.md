# Install with a local agent

Read [AGENTS.md](AGENTS.md) first. Detect the operating system and check for an
existing installation, model cache, shortcut and running process. Preserve those
unless the user asked to change them. Do not start a second copy.

## Windows

Follow [WINDOWS-AGENT-SETUP.md](docs/WINDOWS-AGENT-SETUP.md).
Choose a public ZIP for ordinary use or source for inspection/customization.
Source requires Python 3.10 to 3.12; use `setup.ps1`, run `selftest.py`, then
launch the debug batch file. `SELFTEST_OK` proves recognition can load; a real
sentence appearing in Notepad proves shortcut, microphone and paste together.

The published ZIP can lag `main`. Linux source updates and optional model trials
are not part of Windows 1.2.1. Do not promise the ZIP contains source-only fixes.
Login startup is optional. Do not register its scheduled task just because you
read the runbook.

## Linux

Follow [LINUX.md](docs/LINUX.md). Use a graphical GNOME/XWayland desktop, a Python
environment that can import system GTK bindings, PortAudio and a configured
paste-injection backend. Keep the default multilingual Parakeet model.

Check `python3 dictation.py --send status` before launching or changing shortcuts.
A new installation may use the provided GNOME custom-binding script; an existing
installation may already have an extension owning its hotkey. Do not install
both over each other. Verify which keys are configured instead of guessing.

## Finish condition

1. Model reports ready and only one app owns `127.0.0.1:49731`.
2. User focuses a plain text editor, records a short sentence, and stops.
3. Capsule hides and recognized text appears once in the focused field.
4. Ctrl+V can still recover that text, and cancellation inserts nothing.
5. Report the actual shortcut, launch command, local data location, and any
   checks that need the user to interact with the microphone or desktop.

A passing unit test or successful command dispatch does not prove text was
inserted. Do not repeat an uncertain paste automatically or submit dictated text
to a coding agent as part of installation verification.

macOS is not implemented. Do not run Windows installers or Linux shortcut
scripts on another platform.
