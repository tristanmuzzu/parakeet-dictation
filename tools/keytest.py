"""Diagnostic: shows the raw key names your keyboard produces.

Windows reports LOCALIZED key names (German: Ctrl='strg', Win='linke windows';
other locales differ). If the Ctrl+Win hotkey does not fire, run this, press
your keys, and check what names appear — then extend norm() in dictation.py.

Run:  ..\\.venv\\Scripts\\python.exe keytest.py   (from the tools folder)
      or  .venv\\Scripts\\python.exe tools\\keytest.py  (from the repo root)
"""
import time
import keyboard

count = {"n": 0}


def on(e):
    count["n"] += 1
    print(f"{e.event_type:5} {e.name}", flush=True)


keyboard.hook(on)
print("=" * 50)
print(" KEY TEST - type letters, then try Ctrl+Win.")
print(" Watching for 30 seconds...")
print("=" * 50, flush=True)
time.sleep(30)
print(f"TOTAL key events seen: {count['n']}")
