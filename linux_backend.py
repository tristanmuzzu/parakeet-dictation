"""
Linux/Wayland backend for Parakeet Dictation.

Windows uses a low-level keyboard hook (the `keyboard` package) for the global
hotkey and synthesises Ctrl+V into the focused window. Neither works on a GNOME
Wayland session:

  - `keyboard` reads /dev/input directly, so it needs root, and even as root a
    Wayland compositor does not route global keys to it.
  - No client can synthesise input into another client's window. That is the
    security model, not a gap.

So the two halves are replaced:

  Hotkey  -> GNOME custom keybindings (see install-gnome-hotkeys.sh). GNOME owns
             the key grab and runs `dictation.py --send toggle`, which talks to
             the already-running app over the loopback port it was using purely
             as a single-instance lock. No global key capture in our process.

  Paste   -> wl-copy puts the text on the clipboard, then ydotool synthesises
             Ctrl+V through /dev/uinput. uinput is a kernel-level virtual input
             device, so it sits *below* the compositor and is not subject to the
             restriction above.

ydotool caveat: because it injects at kernel level it has no idea which window
is focused. It types wherever focus happens to be, which is the same behaviour
the Windows build has in practice.

Why the clipboard carries the text and ydotool only ever presses Ctrl+V:
`ydotool type` is NOT usable on this machine. It writes raw evdev keycodes, which
the compositor then maps through the active XKB layout, while ydotool's
character-to-keycode table assumes US QWERTY. This session's layout is German
QWERTZ, so typed text arrives transposed -- measured 2026-08-08, "ydo1" landed in
the widget as "zdo1" -- and ydotool still exits 0, so the corruption is silent.
Putting the payload on the clipboard sidesteps the whole mapping problem: the
only keycodes injected are ctrl (29) and v (47), and v sits in the same physical
place on QWERTZ as on QWERTY. Verified on that layout by pasting
"yz-äö.,;:/ YZ zyzzyva 42" and reading it back out of the target widget
byte-identical. Do not "simplify" this into `ydotool type`.
"""

import logging
import os
import shutil
import socket
import subprocess
import threading
import time

log = logging.getLogger("dictation")

LOCK_HOST, LOCK_PORT = "127.0.0.1", 49731

# Linux input event codes (include/uapi/linux/input-event-codes.h). ydotool's
# `key` verb takes CODE:STATE pairs; names are not portable across its versions,
# raw codes are.
KEY = {
    "ctrl": 29, "shift": 42, "alt": 56, "enter": 28, "esc": 1, "v": 47,
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
}


def is_linux() -> bool:
    return os.name == "posix" and os.uname().sysname == "Linux"


# --------------------------------------------------------------------------
# clipboard
# --------------------------------------------------------------------------
def _clipboard_cmd():
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        return ["wl-copy"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard"]
    return None


def copy_to_clipboard(text: str) -> bool:
    cmd = _clipboard_cmd()
    if not cmd:
        log.error("no clipboard tool found - install wl-clipboard (Wayland) or xclip")
        return False
    try:
        # wl-copy forks a server to own the selection; do not wait on it.
        subprocess.run(cmd, input=text.encode("utf-8"), check=True, timeout=5)
        return True
    except Exception:
        log.exception("clipboard copy failed")
        return False


# --------------------------------------------------------------------------
# synthetic keys
# --------------------------------------------------------------------------
def _ydotool(args) -> bool:
    exe = shutil.which("ydotool")
    if not exe:
        log.error("ydotool not found - synthetic paste unavailable")
        return False
    env = dict(os.environ)
    # ydotoold runs as a *system* service (it needs /dev/uinput, which a user
    # service cannot open unless the session was started after the uinput group
    # was added - a logout/reboot we do not want to require). The daemon chowns
    # this socket to us, so only this user can inject.
    env.setdefault("YDOTOOL_SOCKET", "/run/ydotoold.socket")
    try:
        r = subprocess.run([exe] + args, env=env, capture_output=True, timeout=10)
        if r.returncode != 0:
            log.error("ydotool failed rc=%s err=%s", r.returncode,
                      r.stderr.decode("utf-8", "replace").strip()[:200])
            return False
        return True
    except Exception:
        log.exception("ydotool invocation failed")
        return False


def _chord(*names) -> bool:
    """Press keys in order, release in reverse - the order a real chord has."""
    codes = [KEY[n] for n in names if n in KEY]
    if len(codes) != len(names):
        log.error("unmapped key in chord %s", names)
        return False
    seq = [f"{c}:1" for c in codes] + [f"{c}:0" for c in reversed(codes)]
    return _ydotool(["key"] + seq)


def _atspi_insert(text: str) -> bool:
    """Hand the text straight to the focused widget through AT-SPI EditableText.

    This is the fallback, not the primary path, and the reason is measured rather
    than assumed. AT-SPI needs no keyboard and no focus race and even works while
    the screen is locked, which makes it a genuinely better mechanism *when it
    works* -- but on this desktop it often does not:

      - Electron apps (Claude Desktop, where a lot of this dictation lands)
        publish a tree with no editable text nodes at all, so there is nothing to
        write into.
      - Apps started without the accessibility bridge never appear on the bus.
      - Walking the whole desktop for the focused widget costs ~1s, most of it
        inside gnome-shell, which is never the paste target and is skipped here.

    So Ctrl+V stays the fast path and this catches the cases where injection is
    unavailable (no ydotoold, locked screen). Returns True only after reading the
    text back out of the widget: insert_text returning true is not proof.
    """
    try:
        import gi
        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi
    except Exception as e:
        log.warning("AT-SPI unavailable (%s: %s)", type(e).__name__, e)
        return False
    try:
        Atspi.init()
        deadline = time.time() + 2.0
        budget = [3000]

        def focused_editable(node, depth):
            if depth > 32 or budget[0] <= 0 or time.time() > deadline:
                return None
            budget[0] -= 1
            try:
                text_iface = node.get_text_iface()
                editable = node.get_editable_text_iface()
                if text_iface is not None and editable is not None and \
                        node.get_state_set().contains(Atspi.StateType.FOCUSED):
                    return node
                count = node.get_child_count()
            except Exception:
                return None
            for i in range(count):
                try:
                    child = node.get_child_at_index(i)
                except Exception:
                    continue
                if child is not None:
                    hit = focused_editable(child, depth + 1)
                    if hit is not None:
                        return hit
            return None

        desktop = Atspi.get_desktop(0)
        target = None
        for i in range(desktop.get_child_count()):
            app = desktop.get_child_at_index(i)
            if app is None or (app.get_name() or "") == "gnome-shell":
                continue
            target = focused_editable(app, 0)
            if target is not None:
                break
        if target is None:
            log.warning("AT-SPI found no focused editable widget to write into")
            return False

        text_iface = target.get_text_iface()
        caret = Atspi.Text.get_caret_offset(text_iface)
        before = Atspi.Text.get_character_count(text_iface)
        if not target.insert_text(caret if caret >= 0 else before, text, len(text)):
            log.warning("AT-SPI insert_text returned false")
            return False
        time.sleep(0.15)
        after_count = Atspi.Text.get_character_count(text_iface)
        after = Atspi.Text.get_text(text_iface, 0, after_count) if after_count else ""
        if text not in after:
            log.warning("AT-SPI insert_text reported success but the widget does "
                        "not contain the text - treating as failure")
            return False
        log.info("AT-SPI delivered %d chars to a %s widget", len(text),
                 target.get_role_name())
        return True
    except Exception:
        log.exception("AT-SPI delivery failed")
        return False


def paste_text(text: str) -> None:
    # The clipboard is written first and deliberately left holding the text: it
    # is the user's manual recovery path if every automatic route below fails.
    if not copy_to_clipboard(text):
        return
    # Give the compositor a moment to take ownership of the new selection before
    # the paste asks for it back; without this the target can read the *old*
    # clipboard. Same reason the Windows path sleeps here.
    time.sleep(0.12)
    if _chord("ctrl", "v"):
        return
    # Injection is unavailable (ydotoold down, or /dev/uinput out of reach).
    # AT-SPI needs no keyboard at all, so it can still land the text.
    log.warning("synthetic paste failed - trying AT-SPI")
    if _atspi_insert(text):
        return
    log.warning("delivery failed on every route - the text is on the clipboard, "
                "paste it manually with Ctrl+V")


def press_command_key(cmd: str) -> None:
    time.sleep(0.15)
    if cmd in ("enter", "return"):
        _chord("enter")
    elif cmd in KEY:
        _chord(cmd)
    else:
        log.error("unmapped command key %r", cmd)


# --------------------------------------------------------------------------
# command channel - reuses the single-instance lock socket
# --------------------------------------------------------------------------
def serve_commands(sock, handlers: dict) -> None:
    """Accept one-shot connections on the existing lock socket and dispatch.

    Runs on a daemon thread. The socket is already bound and listening; on
    Windows it is only ever used as a lock, so this is additive.
    """
    def loop():
        while True:
            try:
                conn, _ = sock.accept()
            except OSError:
                return                      # socket closed on shutdown
            try:
                conn.settimeout(2)
                cmd = conn.recv(64).decode("utf-8", "replace").strip()
                fn = handlers.get(cmd)
                log.info("command channel: %r -> %s", cmd, "ok" if fn else "unknown")
                if fn:
                    fn()
                    conn.sendall(b"ok\n")
                else:
                    conn.sendall(b"unknown\n")
            except Exception:
                log.exception("command channel error")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    threading.Thread(target=loop, name="cmd-channel", daemon=True).start()
    log.info("command channel listening on %s:%s", LOCK_HOST, LOCK_PORT)


def send_command(cmd: str) -> int:
    """Client side of --send. Returns a process exit code."""
    try:
        with socket.create_connection((LOCK_HOST, LOCK_PORT), timeout=3) as s:
            s.sendall(cmd.encode("utf-8"))
            reply = s.recv(32).decode("utf-8", "replace").strip()
        if reply == "ok":
            return 0
        print(f"dictation: server did not understand {cmd!r}")
        return 2
    except (ConnectionRefusedError, OSError):
        print("dictation: app is not running")
        return 1
