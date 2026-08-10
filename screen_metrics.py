"""Live screen metrics for the status chip, and the reason they must be live.

THE BUG THIS EXISTS TO PREVENT

The chip vanished when this machine's fractional scale went 1.25 -> 1.0. It was
never a drawing bug: the window was mapped, sized and painted the whole time, at
`201x48+1435+1517` on a screen that is now 1080 tall. 1517 is 437px below the
bottom edge. The overlay had read the screen size once, at startup, into
`self.sw/self.sh` and every placement since was arithmetic on a desktop that no
longer existed:

    y = sh - h - s(96) = 1728 - 48 - 163 = 1517      # sh was the OLD 1728

WHY RE-READING TK'S OWN ANSWER IS NOT THE FIX

The obvious repair is to call `winfo_screenwidth()` at each show instead of
caching it. That does not work, and the reason is worth recording so nobody
"simplifies" this module away.

`winfo screenwidth` returns `WidthOfScreen(Tk_Screen(win))` — a field of Xlib's
`Screen` struct, filled in once from the connection setup. Xlib only refreshes
it when the client calls `XRRUpdateConfiguration()`, and Tk never does: measured
on this machine's actual binary, `libtk8.6.so` does not link `libXrandr` at all
and has zero undefined `XRR*` symbols. So Tk's idea of the screen size is frozen
for the life of the process no matter how often you ask it. A long-running Tk
app simply cannot see a resolution change through Tk.

A *fresh* X connection reads the current root geometry out of the connection
setup, so that is what `x_screen_size()` does. Measured cost: ~7.5ms, against a
handful of calls per minute (the chip is re-placed on status changes, not on
animation frames), and cached for a second on top of that.

SCALE, AND WHY 1.7 WAS NEVER A CONSTANT

The old hardcoded 1.7 was a calibration for one display configuration. At
fractional scale 1.25 mutter gave XWayland a 3072x1728 framebuffer for a
1536x864 logical desktop and the compositor squeezed it onto the 1920x1080
panel, so everything an X client drew reached the glass at 1920/3072 = 0.625 of
its pixel size. 1.7 * 0.625 = 1.0625 was the size that actually looked right.

At scale 1.0 there is no squeeze — the framebuffer *is* the panel — so the same
1.7 would have drawn the chip 70% oversized. The fix is to stop hardcoding the
product and derive it: measure the squeeze, then solve for the multiplier that
lands the chip at the size it has always appeared on glass.

    glass_factor = physical_panel_width / x_screen_width
    ui_scale     = TARGET_GLASS_SCALE / glass_factor

which reproduces 1.7 exactly at the old 1.25 scale and gives 1.0625 at 1.0, so
the chip keeps its familiar physical size across any scale change in either
direction. PARAKEET_UI_SCALE still overrides the result outright.
"""
import os
import time

# The chip's size on glass, in physical pixels per logical unit. This is the
# product the old hardcoded 1.7 was really expressing (1.7 * 0.625), pinned here
# so the chip's apparent size survives a scale change instead of tracking it.
TARGET_GLASS_SCALE = 1.0625

# What the scale was before this module existed. Kept as the fallback for any
# platform where the X/mutter measurements are unavailable (Windows, notably),
# so behaviour there is byte-for-byte unchanged.
LEGACY_SCALE = 1.7

_CACHE_TTL = 1.0
_cache = {"at": 0.0, "value": None}


def clamp_on_screen(x, y, w, h, sw, sh):
    """Pull a w*h rect fully inside a sw*sh screen. Pure, and the regression net.

    Every placement goes through this. Had it existed, the 1.25 -> 1.0 scale
    change would have cost the chip a slightly wrong position rather than making
    it disappear off the bottom of the desktop for hours.
    """
    x = 0 if w >= sw else max(0, min(int(x), sw - w))
    y = 0 if h >= sh else max(0, min(int(y), sh - h))
    return int(x), int(y)


def x_screen_size():
    """Current X root size, from a FRESH connection. None if X is unreachable.

    Fresh on purpose — see the module docstring. An existing connection's cached
    Screen struct cannot be refreshed without XRandR, which Tk does not link.
    """
    try:
        import ctypes
        import ctypes.util
        name = ctypes.util.find_library("X11")
        if not name:
            return None
        lib = ctypes.CDLL(name)
        lib.XOpenDisplay.restype = ctypes.c_void_p
        lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        for fn in ("XDefaultScreen", "XDisplayWidth", "XDisplayHeight"):
            getattr(lib, fn).restype = ctypes.c_int
        lib.XDefaultScreen.argtypes = [ctypes.c_void_p]
        lib.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
        display = lib.XOpenDisplay(None)
        if not display:
            return None
        try:
            screen = lib.XDefaultScreen(display)
            return (lib.XDisplayWidth(display, screen),
                    lib.XDisplayHeight(display, screen))
        finally:
            lib.XCloseDisplay(display)
    except Exception:
        return None


def physical_panel_size():
    """The panel's real pixel mode, straight from mutter. None if unavailable.

    xrandr cannot answer this: under XWayland the outputs it reports are already
    in X framebuffer coordinates, so at fractional scale it echoes the scaled
    size rather than the panel's. mutter's DisplayConfig knows the difference.
    """
    try:
        import gi  # noqa: F401
        from gi.repository import Gio
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        res = bus.call_sync(
            "org.gnome.Mutter.DisplayConfig", "/org/gnome/Mutter/DisplayConfig",
            "org.gnome.Mutter.DisplayConfig", "GetCurrentState", None, None,
            Gio.DBusCallFlags.NONE, 2000, None)
        _serial, monitors, _logical, _props = res.unpack()
        for monitor in monitors:
            for mode in monitor[1]:
                if mode[6].get("is-current"):
                    return int(mode[1]), int(mode[2])
    except Exception:
        return None
    return None


def glass_factor(x_size=None, panel_size=None):
    """How much the compositor shrinks an X pixel on its way to the panel.

    0.625 at the old 1.25 fractional scale (3072 wide framebuffer -> 1920 panel),
    1.0 at scale 1.0. None when either measurement is missing.
    """
    x_size = x_size if x_size is not None else x_screen_size()
    panel_size = panel_size if panel_size is not None else physical_panel_size()
    if not x_size or not panel_size or not x_size[0] or not panel_size[0]:
        return None
    factor = float(panel_size[0]) / float(x_size[0])
    if not 0.1 <= factor <= 10.0:
        return None
    return factor


def ui_scale(factor=None):
    """Logical-unit -> X-pixel multiplier for the chip.

    PARAKEET_UI_SCALE overrides outright, which is both the escape hatch and the
    behaviour that shipped before, so an existing override keeps working.
    """
    override = os.environ.get("PARAKEET_UI_SCALE")
    if override:
        try:
            return max(0.5, min(4.0, float(override)))
        except ValueError:
            pass
    factor = factor if factor is not None else glass_factor()
    if not factor:
        return LEGACY_SCALE
    return max(0.5, min(4.0, TARGET_GLASS_SCALE / factor))


def metrics(fallback_size=None, force=False):
    """(screen_w, screen_h, scale) as they are RIGHT NOW, cached for a second.

    `fallback_size` is the caller's own best guess (Tk's frozen answer, GDK's
    monitor geometry) and is used only when X cannot be reached at all.
    """
    now = time.monotonic()
    if not force and _cache["value"] and now - _cache["at"] < _CACHE_TTL:
        return _cache["value"]

    size = x_screen_size() or fallback_size
    if not size:
        return None
    value = (int(size[0]), int(size[1]),
             ui_scale(glass_factor(x_size=size)))
    _cache["at"] = now
    _cache["value"] = value
    return value
