"""The status chip, drawn by GTK3 with a real ARGB visual.

WHY THIS EXISTS AND NOT THE TKINTER ONE

The tkinter overlay could not be made to look right, and no amount of tuning was
going to fix it. tk has no per-pixel transparency on X11: `-transparentcolor` is
a Windows-only attribute, so every pixel of the window is genuinely painted. A
"rounded" chip therefore has four painted corner blocks, and the only way to
hide them is to guess the colour of whatever is behind the window — which works
on a dark desktop and falls apart over anything bright. There is no shadow
either, because there is nothing to composite against.

GTK3 gives all of it away for free: ask the screen for an RGBA visual, set the
window app-paintable, and cairo's alpha channel reaches the compositor intact.
Rounded corners are actually round, the shadow is a real blurred alpha ramp, and
the chip floats over a white window as convincingly as over the wallpaper.

GTK3 rather than GTK4 on purpose: GTK4 removed both `set_keep_above()` and
`set_visual()`, and an always-on-top status chip needs the first while the whole
point of this rewrite needs the second.

X11 rather than Wayland on purpose, for the same reason pipsqueak forces it:
GNOME exposes no protocol letting an ordinary Wayland client raise itself, so
keep-above is silently a no-op there. Under XWayland `_NET_WM_STATE_ABOVE` is
honoured.

SCALING

XWayland renders X11 clients at 2x this machine's 1536x864 logical desktop and
the compositor scales the buffer back down to the 1920x1080 panel, so anything
drawn here is shrunk by 0.625 on its way to the glass. Nothing in the toolkit
reports that, so the multiplier is explicit and overridable with
PARAKEET_UI_SCALE. See ~/system/README.md, "XWayland reports a 3072x1728 screen".
"""
import math
import os
import sys as _sys
import json as _json

# Force XWayland before GDK initialises. As a Wayland-native client the chip
# cannot raise itself — GNOME exposes no protocol for it — so keep-above would
# be a silent no-op and the chip would vanish behind whatever has focus. Under
# X11 the _NET_WM_STATE_ABOVE hint is honoured. setdefault, so an explicit
# GDK_BACKEND from the launcher still wins.
os.environ.setdefault("GDK_BACKEND", "x11")

import gi  # noqa: E402

gi.require_version("Gtk", "3.0")
# Gdk must be pinned too, not just Gtk. It is imported by name below, and an
# unversioned Gdk resolves to the newest typelib on the machine (4.0 here)
# before Gtk 3 has loaded, which then fails with "Requiring namespace 'Gdk'
# version '3.0', but '4.0' is already loaded".
gi.require_version("Gdk", "3.0")
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
# Without this the draw handler blows up with "Couldn't find foreign struct
# converter for 'cairo.Context'" — the GTK side hands cairo objects across the
# GI boundary and needs the python3-gi-cairo bridge to translate them.
gi.require_foreign("cairo")
from gi.repository import Gdk, Gio, GLib, Gtk, Pango, PangoCairo  # noqa: E402

import cairo  # noqa: E402

try:
    import numpy as _np
except Exception:                                    # pragma: no cover
    _np = None


def _scale_default():
    """1.0, and that is not an oversight.

    The tkinter overlay needed a 1.7x fudge because tk sees the raw XWayland
    buffer (3072x1728) and draws into it, so everything came out at 0.625 of its
    intended size. GTK3 does not have that problem: it reports this monitor as
    1536x864 — the logical desktop — and the compositor handles the rest. Sizes
    here are therefore honest logical pixels. PARAKEET_UI_SCALE still overrides,
    for a different monitor or simple preference.
    """
    try:
        return max(0.5, min(4.0, float(os.environ.get("PARAKEET_UI_SCALE", "1.0"))))
    except ValueError:
        return 1.0


def _hex(rgb, alpha=1.0):
    rgb = rgb.lstrip("#")
    return (int(rgb[0:2], 16) / 255.0,
            int(rgb[2:4], 16) / 255.0,
            int(rgb[4:6], 16) / 255.0,
            alpha)


def _rounded(cr, x, y, w, h, r):
    r = min(r, h / 2, w / 2)
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


def _box_blur_alpha(surface, radius):
    """Blur an A8-ish ARGB surface's alpha in place with three box passes.

    Three box blurs approximate a gaussian closely enough that the eye cannot
    tell, and on a chip this small numpy finishes in well under a millisecond.
    """
    if _np is None or radius < 1:
        return surface
    surface.flush()
    w, h = surface.get_width(), surface.get_height()
    stride = surface.get_stride()
    buf = _np.ndarray(shape=(h, stride // 4, 4), dtype=_np.uint8,
                      buffer=surface.get_data())
    a = buf[:, :w, 3].astype(_np.float32)

    def box(src, r, axis):
        k = 2 * r + 1
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = _np.pad(src, pad, mode="constant")
        c = _np.cumsum(p, axis=axis)
        c = _np.concatenate([_np.zeros_like(_np.take(c, [0], axis=axis)), c], axis=axis)
        hi = _np.take(c, range(k, c.shape[axis]), axis=axis)
        lo = _np.take(c, range(0, c.shape[axis] - k), axis=axis)
        return (hi - lo) / float(k)

    r = max(1, int(radius / 3))
    for _ in range(3):
        a = box(a, r, 0)
        a = box(a, r, 1)
    buf[:, :w, 3] = _np.clip(a, 0, 255).astype(_np.uint8)
    surface.mark_dirty()
    return surface


class GtkOverlay:
    """Same surface area as the tkinter Overlay, so dictation.py barely changes.

    It also stands in for the tk root: `after`, `mainloop` and `destroy` are
    provided so the caller does not need to care which toolkit is running.
    """

    # Palette. The fill is genuinely translucent now, which is what makes the
    # chip read as floating rather than as a sticker.
    FILL = _hex("#0E1116", 0.88)
    BORDER = _hex("#FFFFFF", 0.10)
    INNER = _hex("#FFFFFF", 0.05)      # 1px top highlight, the usual "lit from above"
    TEXT = _hex("#E6EAF2", 0.96)
    SHADOW = (0.0, 0.0, 0.0, 0.55)

    FONT_CANDIDATES = ("Ubuntu Sans", "Ubuntu", "Cantarell", "DejaVu Sans", "Noto Sans")

    def __init__(self, scale=None):
        self.S = scale if scale else _scale_default()
        self.mode = None
        self._text = ""
        self._dot = "#f5b23e"
        self._pulse = 0.0
        self._pulse_dir = 1
        self._pump = None
        self._mini = False
        self._margin = 0
        self._warned_place = False
        self._chip = None
        # Compositor placement offset, measured once. See _measure_offset().
        self._off = (0, 0)
        self._off_known = False
        self._prev_focus = None

        # POPUP, and measured rather than assumed. GTK prints "temporary window
        # without parent, application will not be able to position it on screen"
        # and the warning is misleading: a POPUP lands exactly where move() puts
        # it, because it is override-redirect and the WM never gets a say. A
        # TOPLEVEL is the one mutter repositions — measured side by side, target
        # 695,720: POPUP landed on 695,720, TOPLEVEL on 0,0 standalone and
        # WM-centred inside the app.
        #
        # No set_type_hint() either. Adding NOTIFICATION to the POPUP was what
        # dragged the WM back into placement and parked the chip mid-screen.
        # An anchor toplevel that is realized and never shown.
        #
        # Measured, and it is the difference between the chip appearing and not:
        # standalone (where the harness happened to create another GTK toplevel
        # first) the overlay mapped at +0+0 and drew on top of everything. Inside
        # the app, where it was the process's only GTK window, the same code
        # mapped at +34+33 and drew underneath whatever had the screen. Giving
        # the process a prior toplevel reproduces the working case.
        self._anchor = Gtk.Window(type=Gtk.WindowType.TOPLEVEL)
        self._anchor.set_default_size(1, 1)
        self._anchor.set_decorated(False)
        self._anchor.set_skip_taskbar_hint(True)
        self._anchor.realize()

        self.win = Gtk.Window(type=Gtk.WindowType.TOPLEVEL)
        self.win.set_title("Parakeet Dictation")
        # The DND type hint is what makes GTK create a genuinely
        # override-redirect X window, and it is load-bearing for two separate
        # reasons that both bit during this rewrite:
        #
        #   focus — every other configuration STOLE THE KEYBOARD. Measured: the
        #     compositor reported the overlay as focused while it was showing and
        #     still focused after it was hidden. Dictation pastes into whatever
        #     is focused, so that quietly broke the entire feature.
        #   placement — a managed window is placed by mutter inside the work
        #     area, offset by the top bar and dock, and no amount of move()
        #     changes it.
        #
        # Measured across TOOLTIP / DND / SPLASHSCREEN / NOTIFICATION / none:
        # only DND produced a window the compositor does not manage, list, or
        # focus — and a DND window never rendered at all, raised or not. So the
        # trade is real and neither end of it is acceptable on its own:
        #
        #   managed window  -> renders, steals the keyboard
        #   DND window      -> never steals focus, never appears
        #
        # The window therefore stays managed (it has to, to be visible) and the
        # keyboard is handed straight back in _restore_focus(). Dictation pastes
        # into whatever is focused, so letting the chip keep focus silently broke
        # the whole feature — it was reported focused while showing AND after
        # hiding.
        screen = self.win.get_screen()
        visual = screen.get_rgba_visual()
        self.has_alpha = visual is not None
        if visual is not None:
            self.win.set_visual(visual)
        self.win.set_app_paintable(True)
        self.win.set_decorated(False)
        self.win.set_resizable(False)
        self.win.set_keep_above(True)
        self.win.set_accept_focus(False)
        self.win.set_focus_on_map(False)
        self.win.set_skip_taskbar_hint(True)
        self.win.set_skip_pager_hint(True)
        self.win.stick()

        self.area = Gtk.DrawingArea()
        self.area.connect("draw", self._on_draw)
        self.win.add(self.area)

        # Click-through: the chip must never eat a click meant for the app
        # underneath it. An empty input region is the X11 way to say that.
        self.win.connect("realize", self._on_realize)

        self.font_family = self._pick_font()
        geo = self._monitor_geometry()
        self.sw, self.sh = geo.width, geo.height
        self.ox, self.oy = geo.x, geo.y

        GLib.timeout_add(45, self._animate)

    # ---- helpers ---------------------------------------------------------
    def s(self, v):
        return int(round(v * self.S))

    def _pick_font(self):
        try:
            fam = {f.get_name().lower()
                   for f in self.win.get_pango_context().list_families()}
            for want in self.FONT_CANDIDATES:
                if want.lower() in fam:
                    return want
        except Exception:
            pass
        return "Sans"

    def _monitor_geometry(self):
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor() or display.get_monitor(0)
        return monitor.get_geometry()

    def _on_realize(self, widget):
        gw = widget.get_window()
        try:
            # Take the window manager out of the loop entirely.
            #
            # Measured, not assumed: without this the chip was mapped at roughly
            # screen centre (729,412) no matter how many times move() was called,
            # while GDK's own get_origin() cheerfully reported the position we
            # asked for (695,720) — it echoes the request, not reality. The
            # screenshot settled it. An override-redirect window is never managed,
            # so mutter cannot place it and our coordinates are final.
            gw.set_override_redirect(True)
        except Exception:
            pass
        try:
            region = cairo.Region()           # empty == no input anywhere
            gw.input_shape_combine_region(region, 0, 0)
        except Exception:
            pass

    # ---- layout ----------------------------------------------------------
    def _measure(self, text):
        layout = self.area.create_pango_layout(text)
        desc = Pango.FontDescription(self.font_family)
        # Absolute device pixels, not points: the point size would be reinterpreted
        # through Xft's DPI, which is exactly the indirection this overlay is
        # trying to get out from under.
        desc.set_absolute_size(self.s(10) * Pango.SCALE)
        desc.set_weight(Pango.Weight.MEDIUM)
        layout.set_font_description(desc)
        w, h = layout.get_pixel_size()
        return layout, w, h

    def _geometry_for(self, text):
        pad = self.s(14)
        dot_d = self.s(7)
        gap = self.s(9)
        _, tw, th = self._measure(text)
        cw = pad + dot_d + gap + tw + pad
        ch = max(self.s(28), th + self.s(12))
        margin = self.s(22)                    # room for the shadow to fall
        return cw, ch, margin

    # ---- painting --------------------------------------------------------
    def _on_draw(self, widget, cr):
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        if self._chip is None:
            return
        cx, cy, w, h, m = self._chip
        # Undo the compositor's placement offset so the chip lands where
        # the layout asked for it in screen coordinates.
        x, y = cx - self._off[0], cy - self._off[1]
        radius = h / 2.0

        # --- shadow: a real blurred alpha ramp, not a stack of outlines.
        # Rendered into a chip-sized scratch surface rather than the full-screen
        # window, so the blur cost stays constant instead of scaling with a
        # 1536x864 surface on every frame of the pulse animation.
        blur = self.s(9)
        dy = self.s(3)
        sw_, sh_ = w + 2 * m, h + 2 * m
        shadow = cairo.ImageSurface(cairo.FORMAT_ARGB32, sw_, sh_)
        sc = cairo.Context(shadow)
        _rounded(sc, m, m + dy, w, h, radius)
        sc.set_source_rgba(*self.SHADOW)
        sc.fill()
        _box_blur_alpha(shadow, blur)
        cr.set_source_surface(shadow, x - m, y - m)
        cr.paint()

        # --- chip
        _rounded(cr, x, y, w, h, radius)
        cr.set_source_rgba(*self.FILL)
        cr.fill_preserve()
        cr.set_line_width(max(1.0, self.S * 0.8))
        cr.set_source_rgba(*self.BORDER)
        cr.stroke()

        # 1px inner top highlight so the edge reads as lit rather than drawn
        cr.save()
        _rounded(cr, x + 1, y + 1, w - 2, h - 2, radius)
        cr.clip()
        grad = cairo.LinearGradient(0, y, 0, y + h * 0.5)
        grad.add_color_stop_rgba(0, *self.INNER)
        grad.add_color_stop_rgba(1, 1, 1, 1, 0.0)
        cr.set_source(grad)
        cr.paint()
        cr.restore()

        if self._mini:
            self._draw_dot(cr, x + w / 2.0, y + h / 2.0, self.s(4))
            return

        pad = self.s(14)
        dot_r = self.s(3.5)
        gap = self.s(9)
        cy = y + h / 2.0
        self._draw_dot(cr, x + pad + dot_r, cy, dot_r)

        layout, tw, th = self._measure(self._text)
        cr.move_to(x + pad + dot_r * 2 + gap, cy - th / 2.0)
        cr.set_source_rgba(*self.TEXT)
        PangoCairo.show_layout(cr, layout)

    def _draw_dot(self, cr, cx, cy, r):
        col = self._dot_colour()
        # soft halo, so the dot reads as a light rather than a circle
        halo = cairo.RadialGradient(cx, cy, 0, cx, cy, r * 2.6)
        halo.add_color_stop_rgba(0, *_hex(col, 0.42))
        halo.add_color_stop_rgba(1, *_hex(col, 0.0))
        cr.set_source(halo)
        cr.arc(cx, cy, r * 2.6, 0, 2 * math.pi)
        cr.fill()
        cr.set_source_rgba(*_hex(col, 1.0))
        cr.arc(cx, cy, r, 0, 2 * math.pi)
        cr.fill()

    def _dot_colour(self):
        if self.mode != "recording":
            return self._dot
        a = _hex("#5a2230")
        b = _hex(self._dot)
        t = self._pulse
        mix = tuple(a[i] + (b[i] - a[i]) * t for i in range(3))
        return "#%02x%02x%02x" % tuple(int(c * 255) for c in mix)

    def _animate(self):
        if self.mode == "recording":
            self._pulse += 0.08 * self._pulse_dir
            if self._pulse >= 1:
                self._pulse, self._pulse_dir = 1.0, -1
            elif self._pulse <= 0:
                self._pulse, self._pulse_dir = 0.0, 1
            if self.win.get_visible():
                self.win.queue_draw()
        return True

    # ---- public API (mirrors the tkinter Overlay) -------------------------
    def show_pill(self, text, color, mode):
        self.mode = mode
        self._text = text
        self._dot = color
        self._mini = False
        cw, ch, margin = self._geometry_for(text)
        self._margin = margin
        self._show_chip(cw, ch, margin,
                        (self.sw - cw) // 2,
                        self.sh - ch - self.s(72))

    def show_mini(self, color):
        self.mode = "loading"
        self._dot = color
        self._mini = True
        d = self.s(18)
        margin = self.s(18)
        self._margin = margin
        self._show_chip(d, d, margin,
                        self.sw - d - self.s(28),
                        self.sh - d - self.s(56))

    def _show_chip(self, cw, ch, margin, cx, cy):
        """Draw the chip at (cx, cy) inside a full-screen transparent window.

        This is the second design, and the first one is worth recording because
        it looks so obviously correct: size the window to the chip and move() it
        where you want. It does not work here. Under this compositor the chip
        was mapped at roughly screen centre no matter what — gtk_window_move
        before the map, after the map, on idle, on a timer, gdk_window_move on
        the GdkWindow itself, POPUP versus TOPLEVEL, with and without a type
        hint, with and without a transient parent, and finally
        set_override_redirect(True) at realize. GDK's get_origin() reported the
        requested position every time, which is why the bug survived so long; a
        screenshot showed the chip sitting in the middle of the screen.

        So the window no longer has a position worth arguing about: it covers the
        monitor, it is transparent, and its input region is empty so clicks pass
        straight through. Where the chip appears is now a cairo translation,
        which no window manager gets a vote on.
        """
        self._chip = (cx, cy, cw, ch, margin)
        if not self.win.get_visible():
            self._remember_focus()
        W, H = self.sw, self.sh
        self.area.set_size_request(W, H)
        self.win.set_size_request(W, H)
        self.win.resize(W, H)
        self.win.show_all()
        self.win.move(self.ox, self.oy)
        # An override-redirect window is nobody's responsibility, which is the
        # point — but it also means nothing stacks it for us. raise_() puts it on
        # top without asking for focus, which is exactly the pair we need:
        # visible over a fullscreen video, and never stealing the keyboard from
        # the window dictation is about to paste into.
        # Re-assert keep-above AFTER the map, not just at construction. Set only
        # on the unmapped window it did not stick: the compositor listed the
        # overlay without the "above" flag (pipsqueak, which sets it the same
        # way but stays mapped, kept its), so the chip was mapped and drawing
        # underneath whatever window happened to be in front — present in the
        # window list, invisible on screen.
        self.win.set_keep_above(True)
        gw = self.win.get_window()
        if gw is not None:
            gw.raise_()
        self.win.queue_draw()
        # Three attempts, spread across the map. A single early restore lost the
        # race: mutter assigns focus to the newly mapped window *after* the map
        # completes, so restoring at 60ms handed focus back before it had been
        # taken, and the chip ended up holding the keyboard for its whole
        # lifetime. Measured: with one attempt, focus during showing was the
        # overlay; with these three it stays on the user's window throughout.
        for delay in (150, 400, 900):
            GLib.timeout_add(delay, self._restore_focus)
        if not self._off_known:
            GLib.timeout_add(250, self._measure_offset)

    def _shell_call(self, method, args=None):
        """Talk to the migration-helpers GNOME Shell extension.

        It is already this machine's answer for the three things the compositor
        refuses ordinary clients — screenshots, window geometry, focus — so it is
        the right place to ask for the two things this overlay needs.
        """
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        return bus.call_sync("org.tristan.MigrationHelpers",
                             "/org/tristan/MigrationHelpers",
                             "org.tristan.MigrationHelpers", method,
                             args, None, Gio.DBusCallFlags.NONE, 2000, None)

    def _remember_focus(self):
        """Note which window owns the keyboard, before we take it away."""
        try:
            payload = self._shell_call("ListWindows").unpack()[0]
            if isinstance(payload, str):
                payload = _json.loads(payload)
            for win in payload:
                if win.get("focused") and win.get("title") != "Parakeet Dictation":
                    self._prev_focus = int(win["id"])
                    return
        except Exception:
            pass

    def _restore_focus(self):
        """Give the keyboard back to whatever had it.

        Showing the chip has to map a managed window, and mapping a managed
        window takes focus here no matter what accept_focus / focus_on_map say.
        Handing it straight back is the difference between dictation pasting into
        the user's editor and pasting into an invisible status chip.
        """
        if not self._prev_focus:
            return False
        try:
            self._shell_call("ActivateWindow", GLib.Variant("(u)", (self._prev_focus,)))
        except Exception:
            try:
                self._shell_call("ActivateWindow",
                                 GLib.Variant("(i)", (self._prev_focus,)))
            except Exception as exc:
                print(f"[overlay] could not restore focus: {exc}",
                      file=_sys.stderr, flush=True)
        # Handing focus back ACTIVATES that window, which also raises it — and
        # the chip promptly disappeared underneath it. Raising ourselves again
        # afterwards restores the stacking without taking the keyboard back:
        # raise and focus are separate operations on X11, which is the only
        # reason this combination works at all.
        if self.win.get_visible():
            self.win.set_keep_above(True)
            gw = self.win.get_window()
            if gw is not None:
                gw.raise_()
        return False

    def _measure_offset(self):
        """Learn where the compositor actually put the window, once.

        Every API GDK offers reports the position we asked for rather than the
        one we got — get_origin, get_root_origin, get_position and
        get_frame_extents all said (0,0) while the compositor had the window at
        (34,33), which is the top bar's 33px and the dock's 34px. mutter
        constrains the window to the work area, and GDK's own get_workarea()
        does not know that under XWayland (it returns the full monitor).

        Rather than hard-code 34,33, ask the one component that can actually
        see: the migration-helpers shell extension, which runs inside gnome-shell
        and is already this machine's answer for screenshots, window geometry and
        focus. If it is unavailable the offset stays (0,0) and the chip simply
        sits a few pixels low — a graceful loss, not a broken overlay.
        """
        self._off_known = True
        try:
            bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            res = bus.call_sync("org.tristan.MigrationHelpers",
                                "/org/tristan/MigrationHelpers",
                                "org.tristan.MigrationHelpers", "ListWindows",
                                None, None, Gio.DBusCallFlags.NONE, 2000, None)
            payload = res.unpack()[0]
            if isinstance(payload, str):
                payload = _json.loads(payload)
            for win in payload:
                if win.get("title") == "Parakeet Dictation":
                    dx = int(win.get("x", 0)) - self.ox
                    dy = int(win.get("y", 0)) - self.oy
                    if (dx, dy) != (0, 0):
                        self._off = (dx, dy)
                        print(f"[overlay] compositor placement offset {self._off}; "
                              f"compensating", file=_sys.stderr, flush=True)
                        self.win.queue_draw()
                    break
        except Exception as exc:                      # pragma: no cover
            print(f"[overlay] could not measure placement offset ({exc}); "
                  f"chip may sit a few px low", file=_sys.stderr, flush=True)
        return False

    def _place_legacy(self, W, H, x, y):
        """Size, then move on both sides of the map, then once more when idle.

        GTK warns that a "temporary window without parent" cannot be positioned,
        and it half means it: a single move() around show_all() was honoured in a
        standalone test and silently ignored inside the app, which left the chip
        parked in the middle of the screen. Moving before the map, after the map,
        and again on the next main-loop iteration is what actually pins it — the
        last one lands after the WM has finished its own placement pass.
        """
        self.area.set_size_request(W, H)
        self.win.set_size_request(W, H)
        self.win.resize(W, H)
        self.win.move(x, y)
        self.win.show_all()
        self.win.move(x, y)
        self.win.queue_draw()

        def _pin():
            self.win.move(x, y)
            gw = self.win.get_window()
            if gw is not None:
                gw.move(x, y)          # belt and braces on the GdkWindow itself
            return False
        GLib.idle_add(_pin)
        GLib.timeout_add(50, _pin)

        def _audit():
            gw = self.win.get_window()
            got = tuple(gw.get_origin()[1:]) if gw is not None else None
            if got != (x, y) and not self._warned_place:
                self._warned_place = True
                print(f"[overlay] placement wanted {(x, y)} got {got} "
                      f"size={W}x{H} monitor={self.sw}x{self.sh}+{self.ox}+{self.oy}",
                      file=_sys.stderr, flush=True)
            return False
        GLib.timeout_add(300, _audit)

    def hide(self):
        self.mode = None
        self.win.hide()
        GLib.timeout_add(60, self._restore_focus)

    # ---- tk-root shims ---------------------------------------------------
    def after(self, ms, fn):
        GLib.timeout_add(ms, lambda: (fn(), False)[1])

    def mainloop(self):
        Gtk.main()

    def destroy(self):
        try:
            self.win.destroy()
        finally:
            Gtk.main_quit()

    # convenience so callers can treat it like the old (root, overlay) pair
    @property
    def root(self):
        return self
