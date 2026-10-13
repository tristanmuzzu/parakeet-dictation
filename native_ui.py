"""Native Linux recording capsule. It never accepts focus.

The small override-redirect XWayland window is the entire interface.
Transcription and clipboard delivery happen in the background.
"""
import math
import os
import time

os.environ.setdefault("GDK_BACKEND", "x11")
import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
gi.require_foreign("cairo")
from gi.repository import Gtk, Gdk, GLib, Pango, PangoCairo
import cairo

LIME = "#d8f58a"
AMBER = "#f4cc8a"


def device_label(name):
    """Keep audio-server routing identifiers out of the listening capsule."""
    if "__Mic1__" in name or name == "Mic1":return "Built-in digital microphone"
    if "__Mic2__" in name or name == "Mic2":return "Built-in stereo microphone"
    if name.startswith("bluez_"):return "Bluetooth microphone"
    return name


def color(cr, value, alpha=1):
    cr.set_source_rgba(*(int(value[i:i+2], 16)/255 for i in (1, 3, 5)), alpha)


def rounded(cr, x, y, w, h, r):
    cr.new_sub_path()
    for cx, cy, a in ((x+w-r,y+r,-90), (x+w-r,y+h-r,0),
                      (x+r,y+h-r,90), (x+r,y+r,180)):
        cr.arc(cx, cy, r, math.radians(a), math.radians(a+90))
    cr.close_path()


def bird(cr, x, y, scale=1):
    cr.save()
    cr.translate(x, y)
    cr.scale(scale, scale)
    color(cr, LIME)
    cr.move_to(6, 22); cr.line_to(6, 12)
    cr.curve_to(6, 6, 10, 3, 16, 3)
    cr.curve_to(21, 3, 25, 7, 25, 12)
    cr.line_to(25, 14); cr.line_to(30, 14); cr.line_to(25, 19)
    cr.line_to(19, 19); cr.line_to(19, 22)
    cr.curve_to(19, 27, 15, 30, 10, 30)
    cr.line_to(6, 30); cr.line_to(11, 24); cr.line_to(8, 24)
    cr.close_path(); cr.fill()
    color(cr, "#24311e"); cr.arc(19, 10, 1.6, 0, 7); cr.fill()
    cr.move_to(9, 13); cr.line_to(16, 19); cr.line_to(9, 22)
    cr.set_line_width(1.5); cr.stroke(); cr.restore()


class NativeUI:
    def __init__(self, store, actions, sample):
        self.store, self.actions, self.sample = store, actions, sample
        self.root = self
        self.mode = "idle"
        self.message = "Ready when you are"
        self.subtitle = "Microphone off"
        self.started = time.monotonic()
        self.levels = [0.0]*13
        self.generation = 0
        self.width, self.height = 398, 118
        self.win = Gtk.Window(type=Gtk.WindowType.POPUP)
        self.win.set_title("Parakeet listening capsule")
        self.win.set_decorated(False)
        self.win.set_accept_focus(False)
        self.win.set_focus_on_map(False)
        self.win.set_keep_above(True)
        self.win.set_skip_taskbar_hint(True)
        self.win.set_skip_pager_hint(True)
        self.win.set_app_paintable(True)
        visual = self.win.get_screen().get_rgba_visual()
        if visual:
            self.win.set_visual(visual)
        self.area = Gtk.DrawingArea()
        self.area.set_size_request(self.width, self.height)
        self.area.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.area.connect("draw", self._draw)
        self.area.connect("button-press-event", self._click)
        self.win.add(self.area)
        self._animation_id = GLib.timeout_add(50, self._animate)

    def act(self, name, *args):
        fn = self.actions.get(name)
        if fn:
            return fn(*args)

    def after(self, ms, fn):
        def once():
            fn()
            return False
        return GLib.timeout_add(ms, once)

    def mainloop(self):
        Gtk.main()

    def destroy(self):
        GLib.source_remove(self._animation_id)
        self.win.destroy()
        Gtk.main_quit()

    def _position(self):
        display = self.win.get_display()
        seat = display.get_default_seat()
        _, x, y = seat.get_pointer().get_position()
        monitor = display.get_monitor_at_point(x, y)
        area = monitor.get_workarea()
        px=area.x+(area.width-self.width)//2
        self.win.move(px,max(area.y, area.y+area.height-self.height-36))

    def show(self, mode, message=None, subtitle=None):
        if mode in ("inserted", "recover", "idle"):
            self.hide()
            return
        self.generation += 1
        changed = mode != self.mode
        self.mode = mode
        defaults = {"recording": ("Listening", "Microphone on"),
                    "processing": ("Finishing your words", "Microphone off"),
                    "error": ("Could not transcribe", "Try recording again"),
                    "loading": ("Preparing your voice model", "Runs on this device"),
                    "idle": ("Ready when you are", "Microphone off")}
        self.message, self.subtitle = defaults.get(mode, defaults["error"])
        if message: self.message = message
        if subtitle: self.subtitle = subtitle
        self.area.get_accessible().set_name(self.message + ". " + self.subtitle)
        if mode == "recording" and changed:
            self.started = time.monotonic(); self.levels = [0.0]*13
        self._position()
        self.win.show_all()
        gw=self.win.get_window()
        gw.raise_()
        gw.input_shape_combine_region(cairo.Region(cairo.RectangleInt(16,10,self.width-32,64)),0,0)
        self.area.queue_draw()
        if mode == "error":
            generation = self.generation
            self.after(2500, lambda: self.hide() if generation == self.generation else None)

    def show_pill(self, text, tint, mode):
        self.show("error" if mode == "status" else mode, text)

    def show_mini(self, tint):
        self.show("loading")

    def copy_to_clipboard(self, text):
        # Own the XWayland clipboard on this GTK connection. wl-copy can leave
        # an Electron destination seeing a stale/empty X11 selection even when
        # Wayland readback already contains the right text.
        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        clipboard.set_text(text, -1)
        clipboard.store()
        return True

    def hide(self):
        self.generation += 1
        self.mode = "idle"
        self.win.hide()

    def _animate(self):
        if not self.win.get_visible(): return True
        if self.mode == "recording":
            info = self.sample()
            level = min(1, max(0, info.get("level", 0)*9))
            self.levels = self.levels[1:]+[level]
            self.subtitle = device_label(info.get("device", "Microphone on"))
            if info.get("continuous"): self.message = "Listening · hands-free"
            self.area.queue_draw()
        elif self.mode == "processing" and not self.store.settings["reduced_motion"]:
            self.area.queue_draw()
        return True

    def _text(self, cr, value, x, y, size, tint, width=None):
        layout = PangoCairo.create_layout(cr)
        font = Pango.FontDescription("Ubuntu Sans")
        font.set_absolute_size(size*Pango.SCALE)
        layout.set_font_description(font)
        layout.set_text(value, -1)
        if width:
            layout.set_width(width*Pango.SCALE)
            layout.set_ellipsize(Pango.EllipsizeMode.END)
        color(cr, tint); cr.move_to(x, y); PangoCairo.show_layout(cr, layout)

    def _draw(self, _, cr):
        cr.set_operator(cairo.OPERATOR_SOURCE); cr.set_source_rgba(0,0,0,0); cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        x, y, w, h = 16, 10, self.width-32, 64
        for i in range(12, 0, -1):
            rounded(cr,x-i,y+4-i,w+2*i,h+2*i,22+i)
            cr.set_source_rgba(.03,.07,.025,.012); cr.fill()
        rounded(cr,x,y,w,h,22)
        gradient = cairo.LinearGradient(0,y,0,y+h)
        gradient.add_color_stop_rgba(0,.19,.23,.18,.98)
        gradient.add_color_stop_rgba(1,.10,.14,.11,.98)
        cr.set_source(gradient); cr.fill_preserve()
        color(cr,"#96845c" if self.mode == "error" else "#627152",.9)
        cr.set_line_width(1); cr.stroke()
        bird(cr,29,27,.85)
        self._text(cr,self.message,69,23,13,"#eff6e6",145 if self.mode=="recording" else 204)
        self._text(cr,self.subtitle,69,44,10,"#b9c8aa",145 if self.mode=="recording" else 204)
        if self.mode == "recording":
            for i, level in enumerate(self.levels):
                height=3+level*23
                rounded(cr,223+i*4.4,42-height/2,2.7,height,1.3)
                color(cr,LIME); cr.fill()
            seconds=int(time.monotonic()-self.started)
            self._text(cr,f"{seconds//60:02}:{seconds%60:02}",286,34,11,"#d2dfc3")
        elif self.mode == "processing":
            # Text layout leaves a current point on the Cairo context. Start
            # independent paths so arc() cannot connect the spinner to it.
            cr.save()
            cr.set_line_width(2)
            cr.set_line_cap(cairo.LINE_CAP_ROUND)
            color(cr,"#78866c")
            cr.new_path();cr.arc(297,42.5,9,0,2*math.pi);cr.stroke()
            phase=0 if self.store.settings["reduced_motion"] else time.monotonic()*3
            color(cr,LIME)
            cr.new_path();cr.arc(297,42.5,9,phase,phase+1.8);cr.stroke()
            cr.restore()
        tint = AMBER if self.mode == "error" else LIME
        rounded(cr,331,27,31,31,10);color(cr,tint);cr.fill()
        color(cr,"#293b1d")
        if self.mode == "recording":
            rounded(cr,342,38,9,9,2);cr.fill()
        else:
            # Geometric icon: centered on the same point as the stop square,
            # independent of font bearings and baseline metrics.
            cr.save()
            cr.set_line_width(1.8)
            cr.set_line_cap(cairo.LINE_CAP_ROUND)
            cx, cy, radius = 346.5, 42.5, 4
            cr.new_path()
            cr.move_to(cx-radius,cy-radius);cr.line_to(cx+radius,cy+radius)
            cr.move_to(cx+radius,cy-radius);cr.line_to(cx-radius,cy+radius)
            cr.stroke()
            cr.restore()
        hints={"recording":"Ctrl + Alt + Space to finish", "processing":"Click × to cancel",
               "error":"Ctrl + Alt + Space to try again",
               "loading":"Your audio stays on this device", "idle":"Ctrl + Alt + Space to start"}
        if self.mode in hints:
            hint=hints[self.mode]
            rounded(cr,71,84,256,23,11);color(cr,"#20291d",.93);cr.fill()
            self._text(cr,hint,84,89,10,"#d0dcc4",232)
        return False

    def _click(self, _, event):
        if 331 <= event.x <= 362 and 27 <= event.y <= 58:
            if self.mode == "recording": self.act("toggle")
            elif self.mode == "processing": self.act("cancel")
            elif self.mode in ("recover","error"): self.hide()
            elif self.mode == "idle": self.act("toggle")
            else: self.hide()
        return True
