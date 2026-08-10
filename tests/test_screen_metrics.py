"""Guards for the defect that made the status chip vanish.

The chip did not fail to draw. It was mapped at 201x48+1435+1517 on a desktop
that had become 1920x1080 — 437px below the bottom edge — because the overlay
had cached the screen size at startup and the display scale changed underneath
it. These tests pin the two halves of the repair: placement is clamped to the
screen it is actually on, and the size multiplier is derived from the measured
compositor squeeze instead of being hardcoded for one configuration.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import screen_metrics  # noqa: E402


# --- the historical failure, as a test -----------------------------------

def test_stale_screen_math_cannot_push_the_chip_off_screen():
    """The exact numbers from the incident, clamped back onto the desktop."""
    scale, chip_w, chip_h = 1.7, 201, 48
    stale_w, stale_h = 3072, 1728          # what the process still believed
    live_w, live_h = 1920, 1080            # what the desktop had become

    unclamped_x = (stale_w - chip_w) // 2
    unclamped_y = stale_h - chip_h - int(round(96 * scale))
    assert (unclamped_x, unclamped_y) == (1435, 1517)      # reproduces the bug
    assert unclamped_y + chip_h > live_h                   # ... off the screen

    x, y = screen_metrics.clamp_on_screen(
        unclamped_x, unclamped_y, chip_w, chip_h, live_w, live_h)
    assert 0 <= x and x + chip_w <= live_w
    assert 0 <= y and y + chip_h <= live_h


def test_clamp_keeps_every_rect_fully_visible():
    sw, sh = 1920, 1080
    cases = [(-500, -500), (0, 0), (5000, 5000), (1900, 1070), (960, 540)]
    for x, y in cases:
        cx, cy = screen_metrics.clamp_on_screen(x, y, 200, 48, sw, sh)
        assert 0 <= cx <= sw - 200, (x, y, cx)
        assert 0 <= cy <= sh - 48, (x, y, cy)


def test_clamp_survives_a_chip_wider_than_the_screen():
    x, y = screen_metrics.clamp_on_screen(50, 50, 4000, 3000, 1920, 1080)
    assert (x, y) == (0, 0)


# --- scale derivation ------------------------------------------------------

def test_glass_factor_matches_both_measured_configurations():
    # fractional scale 1.25: 3072-wide XWayland framebuffer onto a 1920 panel
    assert screen_metrics.glass_factor((3072, 1728), (1920, 1080)) == 0.625
    # scale 1.0: the framebuffer is the panel
    assert screen_metrics.glass_factor((1920, 1080), (1920, 1080)) == 1.0


def test_ui_scale_reproduces_the_old_hardcoded_value_at_the_old_scale(monkeypatch):
    """1.7 was never a constant — it was TARGET_GLASS_SCALE / 0.625."""
    monkeypatch.delenv("PARAKEET_UI_SCALE", raising=False)
    assert round(screen_metrics.ui_scale(factor=0.625), 4) == 1.7


def test_ui_scale_shrinks_when_the_compositor_stops_squeezing(monkeypatch):
    """At scale 1.0 the old 1.7 would have drawn the chip 70% oversized."""
    monkeypatch.delenv("PARAKEET_UI_SCALE", raising=False)
    scale = screen_metrics.ui_scale(factor=1.0)
    assert round(scale, 4) == 1.0625
    assert scale < screen_metrics.LEGACY_SCALE


def test_chip_keeps_its_physical_size_across_a_scale_change(monkeypatch):
    """The point of the derivation: same pixels on glass, either direction."""
    monkeypatch.delenv("PARAKEET_UI_SCALE", raising=False)
    base = 28
    for factor in (0.625, 1.0, 0.5, 2.0):
        scale = screen_metrics.ui_scale(factor=factor)
        on_glass = base * scale * factor
        assert abs(on_glass - base * screen_metrics.TARGET_GLASS_SCALE) < 0.01


def test_unmeasurable_display_falls_back_to_the_shipped_behaviour(monkeypatch):
    """No X, no mutter (Windows, a headless box) -> exactly what shipped before."""
    monkeypatch.delenv("PARAKEET_UI_SCALE", raising=False)
    monkeypatch.setattr(screen_metrics, "glass_factor", lambda *a, **k: None)
    assert screen_metrics.ui_scale() == screen_metrics.LEGACY_SCALE


def test_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("PARAKEET_UI_SCALE", "2.5")
    assert screen_metrics.ui_scale(factor=1.0) == 2.5


def test_env_override_is_clamped_and_survives_garbage(monkeypatch):
    monkeypatch.setenv("PARAKEET_UI_SCALE", "99")
    assert screen_metrics.ui_scale(factor=1.0) == 4.0
    monkeypatch.setenv("PARAKEET_UI_SCALE", "not-a-number")
    assert round(screen_metrics.ui_scale(factor=1.0), 4) == 1.0625


def test_absurd_measurements_are_rejected_rather_than_trusted():
    assert screen_metrics.glass_factor((1920, 1080), (0, 0)) is None
    assert screen_metrics.glass_factor((1, 1), (1920, 1080)) is None   # 1920x squeeze


def test_missing_measurement_is_none_not_a_wrong_number(monkeypatch):
    """A None argument means 'go and measure'; an unmeasurable display means None."""
    monkeypatch.setattr(screen_metrics, "x_screen_size", lambda: None)
    monkeypatch.setattr(screen_metrics, "physical_panel_size", lambda: None)
    assert screen_metrics.glass_factor() is None
