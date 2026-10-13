# Verification, 2026-10-13

Current source passes 138 pytest checks and 21 subtests on Linux. Tests use
synthetic audio and isolated settings/history. They cover dictionary handling,
chunk boundaries, cancellation, worker errors, model selection and clipboard
handoff. The CI workflow runs these regressions without microphone access or
model downloads.

The README capsule image comes from a real microphone capture check. That take
was cancelled; it demonstrates recording UI, not recognition accuracy or paste.
The older Windows GIF is a workflow illustration, not a screen recording.

Linux source includes changes newer than the Windows 1.2.1 package. This pass
has not live-tested Windows or rebuilt its ZIP. GUI focus, hotkeys and actual
insertion still need verification on the user's target desktop.

The cached default Parakeet model also passed `tests/smoke_voice.py` on synthetic
speech: short and long recordings, a short final chunk after a hard boundary,
and paced streaming all retained the final words. One chunk completed before
stop; the remaining text finished 1.59 seconds afterward in this bounded run.
This measures recognition on that fixture, not microphone-to-paste latency.
