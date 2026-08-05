"""Show what the text pipeline does to a dictation — no mic, no model, instant.

Run:  .venv\\Scripts\\python.exe tools\\formatcheck.py

Two jobs. First, it answers "is the formatting code actually on this machine?"
without needing you to dictate anything: if the import at the bottom fails or
prints OLD BUILD, your install predates the feature and needs updating.
Second, it shows before/after on realistic input so you can see exactly which
rules fire and tune them (FORMAT, PARA_GAP in dictation.py) with a fast loop.

Pass your own text to try it on something real:

    .venv\\Scripts\\python.exe tools\\formatcheck.py "first off this. secondly that."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dictation

# Segments as the app would receive them, with the pause (in seconds of leading
# silence) that preceded each one. This is the same shape stop_and_transcribe
# builds from AsrSession.ordered_pairs().
SAMPLE = [
    ("okay so I had a thought about the dictation tool today. it works well "
     "but the output is hard to re read.", 0.0),
    ("so first off, the transcription itself is genuinely accurate enough.", 1.2),
    ("secondly, the formatting is what actually makes it hard to scan.", 0.2),
    ("lastly, none of this should slow the thing down at all.", 0.9),
    ("as somebody once put it, quote make it readable unquote. that is the "
     "whole ask here.", 1.4),
]


def render(pairs):
    texts = [t for t, _ in pairs]
    gaps = [g for _, g in pairs]
    before = dictation.clean_text(" ".join(texts))
    joined = dictation.join_segments(texts, gaps, allow_breaks=dictation.FORMAT)
    after = dictation.clean_text(joined)
    if dictation.FORMAT:
        after = dictation.format_text(after, allow_breaks=True)
    return before, after


def main():
    if not hasattr(dictation, "format_text"):
        print("OLD BUILD: this install predates the formatting feature.")
        return 1

    if len(sys.argv) > 1:
        # One blob of text: no pause information, so paragraph breaks cannot
        # fire — only the text-level rules do.
        pairs = [(" ".join(sys.argv[1:]), 0.0)]
        print("(single argument: no pause data, so no paragraph breaks)\n")
    else:
        pairs = SAMPLE

    before, after = render(pairs)
    print(f"FORMAT={dictation.FORMAT}  CLEANUP={dictation.CLEANUP}  "
          f"PARA_GAP={dictation.PARA_GAP}s\n")
    print("=" * 70)
    print("BEFORE (what the old build pasted)")
    print("=" * 70)
    print(before)
    print()
    print("=" * 70)
    print("AFTER (what this build pastes)")
    print("=" * 70)
    print(after)
    print()
    if before == after:
        print("No change. Either FORMAT is off, or nothing in this text "
              "triggered a rule.")
    else:
        print("Formatting is active on this install.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
