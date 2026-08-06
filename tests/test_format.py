"""Unit tests for the structural formatting pass — pure logic, no pytest.

Run:  .venv\\Scripts\\python.exe tests\\test_format.py

Two halves to this. The POSITIVE tests show the readability wins: sentence
capitalization, spoken quotes, spoken enumerations, pause-driven paragraphs.
The NEGATIVE tests are the ones that actually matter — every rule here is
allowed to skip, but none of them is allowed to mangle text. If a rule is
unsure it must leave the words exactly as spoken, so most of what follows is
proving that it does.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dictation import (clean_text, format_text, join_segments,
                       _find_enumeration, PARA_GAP)


# --- sentence capitalization ------------------------------------------------

def test_capitalizes_after_sentence_end():
    got = format_text("I had a thought today. because Parakeet is free.")
    assert got == "I had a thought today. Because Parakeet is free.", got
    print("ok test_capitalizes_after_sentence_end")


def test_capitalizes_after_question_and_bang():
    got = format_text("Is that doable? maybe. it works! good.")
    assert got == "Is that doable? Maybe. It works! Good.", got
    print("ok test_capitalizes_after_question_and_bang")


def test_leaves_abbreviations_alone():
    got = format_text("Use a small model, e.g. one that fits in RAM.")
    assert got == "Use a small model, e.g. one that fits in RAM.", got
    print("ok test_leaves_abbreviations_alone")


def test_leaves_initials_alone():
    got = format_text("Ask J. smith about it.")
    assert got == "Ask J. smith about it.", got
    print("ok test_leaves_initials_alone")


def test_leaves_decimals_alone():
    got = format_text("Version 3.5 shipped.")
    assert got == "Version 3.5 shipped.", got
    print("ok test_leaves_decimals_alone")


# --- spoken quotes ----------------------------------------------------------

def test_spoken_quote():
    got = format_text("He said quote this is fine unquote and left.")
    assert got == 'He said "this is fine" and left.', got
    print("ok test_spoken_quote")


def test_spoken_quote_end_quote_variant():
    got = format_text("She wrote quote ship it end of quote yesterday.")
    assert got == 'She wrote "ship it" yesterday.', got
    print("ok test_spoken_quote_end_quote_variant")


def test_bare_quote_word_untouched():
    src = "Can you quote me a price for that work?"
    assert format_text(src) == src, format_text(src)
    print("ok test_bare_quote_word_untouched")


def test_absurdly_long_quote_span_rejected():
    src = ("I want to quote the docs " + "and some more filler words " * 12
           + "and then unquote it later.")
    assert format_text(src) == src, "over-long span must not be quoted"
    print("ok test_absurdly_long_quote_span_rejected")


# --- enumerations -----------------------------------------------------------

def test_enumeration_becomes_numbered_list():
    src = ("First off, I want to check the file. Secondly, we should run "
           "the tests. Lastly, we can ship it today.")
    got = format_text(src)
    assert got == ("1. I want to check the file.\n"
                   "2. We should run the tests.\n"
                   "3. We can ship it today."), got
    print("ok test_enumeration_becomes_numbered_list")


def test_enumeration_keeps_intro_paragraph():
    src = ("Here is the plan. First of all, we read the config file. "
           "Second, we validate every entry.")
    got = format_text(src)
    assert got == ("Here is the plan.\n\n"
                   "1. We read the config file.\n"
                   "2. We validate every entry."), got
    print("ok test_enumeration_keeps_intro_paragraph")


def test_number_one_two_form():
    src = ("Number one, the model loads too slowly. Number two, the hotkey "
           "sometimes misses.")
    got = format_text(src)
    assert got.startswith("1. The model loads too slowly."), got
    assert "2. The hotkey sometimes misses." in got, got
    print("ok test_number_one_two_form")


def test_paragraph_after_list_becomes_outro():
    # A pause after the last item ends the list; what follows is its own
    # paragraph, not a continuation of item N.
    src = ("First off, we read the config file. Secondly, we validate it "
           "properly.\n\nThat is the whole plan for today.")
    got = format_text(src)
    assert got == ("1. We read the config file.\n"
                   "2. We validate it properly.\n\n"
                   "That is the whole plan for today."), got
    print("ok test_paragraph_after_list_becomes_outro")


def test_pause_inside_item_stays_one_line():
    # Thinking mid-item must not split the list line in two.
    src = ("First off, we should check the log file\n\nand see what it says. "
           "Secondly, we restart the whole service.")
    got = format_text(src)
    assert got == ("1. We should check the log file and see what it says.\n"
                   "2. We restart the whole service."), got
    print("ok test_pause_inside_item_stays_one_line")


def test_lead_in_word_before_marker():
    src = ("So first off, I want to check the file. And secondly, we should "
           "run the tests today.")
    got = format_text(src)
    assert got == ("1. I want to check the file.\n"
                   "2. We should run the tests today."), got
    print("ok test_lead_in_word_before_marker")


def test_real_speech_enumeration():
    # Straight from a real dictation. Parakeet put a PERIOD after "first"
    # (not a comma), "Secondly" opens a sentence, and "thirdly" is buried
    # mid-clause after "and". All three have to be caught.
    src = ("So I walked away and the thing they did was first. run towards "
           "the big tree. Secondly, I headed home and thirdly I didn't do "
           "anything else.")
    got = format_text(src)
    assert got == ("So I walked away and the thing they did was:\n\n"
                   "1. Run towards the big tree.\n"
                   "2. I headed home\n"
                   "3. I didn't do anything else."), got
    print("ok test_real_speech_enumeration")


def test_strong_marker_mid_sentence():
    # "secondly"/"thirdly" have no non-enumerating use, so they count anywhere.
    src = ("First off, we check the log file. Then secondly we restart the "
           "whole service properly.")
    got = format_text(src)
    assert got == ("1. We check the log file.\n"
                   "2. We restart the whole service properly."), got
    print("ok test_strong_marker_mid_sentence")


def test_connective_stripped_from_item():
    src = ("Firstly I read the config file and secondly I validate every "
           "single entry.")
    got = format_text(src)
    assert got == ("1. I read the config file\n"
                   "2. I validate every single entry."), got
    print("ok test_connective_stripped_from_item")


def test_intro_gets_a_colon():
    src = ("Here is what we do first, read the config file. Second, validate "
           "every entry.")
    got = format_text(src)
    assert got.startswith("Here is what we do:\n\n1. "), got
    print("ok test_intro_gets_a_colon")


def test_adjectival_ordinals_are_not_a_list():
    # "the first thing" / "a second chance" are ordinary adjectives. Without
    # punctuation after the ordinal there was no pause, so there is no list.
    for src in [
        "The first thing I noticed was the color of it. The second thing was "
        "the smell of it.",
        "For the first time in years I felt good. I got a second chance at "
        "life here.",
        "He came in first in the race and second in the long jump event.",
        "Wait a second before you retry that. The first attempt already "
        "failed once.",
    ]:
        assert _find_enumeration(src) is None, src
        assert format_text(src) == src, format_text(src)
    print("ok test_adjectival_ordinals_are_not_a_list")


def test_finally_needs_a_pause():
    # "I finally got it working" is adverbial, not a list item.
    src = "First, I tried rebooting the machine. Then I finally got it working."
    assert _find_enumeration(src) is None
    assert format_text(src) == src, format_text(src)
    print("ok test_finally_needs_a_pause")


def test_single_marker_is_not_a_list():
    src = "First of all, I think we should just try it and see."
    assert format_text(src) == src, format_text(src)
    print("ok test_single_marker_is_not_a_list")


def test_counting_is_not_a_list():
    # Too few words per item: this is someone counting, not enumerating.
    src = "First, second, third."
    assert _find_enumeration(src) is None
    assert format_text(src) == src, format_text(src)
    print("ok test_counting_is_not_a_list")


def test_list_must_start_at_first():
    src = ("Second, we should check the logs carefully. Third, we restart "
           "the whole service.")
    assert _find_enumeration(src) is None
    assert format_text(src) == src, format_text(src)
    print("ok test_list_must_start_at_first")


def test_out_of_order_markers_rejected():
    src = ("First off, we look at the log file. Third, we give up on it "
           "entirely. Second, we try again later.")
    assert _find_enumeration(src) is None, "descending ordinals are not a list"
    print("ok test_out_of_order_markers_rejected")


def test_mid_sentence_ordinal_ignored():
    # "second" here is a unit of time, and it is not at a sentence start.
    src = ("First off, we should wait a second before retrying. Then we can "
           "look at what the second attempt returned.")
    assert _find_enumeration(src) is None, "only one real marker present"
    assert format_text(src) == src, format_text(src)
    print("ok test_mid_sentence_ordinal_ignored")


def test_enumeration_suppressed_without_breaks():
    src = ("First off, I want to check the file. Secondly, we should run "
           "the tests today.")
    got = format_text(src, allow_breaks=False)
    assert "\n" not in got, "continuous mode must stay on one line"
    assert got == src, got
    print("ok test_enumeration_suppressed_without_breaks")


# --- pause-driven paragraphs ------------------------------------------------

def test_long_pause_after_sentence_breaks_paragraph():
    got = join_segments(["That is the first thought.", "Now a new one."],
                        [0.0, PARA_GAP + 0.3])
    assert got == "That is the first thought.\n\nNow a new one.", got
    print("ok test_long_pause_after_sentence_breaks_paragraph")


def test_short_pause_does_not_break():
    got = join_segments(["That is the first thought.", "Now a new one."],
                        [0.0, 0.1])
    assert got == "That is the first thought. Now a new one.", got
    print("ok test_short_pause_does_not_break")


def test_pause_mid_sentence_never_breaks():
    # Long pause, but the previous segment did not finish a sentence. Thinking
    # mid-thought must never be promoted to a paragraph break.
    got = join_segments(["I was thinking that maybe we", "could try it later."],
                        [0.0, PARA_GAP + 1.0])
    assert got == "I was thinking that maybe we could try it later.", got
    print("ok test_pause_mid_sentence_never_breaks")


def test_breaks_disabled_joins_with_space():
    got = join_segments(["Done here.", "Next thing."],
                        [0.0, PARA_GAP + 1.0], allow_breaks=False)
    assert got == "Done here. Next thing.", got
    print("ok test_breaks_disabled_joins_with_space")


def test_missing_gaps_are_safe():
    got = join_segments(["One sentence here.", "Another one."], [])
    assert got == "One sentence here. Another one.", got
    print("ok test_missing_gaps_are_safe")


def test_empty_segments_skipped():
    got = join_segments(["Real text here.", "", "  ", "More text."],
                        [0.0, 0.0, 0.0, PARA_GAP + 0.5])
    assert got == "Real text here.\n\nMore text.", got
    print("ok test_empty_segments_skipped")


# --- clean_text stays newline-safe -----------------------------------------

def test_clean_text_single_line_unchanged_behaviour():
    # Regression guard: with no newlines the old behaviour must be identical.
    got = clean_text("um  so  the the cat sat , right ?")
    assert got == "So the cat sat, right?", got
    print("ok test_clean_text_single_line_unchanged_behaviour")


def test_clean_text_preserves_paragraphs():
    # clean_text keeps the break and drops the filler; capitalizing the new
    # sentence is format_text's job, which runs after it.
    got = clean_text("First thought here.\n\nuh second thought here.")
    assert got == "First thought here.\n\nsecond thought here.", got
    assert format_text(got) == "First thought here.\n\nSecond thought here."
    print("ok test_clean_text_preserves_paragraphs")


def test_clean_text_caps_blank_lines():
    got = clean_text("One.\n\n\n\nTwo.")
    assert got == "One.\n\nTwo.", got
    print("ok test_clean_text_caps_blank_lines")


def test_clean_text_strips_space_around_breaks():
    got = clean_text("One.   \n   Two.")
    assert got == "One.\nTwo.", got
    print("ok test_clean_text_strips_space_around_breaks")


# --- end-to-end ordering ----------------------------------------------------

def test_full_pipeline_order():
    # What the app actually does: join -> clean -> format.
    pieces = ["um so first off, I want to check the whole file.",
              "secondly, we should run the tests properly.",
              "that is basically it for now."]
    gaps = [0.0, 0.1, PARA_GAP + 0.5]
    out = format_text(clean_text(join_segments(pieces, gaps)))
    assert out.startswith("1. I want to check the whole file."), out
    assert "2. We should run the tests properly." in out, out
    assert "um" not in out.lower().split(), out
    print("ok test_full_pipeline_order")


def test_empty_input_is_safe():
    assert format_text("") == ""
    assert format_text(None) is None
    assert join_segments([], []) == ""
    print("ok test_empty_input_is_safe")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL FORMAT TESTS PASSED")

