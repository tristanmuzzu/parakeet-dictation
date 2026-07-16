"""Unit tests for the continuous-mode command keywords — pure logic, no pytest.

Run:  .venv\\Scripts\\python.exe tests\\test_commands.py

detect_and_strip_command() looks at the END of the transcribed text: "send"
(and common ASR mishearings) means press Enter after pasting, a bare number
word means type that digit. Anywhere else in the sentence those are ordinary
words and must pass through untouched. The detector itself is mode-agnostic;
the continuous-mode gate lives at the call sites, so these tests only cover
the matching.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dictation import detect_and_strip_command


def test_send_variants():
    for word in ("send", "sent", "sand", "sendt", "sends", "enter",
                 "Send", "SEND"):
        text, cmd = detect_and_strip_command(f"fix the bug and {word}")
        assert cmd == "enter", f"{word!r} should map to enter, got {cmd!r}"
        assert text == "fix the bug and", f"command word not stripped: {text!r}"
    print("ok test_send_variants")


def test_send_with_trailing_punctuation():
    for tail in ("send.", "send,"):
        text, cmd = detect_and_strip_command(f"looks good {tail}")
        assert cmd == "enter", f"{tail!r} should still match"
        assert text == "looks good"
    print("ok test_send_with_trailing_punctuation")


def test_number_words():
    words = ["one", "two", "three", "four", "five",
             "six", "seven", "eight", "nine"]
    for i, word in enumerate(words, start=1):
        text, cmd = detect_and_strip_command(f"option {word}")
        assert cmd == str(i), f"{word!r} should map to {i}, got {cmd!r}"
        assert text == "option"
    print("ok test_number_words")


def test_bare_command_leaves_empty_text():
    # Saying just "send" pastes nothing and only presses Enter.
    text, cmd = detect_and_strip_command("send")
    assert cmd == "enter" and text == ""
    text, cmd = detect_and_strip_command("three")
    assert cmd == "3" and text == ""
    print("ok test_bare_command_leaves_empty_text")


def test_mid_sentence_is_not_a_command():
    for s in ("send the email to Bob",
              "enter the building on the left",
              "one of these days",
              "the parcel was sent yesterday"):
        text, cmd = detect_and_strip_command(s)
        assert cmd is None, f"false positive on {s!r}: {cmd!r}"
        assert text == s, f"text must be untouched, got {text!r}"
    print("ok test_mid_sentence_is_not_a_command")


def test_number_wins_over_send():
    # Number check runs first: "... send one" types the digit, keeps "send".
    text, cmd = detect_and_strip_command("please send one")
    assert cmd == "1" and text == "please send"
    print("ok test_number_wins_over_send")


def test_embedded_words_do_not_match():
    # \b guards: "weekend", "phone", "center" contain command substrings.
    for s in ("see you this weekend", "call me on the phone",
              "meet at the center"):
        text, cmd = detect_and_strip_command(s)
        assert cmd is None, f"substring false positive on {s!r}"
    print("ok test_embedded_words_do_not_match")


def test_empty_and_whitespace():
    assert detect_and_strip_command("") == ("", None)
    text, cmd = detect_and_strip_command("ship it and send")
    assert text.endswith("and") and not text.endswith(" "), \
        "trailing whitespace must be stripped with the command word"
    print("ok test_empty_and_whitespace")


if __name__ == "__main__":
    test_send_variants()
    test_send_with_trailing_punctuation()
    test_number_words()
    test_bare_command_leaves_empty_text()
    test_mid_sentence_is_not_a_command()
    test_number_wins_over_send()
    test_embedded_words_do_not_match()
    test_empty_and_whitespace()
    print("\nALL COMMAND TESTS PASSED")
