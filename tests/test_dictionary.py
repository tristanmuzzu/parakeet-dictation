"""Unit tests for the personal correction dictionary — pure logic, no pytest.

Run:  .venv\\Scripts\\python.exe tests\\test_dictionary.py

The dictionary has two layers: sound-matching (you write only the correct word,
it fixes anything that sounds close) and explicit wrong -> right overrides for
the cases phonetics can't bridge. These tests cover both, plus the guardrails
that stop the sound-matching from firing on words that merely rhyme.

Every test that touches disk points dictation.DICTIONARY_FILE at a throwaway
temp file and clears the module's cached data, so nothing here ever reads or
writes the real dictionary.txt next to the app.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dictation


def _use_temp_file():
    """Point the module at a fresh temp dictionary and clear its cache. Starts
    with no file on disk, mirroring a first run."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    os.remove(path)
    dictation.DICTIONARY_FILE = path
    dictation._dict_rules = []
    dictation._dict_entries = []
    dictation._dict_mtime = None
    return path


def _write(path, text, when=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if when is not None:          # force a known mtime so hot-reload is testable
        os.utime(path, (when, when))


# --- sound_key: the phonetic reduction the whole thing rests on ---------------

def test_sound_key_units():
    sk = dictation.sound_key
    # Words that sound alike collapse to the same key.
    assert sk("fivo") == "fv", sk("fivo")
    assert sk("fewo") == "fv", sk("fewo")
    assert sk("fave") == "fv", sk("fave")
    assert sk("fiwo") == "fv", sk("fiwo")
    # w->v, c->k cross the German/English spelling of the same sound.
    assert sk("direct") == "drkt", sk("direct")
    assert sk("direkt") == "drkt", sk("direkt")
    # Multi-letter classes and umlauts.
    assert sk("phone") == sk("fone"), (sk("phone"), sk("fone"))
    # Umlauts fold to their base vowel: schön and schon reduce identically.
    assert sk("schön") == sk("schon"), (sk("schön"), sk("schon"))
    # No usable letters -> empty key, never crashes.
    assert sk("123 !!!") == ""
    assert sk("") == ""
    print("ok test_sound_key_units")


# --- sound-matching: the primary UX ------------------------------------------

def test_fuzzy_single_word_variants():
    path = _use_temp_file()
    _write(path, "Zeticle\n")
    assert dictation.apply_dictionary("i work at zeticle now") == "i work at Zeticle now"
    assert dictation.apply_dictionary("zeticel makes software") == "Zeticle makes software"
    print("ok test_fuzzy_single_word_variants")


def test_fuzzy_phrase_variants():
    path = _use_temp_file()
    _write(path, "FeWo direkt\n")
    # None of these spellings are listed anywhere — all corrected by sound.
    assert dictation.apply_dictionary("fivo direct") == "FeWo direkt"
    assert dictation.apply_dictionary("fiwo direkt") == "FeWo direkt"
    # "fave direkt" only clears the bar on the phrase mean (fave is a stretch
    # on its own, direkt is exact), which is exactly the point of averaging.
    assert dictation.apply_dictionary("fave direkt") == "FeWo direkt"
    print("ok test_fuzzy_phrase_variants")


def test_fuzzy_casing_preserved_from_entry():
    path = _use_temp_file()
    _write(path, "NeoData\n")
    # The transcription is lowercase; output takes the entry's exact casing.
    assert dictation.apply_dictionary("neodata") == "NeoData"
    assert dictation.apply_dictionary("neo data is not one word") == "neo data is not one word"
    print("ok test_fuzzy_casing_preserved_from_entry")


# --- guardrails: sound-matching must NOT overreach ----------------------------

def test_fuzzy_nonmatch_rhyme_guard():
    path = _use_temp_file()
    _write(path, "FeWo direkt\n")
    # "fever pitch" shares a first sound with FeWo but is nowhere near it.
    assert dictation.apply_dictionary("fever pitch") == "fever pitch"
    print("ok test_fuzzy_nonmatch_rhyme_guard")


def test_fuzzy_nonmatch_wrong_wordcount():
    path = _use_temp_file()
    _write(path, "FeWo direkt\n")
    # A 2-word entry never rewrites a lone word, and "direct flight" starts on
    # the wrong sound, so an ordinary sentence is left completely alone.
    assert dictation.apply_dictionary("book a direct flight") == "book a direct flight"
    print("ok test_fuzzy_nonmatch_wrong_wordcount")


def test_fuzzy_short_words_excluded():
    path = _use_temp_file()
    # "cat" is under the 4-letter bar, so it is NOT sound-matched (which would
    # wreck "bat", "cot", "kit"...). Short words need an explicit override.
    _write(path, "cat\n")
    assert dictation.apply_dictionary("the bat sat") == "the bat sat"
    assert dictation.apply_dictionary("kit and cot") == "kit and cot"
    print("ok test_fuzzy_short_words_excluded")


# --- explicit overrides: the advanced escape hatch ---------------------------

def test_explicit_override_rule():
    path = _use_temp_file()
    _write(path, "cloud code -> Claude Code\n")
    assert dictation.apply_dictionary("open CLOUD CODE now") == "open Claude Code now"
    assert dictation.apply_dictionary("Cloud Code rocks") == "Claude Code rocks"
    print("ok test_explicit_override_rule")


def test_explicit_runs_before_fuzzy():
    path = _use_temp_file()
    # Override maps a mangling phonetics can't reach; a separate sound entry also
    # loaded. Explicit fires first, sound-matching second, both land.
    _write(path, "cloud code -> Claude Code\nZeticle\n")
    assert dictation.apply_dictionary("cloud code by zeticel") == "Claude Code by Zeticle"
    print("ok test_explicit_runs_before_fuzzy")


def test_explicit_verbatim_right_side():
    path = _use_temp_file()
    # Backslashes / dollar signs in the right side stay literal.
    _write(path, r"backref -> a\1b C$ 100%" + "\n")
    assert dictation.apply_dictionary("backref") == r"a\1b C$ 100%"
    print("ok test_explicit_verbatim_right_side")


def test_explicit_longest_first():
    path = _use_temp_file()
    _write(path, "cloud -> CLOUD\ncloud code -> Claude Code\n")
    assert dictation.apply_dictionary("cloud code") == "Claude Code"
    assert dictation.apply_dictionary("the cloud is big") == "the CLOUD is big"
    print("ok test_explicit_longest_first")


# --- parsing, malformed input, counts ----------------------------------------

def test_parse_counts_words_and_rules():
    rules, entries, n_words, n_rules = dictation._parse_dictionary(
        "# a comment line\n"
        "\n"
        "   \n"
        "FeWo direkt\n"                 # word entry (both words >= 4 letters)
        "Zeticle\n"                     # word entry
        "cat\n"                         # word entry, too short to sound-match
        "cloud code -> Claude Code\n"   # explicit override
        "a | b -> c\n"                  # explicit override, two alternatives
    )
    assert n_words == 3, n_words           # FeWo direkt, Zeticle, cat all counted
    assert n_rules == 2, n_rules           # two override lines
    # cat is under the length bar, so only two entries are sound-matchable.
    assert len(entries) == 2, len(entries)
    # Two override lines, three left-hand patterns (a, b, cloud code).
    assert len(rules) == 3, len(rules)
    print("ok test_parse_counts_words_and_rules")


def test_parse_malformed_skipped():
    rules, entries, n_words, n_rules = dictation._parse_dictionary(
        "-> only a right side\n"        # empty left => skip
        "empty right side ->\n"         # empty right => skip
        "   |  | -> junk\n"             # left is only separators => skip
        "good -> fine\n"
    )
    assert n_rules == 1, n_rules
    assert len(rules) == 1
    assert rules[0][1] == "fine"
    assert n_words == 0 and entries == []
    print("ok test_parse_malformed_skipped")


def test_template_autocreate():
    path = _use_temp_file()
    assert not os.path.exists(path)
    dictation.ensure_dictionary_file()
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        body = f.read()
    assert "Teach Parakeet your words" in body
    assert "FeWo direkt" in body                 # correct-word example
    assert "-> Claude Code" in body              # override example documented too
    # A second call must never clobber a file the user has since edited.
    _write(path, "custom -> Custom\n")
    dictation.ensure_dictionary_file()
    with open(path, encoding="utf-8") as f:
        assert f.read() == "custom -> Custom\n"
    print("ok test_template_autocreate")


def test_hot_reload_on_mtime_change():
    path = _use_temp_file()
    _write(path, "Zeticle\n", when=time.time() - 10)
    assert dictation.apply_dictionary("zeticel") == "Zeticle"
    # Rewrite with a different entry and a newer mtime: picked up on the next
    # application, no restart.
    _write(path, "NeoData\n", when=time.time())
    assert dictation.apply_dictionary("neodata") == "NeoData"
    assert dictation.apply_dictionary("zeticel") == "zeticel"   # old entry gone
    print("ok test_hot_reload_on_mtime_change")


def test_apply_never_throws():
    path = _use_temp_file()
    _write(path, "Zeticle\n")
    assert dictation.apply_dictionary("") == ""
    assert dictation.apply_dictionary(None) is None
    print("ok test_apply_never_throws")


if __name__ == "__main__":
    test_sound_key_units()
    test_fuzzy_single_word_variants()
    test_fuzzy_phrase_variants()
    test_fuzzy_casing_preserved_from_entry()
    test_fuzzy_nonmatch_rhyme_guard()
    test_fuzzy_nonmatch_wrong_wordcount()
    test_fuzzy_short_words_excluded()
    test_explicit_override_rule()
    test_explicit_runs_before_fuzzy()
    test_explicit_verbatim_right_side()
    test_explicit_longest_first()
    test_parse_counts_words_and_rules()
    test_parse_malformed_skipped()
    test_template_autocreate()
    test_hot_reload_on_mtime_change()
    test_apply_never_throws()
    print("\nALL DICTIONARY TESTS PASSED")
